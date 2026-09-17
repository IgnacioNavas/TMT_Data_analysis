# ─────────────────────────────────────────────────────────────────────────────────────────
# limma statistics for the diaPASEF dataset (hme1_diaPASEF / hme1_lfq).
#
# Helper functions for notebooks/01_preprocessing/limma_for_pvalues_diapasef.rmd. One fit per
# cell line, EGF only, and ~ 0 + group by default: the runs are independent injections loaded
# in randomised order, so nothing pairs a timepoint with a particular replicate's control.
#
#   source(file.path(PROJECT_ROOT, "src", "r_utils.R"))          # stage, checkpoint, ...
#   source(file.path(PROJECT_ROOT, "src", "limma_common.R"))     # sort_timepoints, ...
#   source(file.path(PROJECT_ROOT, "src", "limma_diapasef.R"))
#
# ⚠️ Do NOT source this together with src/limma_tmt.R in one session. Both define
# `parse_abs_columns`, `deduplicate_controls`, `read_dataset` and `limma_output_path` with
# different field conventions (`replicate` here, `plex` there); the second source() silently
# wins. The check below turns that into a visible warning.
#
# ⚠️ Every argument that defaulted to a notebook global in the .rmd (CONTROL, CONDITIONS,
# CELL_LINES, MIN_RESIDUAL_DF, MIN_CONTROL_OBS, NORMALIZE, BLOCK_ON_REPLICATE, MAX_SITES,
# INPUT_FILE, OUT_SUFFIX, VERBOSE) is now an explicit argument with a literal default, and the
# notebook passes its configuration in. Progress reporting reads the `tmt.verbose` option —
# see src/r_utils.R.
#
# Requires: limma, and src/r_utils.R + src/limma_common.R sourced first.
# ─────────────────────────────────────────────────────────────────────────────────────────

if (exists(".limma_module", envir = globalenv()) &&
    !identical(get(".limma_module", envir = globalenv()), "diapasef")) {
  warning("limma_diapasef.R: src/limma_tmt.R is already sourced in this session. ",
          "The two define the same function names with different field conventions; ",
          "restart R and source only the one the notebook needs.", call. = FALSE)
}
.limma_module <- "diapasef"


limma_output_path <- function(input_path,
                              out_suffix = "_limma_pvalues.tsv") {
  # Build the output path of the limma table, next to the input file.
  #
  # The result is written into the folder the input was read from, so a run on the development
  # sample lands in data/ and a full run lands in Experiment/hme1_diaPASEF/Data/Processed/. The
  # name reuses the input's stem, so the date prefix carries over
  # (20260818_..._transformed_nolimma.tsv -> 20260818_..._limma_pvalues.tsv).
  #
  # Args:
  #   input_path: path of the processed .tsv being analysed.
  #   out_suffix: suffix appended to the input stem, including the extension.
  #
  # Returns:
  #   Absolute path of the .tsv to write. Never equal to the input path.

  stem <- basename(input_path)
  stem <- sub("_transformed_nolimma\\.tsv$", "", stem)
  stem <- sub("_log2_processed\\.tsv$", "", stem)
  stem <- sub("\\.tsv$", "", stem)               # fallback if the input is named differently

  output_path <- file.path(dirname(input_path), paste0(stem, out_suffix))
  if (!dir.exists(dirname(output_path))) {
    stop("limma_output_path: output folder '", dirname(output_path), "' does not exist.")
  }
  if (identical(output_path, input_path)) {
    stop("limma_output_path: refusing to overwrite the input file '", input_path, "'.")
  }
  output_path
}


parse_abs_columns <- function(column_names,
                              cell_line,
                              conditions) {
  # Parse the log2:abs column names of one cell line into their component fields.
  #
  # The cell line is matched as the whole first field (the pattern is anchored and the field
  # cannot contain "_"), so a name that prefixes another — BRAFS151A1 vs BRAFS151A2 — cannot pull
  # in both, and the MIX_* control channels are never selected.
  #
  # Args:
  #   column_names: character vector of all column names in the dataset.
  #   cell_line: cell line prefix to keep, e.g. "WT".
  #   conditions: character vector of condition names, e.g. c("EGF").
  #
  # Returns:
  #   data.frame with one row per log2:abs column and the columns
  #   `column`, `condition`, `timepoint`, `replicate`.

  pattern <- paste0("^", cell_line, "_log2:abs_(",
                    paste(conditions, collapse = "|"),
                    ")_([^_]+)_(r[0-9]+)$")

  matched <- grep(pattern, column_names, value = TRUE)
  if (length(matched) == 0) {
    stop("parse_abs_columns: no log2:abs columns found for cell line '", cell_line, "'.")
  }

  fields <- regmatches(matched, regexec(pattern, matched))

  data.frame(column    = matched,
             condition = vapply(fields, `[`, character(1), 2),
             timepoint = vapply(fields, `[`, character(1), 3),
             replicate = vapply(fields, `[`, character(1), 4),
             stringsAsFactors = FALSE)
}


deduplicate_controls <- function(df,
                                 column_info,
                                 conditions,
                                 control_timepoints = c("full", "starve")) {
  # Remove the duplicated control columns that appear once per condition arm.
  #
  # NO-OP ON THIS DATASET: with a single condition there is nothing to de-duplicate, and the
  # function only adds the `group` column. It is kept unchanged from the TMT notebook so that the
  # two files stay comparable and so that adding INS / EGFnINS arms later needs no new code — in
  # the TMT files `full` and `starve` are one physical channel written out under every condition,
  # and feeding all copies to lmFit would be pseudo-replication.
  #
  # Args:
  #   df: the full dataset, used only to compare the duplicated columns.
  #   column_info: data.frame returned by parse_abs_columns().
  #   conditions: character vector of condition names; the first one is kept.
  #   control_timepoints: timepoints that are shared across condition arms.
  #
  # Returns:
  #   column_info restricted to unique samples, with an added `group` column giving the
  #   experimental group of each sample (control timepoints keep their bare name).

  keep_condition <- conditions[1]

  for (timepoint in control_timepoints) {
    for (replicate in unique(column_info$replicate)) {
      reference <- column_info$column[column_info$condition == keep_condition &
                                      column_info$timepoint == timepoint &
                                      column_info$replicate == replicate]
      if (length(reference) != 1) next

      for (other_condition in conditions[-1]) {
        duplicate <- column_info$column[column_info$condition == other_condition &
                                        column_info$timepoint == timepoint &
                                        column_info$replicate == replicate]
        if (length(duplicate) != 1) next

        if (!isTRUE(all.equal(df[[reference]], df[[duplicate]]))) {
          stop("deduplicate_controls: '", duplicate, "' is not identical to '", reference,
               "'. The control channels are not duplicates in this dataset — ",
               "the design must be rebuilt before fitting.")
        }
      }
    }
  }

  is_duplicate_control <- column_info$timepoint %in% control_timepoints &
                          column_info$condition != keep_condition
  unique_info <- column_info[!is_duplicate_control, , drop = FALSE]

  unique_info$group <- ifelse(unique_info$timepoint %in% control_timepoints,
                              unique_info$timepoint,
                              paste(unique_info$condition, unique_info$timepoint, sep = "_"))

  unique_info
}


site_coverage <- function(df,
                          samples,
                          control = "starve") {
  # Count, per site, how many of one cell line's runs observed it.
  #
  # This is the diaPASEF replacement for the TMT `n:reps` column. It cannot be read off a stored
  # column: `n:reps` does not exist in the processed diaPASEF table, and DIA missingness is
  # per run rather than per plex, so the number of replicates a site was seen in does not describe
  # which runs those were.
  #
  # Args:
  #   df: the dataset, containing the log2:abs columns listed in `samples`.
  #   samples: data.frame returned by deduplicate_controls(), one row per run, with the columns
  #     `column` and `group`.
  #   control: name of the control group, taken as an argument rather than read from the global so
  #     it cannot drift away from the `control` the fit is using.
  #
  # Returns:
  #   data.frame with one row per site of `df` and the columns
  #     n_obs          — runs of this cell line in which the site was quantified,
  #     n_control_obs  — runs of the control group in which it was quantified,
  #     n_groups_obs   — timepoint groups observed at least once,
  #     n_fgroups_obs  — the same, counting only the groups the omnibus F is built from
  #       (control + stimulation timepoints, i.e. everything except `full`). A site needs all of
  #       them to get an F; `full` is irrelevant to it, so counting `full` would under-report.

  observed <- !is.na(as.matrix(df[, samples$column, drop = FALSE]))

  group_levels <- unique(as.character(samples$group))
  per_group <- vapply(group_levels,
                      function(group) {
                        rowSums(observed[, samples$group == group, drop = FALSE]) > 0
                      },
                      logical(nrow(df)))
  per_group <- matrix(per_group, nrow = nrow(df),
                      dimnames = list(NULL, group_levels))   # keep the shape for a 1-site frame

  f_groups <- setdiff(group_levels, "full")

  data.frame(n_obs         = rowSums(observed),
             n_control_obs = rowSums(observed[, samples$group == control, drop = FALSE]),
             n_groups_obs  = rowSums(per_group),
             n_fgroups_obs = rowSums(per_group[, f_groups, drop = FALSE]))
}


replicate_effect_report <- function(M,
                                    samples,
                                    label = "") {
  # Measure what a replicate block term would absorb, whatever design is actually being fitted.
  #
  # The diaPASEF runs are independent injections in randomised order, so blocking on the replicate
  # is a choice rather than a description of the experiment (see "The design"). This makes the
  # choice empirical: it fits ~ 0 + group + replicate on the same matrix and tests, per site,
  # H0: every replicate offset is zero. Two numbers come out — how often the offsets are
  # detectable, and how large the largest one typically is, in log2 units directly comparable to a
  # fold change.
  #
  # It is deliberately run in both modes. With BLOCK_ON_REPLICATE = FALSE it reports the systematic
  # error the unblocked fit is exposed to on unevenly observed sites; with TRUE it reports what the
  # two spent degrees of freedom bought.
  #
  # Args:
  #   M: the expression matrix being fitted, sites in rows and runs in columns.
  #   samples: data.frame with one row per run, carrying the `group` and `replicate` factors.
  #   label: tag printed with the report, e.g. the cell line.
  #
  # Returns:
  #   Invisibly, a list with n_tested, n_significant and median_abs_offset, or NULL if the design
  #   has no replicate term to test (a single replicate).

  design <- model.matrix(~ 0 + group + replicate, data = samples)
  colnames(design) <- make.names(colnames(design))
  replicate_coefs <- grep("^replicate", colnames(design), value = TRUE)

  if (length(replicate_coefs) == 0 || qr(design)$rank < ncol(design)) {
    cat("     replicate effect: not estimable (single replicate or rank-deficient design)\n")
    return(invisible(NULL))
  }

  fit <- muffle_warnings(lmFit(M, design), "replicate-effect lmFit")
  fit_r <- muffle_warnings(eBayes(contrasts.fit(fit,
                                                makeContrasts(contrasts = replicate_coefs,
                                                              levels = design)),
                                  trend = TRUE,
                                  robust = TRUE),
                           "replicate-effect F")

  fdr <- p.adjust(fit_r$F.p.value, method = "BH")

  # Largest absolute offset per site, in log2 units: directly comparable to a fold change.
  offsets <- suppressWarnings(apply(abs(fit_r$coefficients), 1, max, na.rm = TRUE))
  offsets[!is.finite(offsets)] <- NA_real_

  n_tested      <- sum(!is.na(fdr))
  n_significant <- sum(fdr < 0.05, na.rm = TRUE)

  cat(sprintf(paste0("     replicate effect%s: offsets non-zero at FDR < 0.05 in %d of %d sites (%.1f%%)",
                     " | median largest |offset| = %.3f log2\n"),
              if (nzchar(label)) paste0(" (", label, ")") else "",
              n_significant, n_tested,
              if (n_tested > 0) 100 * n_significant / n_tested else NA_real_,
              median(offsets, na.rm = TRUE)))
  flush.console()

  invisible(list(n_tested = n_tested,
                 n_significant = n_significant,
                 median_abs_offset = median(offsets, na.rm = TRUE)))
}


read_dataset <- function(file_path,
                         dataset_name,
                         cell_lines) {
  # Read the processed dataset and report everything needed to explain a slow or wrong run.
  #
  # The file is read once and reused by every cell-line fit and by both verification steps. The
  # diagnostics answer, in order: is the file the size we think it is, did it parse into the
  # expected shape, are the intensity columns numeric, are missing values stored as NA rather than
  # as 0, and how far apart are the per-run medians (the normalisation exposure).
  #
  # Args:
  #   file_path: path to the processed .tsv.
  #   dataset_name: dataset key, used in the printed report.
  #   cell_lines: cell lines whose per-run median span is reported.
  #
  # Returns:
  #   The dataset as a data.frame, with column names untouched.

  size_mb <- file.size(file_path) / 1e6
  cat(sprintf("[%s]      file: %.0f MB  %s\n",
              format(Sys.time(), "%H:%M:%S"), size_mb, basename(file_path)))
  flush.console()

  df <- stage(sprintf("read.delim (%.0f MB)", size_mb),
              read.delim(file_path,
                         sep = "\t",
                         check.names = FALSE,
                         na.strings = c("NA", "NaN", ""),
                         stringsAsFactors = FALSE))

  cat(sprintf("[%s]      parsed: %d rows x %d columns\n",
              format(Sys.time(), "%H:%M:%S"), nrow(df), ncol(df)))

  abs_columns <- grep("_log2:abs_", colnames(df), value = TRUE)
  non_numeric <- abs_columns[!vapply(df[abs_columns], is.numeric, logical(1))]
  if (length(non_numeric) > 0) {
    cat(sprintf("      WARNING: %d log2:abs column(s) did not parse as numeric, e.g. %s\n",
                length(non_numeric), paste(head(non_numeric, 3), collapse = ", ")))
  }

  # Missingness must be NA. If a fillna(0) reached this file, the zeros are read as real
  # intensities: limma then sees a complete matrix, takes its fast path, and returns confident
  # nonsense. This is the defect recorded in clustering_method_decision.md §1.
  numeric_abs <- abs_columns[vapply(df[abs_columns], is.numeric, logical(1))]
  if (length(numeric_abs) > 0) {
    values    <- as.matrix(df[numeric_abs])
    n_missing <- sum(is.na(values))
    n_zero    <- sum(values == 0, na.rm = TRUE)
    cat(sprintf("      log2:abs cells: %.1f%% NA, %.1f%% exactly zero  (%d runs)\n",
                100 * n_missing / length(values),
                100 * n_zero / length(values),
                length(numeric_abs)))
    if (n_zero > 0.01 * length(values)) {
      cat("      WARNING: many exact zeros in log2:abs — missing values may have been\n",
          "               written as 0 instead of NA. Fix before trusting any p-value.\n", sep = "")
    }

    # Normalisation exposure: how far apart the per-run medians are. NORMALIZE = "none" fits the
    # data as stored, so whatever span is printed here is inherited by every log fold change.
    run_medians <- apply(values, 2, median, na.rm = TRUE)
    cat(sprintf("      per-run log2:abs medians: span %.2f log2 over all runs\n",
                diff(range(run_medians, na.rm = TRUE))))
    for (cell_line in cell_lines) {
      own <- run_medians[startsWith(names(run_medians), paste0(cell_line, "_"))]
      if (length(own) > 1) {
        cat(sprintf("        %-14s %5.2f log2 across its %d runs\n",
                    cell_line, diff(range(own, na.rm = TRUE)), length(own)))
      }
    }
  }

  report_memory(sprintf("after reading %s", dataset_name))
  flush.console()
  df
}


run_limma_cell_line <- function(df,
                                cell_line,
                                conditions,
                                control = "starve",
                                min_residual_df = 2,
                                min_control_obs = 2,
                                normalize = "none",
                                block_on_replicate = FALSE,
                                max_sites = Inf) {
  # Fit the limma model on one cell line and return its table of statistics.
  #
  # Every stimulation timepoint is contrasted against the control timepoint, using ~ 0 + group
  # (the default: the runs are independent injections in randomised order, so nothing pairs a
  # timepoint with a particular replicate's control) or ~ 0 + group + replicate when
  # block_on_replicate is TRUE. Three sets of statistics are produced: the moderated t contrast
  # per timepoint, an omnibus moderated F per condition, and the global omnibus F (identical to
  # the per-condition one while there is a single condition).
  #
  # Args:
  #   df: the dataset as returned by read_dataset() — read once and reused, so the file is not
  #     parsed again per cell line.
  #   cell_line: cell line column prefix to analyse, e.g. "BRAFS151A1".
  #   conditions: condition names present in the column headers.
  #   control: control timepoint every contrast is taken against.
  #   min_residual_df: residual degrees of freedom every fitted site is required to have; the
  #     observation threshold is derived from it as ncol(design) + min_residual_df.
  #   min_control_obs: runs of the control timepoint a site must be observed in.
  #   normalize: between-sample normalisation method passed to normalize_matrix().
  #   block_on_replicate: TRUE to add the replicate term to the design. FALSE (default) fits the
  #     unblocked model; the replicate-effect diagnostic is reported either way.
  #   max_sites: cap on the number of sites fitted, for smoke-testing; Inf for a real run.
  #
  # Returns:
  #   data.frame with one row per tested site: the `site` key followed by this cell line's
  #   logFC / pvalue / FDR / adjustedFDR columns per timepoint and its F-test columns. Sites that
  #   did not pass the coverage filter are absent, not NA-filled — they are re-introduced as NA
  #   when the cell lines are merged.

  cat("\n=====", cell_line, "=====\n")
  checkpoint(paste("fitting", cell_line), reset = TRUE)

  stopifnot(!anyDuplicated(df$site))

  # --- samples -------------------------------------------------------------------------
  column_info <- stage("parse log2:abs column names",
                       parse_abs_columns(colnames(df), cell_line, conditions))
  cat("     log2:abs columns:", nrow(column_info), "\n")

  samples <- stage("de-duplicate control channels",
                   deduplicate_controls(df, column_info, conditions))
  cat("     unique sample runs after de-duplicating the controls:", nrow(samples), "\n")

  timepoints <- sort_timepoints(unique(column_info$timepoint))
  cat("     timepoints:", paste(timepoints, collapse = ", "), "\n")

  group_levels <- c(control,
                    setdiff(c("full", "starve"), control),
                    unlist(lapply(conditions, function(condition) {
                      paste(condition,
                            setdiff(timepoints, c("full", "starve")),
                            sep = "_")
                    })))
  group_levels <- group_levels[group_levels %in% samples$group]

  samples$group     <- factor(samples$group, levels = group_levels)
  samples$replicate <- factor(samples$replicate)

  # --- which sites are testable --------------------------------------------------------
  # The design is built here, before the filtering, because MIN_OBS is derived from its width:
  # df.residual = n_obs - rank(design) and rank <= ncol(design), so n_obs >= ncol + min_residual_df
  # guarantees the residual df. This is asserted after the fit rather than trusted.
  design <- if (block_on_replicate) {
    model.matrix(~ 0 + group + replicate, data = samples)
  } else {
    model.matrix(~ 0 + group, data = samples)
  }
  colnames(design) <- make.names(colnames(design))
  min_obs <- ncol(design) + min_residual_df
  cat(sprintf("     design formula: ~ 0 + group%s\n",
              if (block_on_replicate) " + replicate" else "   (unblocked: any run contrasted against all observed controls)"))

  coverage <- stage("count per-site coverage", site_coverage(df, samples, control = control))
  tested   <- coverage$n_obs >= min_obs & coverage$n_control_obs >= min_control_obs
  tested[is.na(tested)] <- FALSE

  if (is.finite(max_sites) && sum(tested) > max_sites) {
    tested[which(tested)[-seq_len(max_sites)]] <- FALSE
    cat("     NOTE: MAX_SITES is set — only", max_sites, "sites will be fitted.\n")
  }

  cat(sprintf("     coverage filter: n_obs >= %d (%d design columns + %d df) and %s observed in >= %d runs\n",
              min_obs, ncol(design), min_residual_df, control, min_control_obs))
  # The two failure counts overlap — a site with almost no runs usually fails both.
  cat(sprintf("     sites tested: %d of %d  | dropped: %d (too few runs: %d; no usable %s: %d; the two overlap)\n",
              sum(tested), nrow(df), sum(!tested),
              sum(coverage$n_obs < min_obs),
              control,
              sum(coverage$n_control_obs < min_control_obs)))

  if (sum(tested) == 0) {
    warning("run_limma_cell_line: no site passes the coverage filter for '", cell_line,
            "' — skipped.")
    return(NULL)
  }

  # An omnibus F needs every group estimable for that site (see "Omnibus F and missing
  # timepoints"): limma computes it from the vector of contrast t-statistics, and one NA in that
  # vector makes the whole F NA. This is how many sites can possibly get one.
  n_f_eligible <- sum(tested & coverage$n_fgroups_obs == nlevels(samples$group) - ("full" %in% timepoints))
  cat(sprintf("     of those, %d (%.1f%%) were observed at every timepoint the F is built from — only these can yield an omnibus F\n",
              n_f_eligible, 100 * n_f_eligible / sum(tested)))

  # --- expression matrix ---------------------------------------------------------------
  M <- stage("build expression matrix",
             as.matrix(df[tested, samples$column, drop = FALSE]))
  rownames(M) <- df$site[tested]
  storage.mode(M) <- "double"
  M <- stage(paste0("normalise (", normalize, ")"), normalize_matrix(M, normalize))
  cat("     matrix:", nrow(M), "sites x", ncol(M), "runs | ",
      sprintf("%.1f%% NA", 100 * sum(is.na(M)) / length(M)), "\n")

  # What a replicate block would absorb — reported whichever design is being fitted, so the
  # BLOCK_ON_REPLICATE choice rests on this number rather than on principle.
  invisible(stage("replicate-effect diagnostic",
                  replicate_effect_report(M, samples, label = cell_line)))

  # --- design and fit ------------------------------------------------------------------
  cat("     design:", nrow(design), "x", ncol(design),
      " | rank:", qr(design)$rank, "\n")
  stopifnot(qr(design)$rank == ncol(design))

  cat(sprintf("     lmFit will fit %d sites x %d coefficients%s\n",
              nrow(M), ncol(design),
              if (anyNA(M)) " (per-site loop: NAs present)" else " (single QR: no NAs)"))
  fit <- stage("lmFit", muffle_warnings(lmFit(M, design), "lmFit"))
  report_memory("after lmFit")

  # The point of deriving min_obs from ncol(design): every fitted site must clear min_residual_df.
  cat(sprintf("     residual df: min %d, median %.0f, max %d\n",
              min(fit$df.residual), median(fit$df.residual), max(fit$df.residual)))
  stopifnot(min(fit$df.residual) >= min_residual_df)

  # --- contrasts -----------------------------------------------------------------------
  stimulation_timepoints <- setdiff(timepoints, c("full", "starve"))
  contrast_groups <- setdiff(levels(samples$group), control)

  fit_t <- stage(sprintf("t-contrasts + eBayes (%d contrasts)", length(contrast_groups)),
                 muffle_warnings(
                   eBayes(contrasts.fit(fit,
                                        build_contrasts(contrast_groups, control, design)),
                          trend = TRUE,
                          robust = TRUE),
                   "t-contrasts"))

  f_groups <- list()
  for (condition in conditions) {
    f_groups[[condition]] <- paste(condition, stimulation_timepoints, sep = "_")
  }
  f_groups[["ALL"]] <- unique(unlist(f_groups, use.names = FALSE))

  f_fits <- list()
  for (f_name in names(f_groups)) {
    f_fits[[f_name]] <- stage(sprintf("omnibus F: %s", f_name),
                              muffle_warnings(
                                eBayes(contrasts.fit(fit,
                                                     build_contrasts(f_groups[[f_name]],
                                                                     control,
                                                                     design)),
                                       trend = TRUE,
                                       robust = TRUE),
                                paste("F test", f_name)))
  }
  report_memory("after all fits")

  # --- assemble ------------------------------------------------------------------------
  # `contrast_groups` is already in reporting order: full first, then the condition with its
  # timepoints ascending. `labels` keeps that order and is reused to order the output columns.
  labels <- vapply(contrast_groups,
                   function(group) {
                     if (group %in% timepoints) paste0(conditions[1], "_", group) else group
                   },
                   character(1),
                   USE.NAMES = FALSE)

  results <- stage("assemble result columns", {
    assembled <- data.frame(site = rownames(M), stringsAsFactors = FALSE)

    for (i in seq_along(contrast_groups)) {
      group <- contrast_groups[i]
      label <- labels[i]

      assembled[[paste0(cell_line, "_log2:limmaFC_", label)]] <- fit_t$coefficients[, group]
      assembled <- add_test_columns(assembled,
                                    fit_t$p.value[, group],
                                    list(pvalue      = paste0(cell_line, "_log2:pvalue_", label),
                                         fdr         = paste0(cell_line, "_log2:FDR_", label),
                                         adjustedfdr = paste0(cell_line, "_log2:adjustedFDR_", label)))
    }

    for (name in names(f_fits)) {
      assembled <- add_test_columns(assembled,
                                    f_fits[[name]]$F.p.value,
                                    list(pvalue      = paste0(cell_line, "_log2:Fpvalue_", name, "_omnibus"),
                                         fdr         = paste0(cell_line, "_log2:FFDR_", name, "_omnibus"),
                                         adjustedfdr = paste0(cell_line, "_log2:adjustedFFDR_", name, "_omnibus")))
    }
    assembled
  })

  results <- stage("reorder columns by data type",
                   order_by_datatype(results, cell_line, labels, names(f_fits)))
  checkpoint(sprintf("%s fitted: %d sites x %d columns",
                     cell_line, nrow(results), ncol(results)))

  # --- report --------------------------------------------------------------------------
  cat("\nsites at FDR < 0.05 per contrast (of the sites where the contrast was estimable):\n")
  for (label in labels) {
    fdr_col <- paste0(cell_line, "_log2:FDR_", label)
    cat(sprintf("  %-16s %5d significant / %5d tested / %d fitted\n", label,
                sum(results[[fdr_col]] < 0.05, na.rm = TRUE),
                sum(!is.na(results[[fdr_col]])),
                nrow(results)))
  }
  cat("sites at F-test FDR < 0.05:\n")
  for (name in names(f_fits)) {
    fdr_col <- paste0(cell_line, "_log2:FFDR_", name, "_omnibus")
    cat(sprintf("  %-16s %5d significant / %5d with an F / %d fitted\n", name,
                sum(results[[fdr_col]] < 0.05, na.rm = TRUE),
                sum(!is.na(results[[fdr_col]])),
                nrow(results)))
  }

  results
}


verify_contrast_definition <- function(df,
                                       results,
                                       cell_line,
                                       condition,
                                       control = "starve",
                                       block_on_replicate = FALSE,
                                       checks = c("2", "10", "90")) {
  # Check limma's logFC against the difference the fitted design defines it to be.
  #
  # Unblocked: the difference of the observed group means, checked on every tested site.
  # Blocked: the mean of the per-replicate differences, checked on fully observed sites only
  # (where the two definitions coincide; see the section above).
  #
  # Args:
  #   df: the dataset that was fitted, as returned by read_dataset() — reused rather than re-read.
  #   results: results data.frame returned by run_limma_cell_line() for this cell line.
  #   cell_line: cell line column prefix.
  #   condition: condition name in the column headers.
  #   control: control timepoint used as the contrast reference.
  #   block_on_replicate: the design that was fitted; selects which identity is checked.
  #   checks: timepoints to verify.
  #
  # Returns:
  #   Invisibly, the maximum absolute discrepancy found, or NA if no site could be checked.

  own_columns <- grep(paste0("^", cell_line, "_log2:abs_"), colnames(df), value = TRUE)
  replicates  <- unique(sub(".*_(r[0-9]+)$", "\\1", own_columns))

  # The rows the identity is expected to hold on.
  checked <- if (block_on_replicate) {
    df[rowSums(is.na(df[own_columns])) == 0, , drop = FALSE]
  } else {
    df
  }
  checked <- checked[checked$site %in% results$site, , drop = FALSE]

  if (nrow(checked) == 0) {
    cat(sprintf("  %-14s no site available to verify%s\n", cell_line,
                if (block_on_replicate) " (none fully observed)" else ""))
    return(invisible(NA_real_))
  }

  run_columns <- function(timepoint) {
    paste0(cell_line, "_log2:abs_", condition, "_", timepoint, "_", replicates)
  }

  worst <- 0
  for (timepoint in checks) {
    fc_col <- paste0(cell_line, "_log2:limmaFC_", condition, "_", timepoint)
    if (!fc_col %in% colnames(results)) next

    manual <- if (block_on_replicate) {
      rowMeans(as.matrix(checked[run_columns(timepoint)]) -
               as.matrix(checked[run_columns(control)]))
    } else {
      # Observed group means: rowMeans(na.rm = TRUE) is exactly what the unblocked fit estimates.
      rowMeans(checked[run_columns(timepoint)], na.rm = TRUE) -
      rowMeans(checked[run_columns(control)],   na.rm = TRUE)
    }

    from_limma  <- results[[fc_col]][match(checked$site, results$site)]
    comparable  <- is.finite(manual) & is.finite(from_limma)
    discrepancy <- if (any(comparable)) max(abs(manual[comparable] - from_limma[comparable])) else NA_real_
    worst <- max(worst, discrepancy, na.rm = TRUE)

    cat(sprintf("  %-14s %-4s n=%5d  max |limma - %s| = %.3e\n",
                cell_line, timepoint, sum(comparable),
                if (block_on_replicate) "within-replicate mean" else "difference of group means",
                discrepancy))
  }

  invisible(worst)
}
