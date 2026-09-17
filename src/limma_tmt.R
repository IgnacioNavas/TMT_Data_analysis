# ─────────────────────────────────────────────────────────────────────────────────────────
# limma statistics for the TMT datasets (hek_1, hme1_1, hme1_2).
#
# Helper functions for notebooks/01_preprocessing/limma_for_pvalues.rmd. The design is
# ~ 0 + group + plex: each TMT plex is a physical block, so every contrast against the starve
# control is estimated inside the plex it belongs to.
#
#   source(file.path(PROJECT_ROOT, "src", "r_utils.R"))        # stage, checkpoint, ...
#   source(file.path(PROJECT_ROOT, "src", "limma_common.R"))   # sort_timepoints, ...
#   source(file.path(PROJECT_ROOT, "src", "limma_tmt.R"))
#
# ⚠️ Do NOT source this together with src/limma_diapasef.R in one session. Both define
# `parse_abs_columns`, `deduplicate_controls`, `read_dataset` and `limma_output_path` with
# different field conventions (`plex` here, `replicate` there); the second source() silently
# wins. The check below turns that into a visible warning.
#
# ⚠️ Every argument that defaulted to a notebook global in the .rmd (CELL_LINE, CONTROL,
# CONDITIONS, MIN_PLEX, NORMALIZE, MAX_SITES, VERBOSE) is now an explicit argument with a
# literal default, and the notebook passes its configuration in. Progress reporting reads the
# `tmt.verbose` option — see src/r_utils.R.
#
# Requires: limma, and src/r_utils.R + src/limma_common.R sourced first.
# ─────────────────────────────────────────────────────────────────────────────────────────

if (exists(".limma_module", envir = globalenv()) &&
    !identical(get(".limma_module", envir = globalenv()), "tmt")) {
  warning("limma_tmt.R: src/limma_diapasef.R is already sourced in this session. ",
          "The two define the same function names with different field conventions; ",
          "restart R and source only the one the notebook needs.", call. = FALSE)
}
.limma_module <- "tmt"


limma_output_path <- function(dataset_name,
                              data_files,
                              out_suffix = "_limma_pvalues.tsv") {
  # Build the output path of one dataset's limma table, next to that dataset's input file.
  #
  # Results are written into the same Experiment/{dataset}/Data/Processed/ folder the input was
  # read from, so each experiment folder stays self-contained. The name reuses the input's stem
  # (date + dataset), so the date prefix of the processed file carries over to the statistics.
  #
  # Args:
  #   dataset_name: dataset key, e.g. "hme1_2"; must be a name of `data_files`.
  #   data_files: named vector of input file paths.
  #   out_suffix: suffix appended to the input stem, including the extension.
  #
  # Returns:
  #   Absolute path of the .tsv to write. Never equal to the input path.

  input_path <- data_files[[dataset_name]]
  stem <- sub("_log2_processed\\.tsv$", "", basename(input_path))
  stem <- sub("\\.tsv$", "", stem)          # fallback if the input is named differently

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
  # Args:
  #   column_names: character vector of all column names in the dataset.
  #   cell_line: cell line prefix to keep, e.g. "WT".
  #   conditions: character vector of condition names, e.g. c("EGF", "INS", "EGFnINS").
  #
  # Returns:
  #   data.frame with one row per log2:abs column and the columns
  #   `column`, `condition`, `timepoint`, `plex`.

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
             plex      = vapply(fields, `[`, character(1), 4),
             stringsAsFactors = FALSE)
}


deduplicate_controls <- function(df,
                                 column_info,
                                 conditions,
                                 control_timepoints = c("full", "starve")) {
  # Remove the duplicated control columns that appear once per condition arm.
  #
  # `full` and `starve` are a single physical channel per plex but are written out under every
  # condition. Only the copy belonging to the first condition is kept, and the discarded copies
  # are checked to be identical to it first — if a future dataset stops duplicating them, this
  # raises rather than silently averaging different samples.
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
    for (plex in unique(column_info$plex)) {
      reference <- column_info$column[column_info$condition == keep_condition &
                                      column_info$timepoint == timepoint &
                                      column_info$plex      == plex]
      if (length(reference) != 1) next

      for (other_condition in conditions[-1]) {
        duplicate <- column_info$column[column_info$condition == other_condition &
                                        column_info$timepoint == timepoint &
                                        column_info$plex      == plex]
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


read_dataset <- function(file_path,
                         dataset_name) {
  # Read one processed dataset and report everything needed to explain a slow or wrong run.
  #
  # The file is read once per dataset and reused by the fit and both verification steps, rather
  # than being re-read three times. The diagnostics printed here answer, in order: is the file
  # the size we think it is, did it parse into the expected shape, are the intensity columns
  # numeric, and are missing values stored as NA rather than as 0 (see the note below).
  #
  # Args:
  #   file_path: path to the processed .tsv.
  #   dataset_name: dataset key, used in the printed report.
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
    cat(sprintf("      log2:abs cells: %.1f%% NA, %.1f%% exactly zero\n",
                100 * n_missing / length(values),
                100 * n_zero / length(values)))
    if (n_zero > 0.01 * length(values)) {
      cat("      WARNING: many exact zeros in log2:abs — missing values may have been\n",
          "               written as 0 instead of NA. Fix before trusting any p-value.\n", sep = "")
    }
  }

  report_memory(sprintf("after reading %s", dataset_name))
  flush.console()
  df
}


run_limma_dataset <- function(df,
                              dataset_name,
                              cell_line,
                              conditions,
                              control = "starve",
                              min_plex = 2,
                              normalize = "none",
                              max_sites = Inf) {
  # Fit the blocked limma model on one dataset and return the full table of statistics.
  #
  # Every stimulation timepoint is contrasted against the control timepoint within its own
  # plex, using the design ~ 0 + group + plex. Three sets of statistics are produced: the
  # moderated t contrast per timepoint, an omnibus moderated F per condition, and one global
  # omnibus F per site across all conditions.
  #
  # Args:
  #   df: the dataset as returned by read_dataset() — read once and reused, so the file is not
  #     parsed again here.
  #   dataset_name: short name of the dataset, used in the printed report.
  #   cell_line: cell line column prefix to analyse, e.g. "WT".
  #   conditions: condition names present in the column headers.
  #   control: control timepoint every contrast is taken against.
  #   min_plex: minimum value of `n:reps` for a site to be tested.
  #   normalize: between-sample normalisation method passed to normalize_matrix().
  #   max_sites: cap on the number of sites fitted, for smoke-testing; Inf for a real run.
  #
  # Returns:
  #   data.frame with one row per tested site: the `site` key followed by the logFC / pvalue /
  #   FDR / adjustedFDR columns per timepoint and the F-test columns per condition and global.

  cat("\n=====", dataset_name, "=====\n")
  checkpoint(paste("fitting", dataset_name), reset = TRUE)

  stopifnot(!anyDuplicated(df$site))
  stopifnot("n:reps" %in% colnames(df))

  # --- samples -------------------------------------------------------------------------
  column_info <- stage("parse log2:abs column names",
                       parse_abs_columns(colnames(df), cell_line, conditions))
  cat("     log2:abs columns:", nrow(column_info), "\n")

  samples <- stage("de-duplicate control channels",
                   deduplicate_controls(df, column_info, conditions))
  cat("     unique sample channels after de-duplicating the controls:", nrow(samples), "\n")

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

  samples$group <- factor(samples$group, levels = group_levels)
  samples$plex  <- factor(samples$plex)

  # --- expression matrix ---------------------------------------------------------------
  tested <- df[["n:reps"]] >= min_plex
  tested[is.na(tested)] <- FALSE
  if (is.finite(max_sites) && sum(tested) > max_sites) {
    tested[which(tested)[-seq_len(max_sites)]] <- FALSE
    cat("     NOTE: MAX_SITES is set — only", max_sites, "sites will be fitted.\n")
  }
  cat("     sites tested (n:reps >=", min_plex, "):", sum(tested), "of", nrow(df),
      " | dropped:", sum(!tested), "\n")

  M <- stage("build expression matrix",
             as.matrix(df[tested, samples$column, drop = FALSE]))
  rownames(M) <- df$site[tested]
  storage.mode(M) <- "double"
  M <- stage(paste0("normalise (", normalize, ")"), normalize_matrix(M, normalize))
  cat("     matrix:", nrow(M), "sites x", ncol(M), "samples | ",
      sprintf("%.1f%% NA", 100 * sum(is.na(M)) / length(M)), "\n")

  # Missingness is supposed to be plex-wise: within one plex a site is either present in all 17
  # channels or absent from all of them. A "partial" plex block breaks the premise behind
  # MIN_PLEX and means n:reps no longer equals the number of plexes the site was seen in.
  # (The number of distinct patterns is not the right check — with 4 plexes and n:reps in 2..4
  # there are legitimately C(4,2)+C(4,3)+C(4,4) = 11 of them.)
  partial_blocks <- 0
  for (plex_level in levels(samples$plex)) {
    block   <- is.na(M[, samples$plex == plex_level, drop = FALSE])
    n_na    <- rowSums(block)
    partial_blocks <- partial_blocks + sum(n_na > 0 & n_na < ncol(block))
  }
  total_blocks <- nrow(M) * nlevels(samples$plex)
  cat(sprintf("     plex blocks partially missing: %d of %d (%.2f%%)%s\n",
              partial_blocks, total_blocks, 100 * partial_blocks / total_blocks,
              if (partial_blocks == 0) " — plex-wise, as expected" else
                " — missingness is NOT purely plex-wise; n:reps may not mean what MIN_PLEX assumes"))

  # --- design and fit ------------------------------------------------------------------
  design <- model.matrix(~ 0 + group + plex, data = samples)
  colnames(design) <- make.names(colnames(design))
  cat("     design:", nrow(design), "x", ncol(design),
      " | rank:", qr(design)$rank, "\n")
  stopifnot(qr(design)$rank == ncol(design))

  cat(sprintf("     lmFit will fit %d sites x %d coefficients%s\n",
              nrow(M), ncol(design),
              if (anyNA(M)) " (per-site loop: NAs present)" else " (single QR: no NAs)"))
  fit <- stage("lmFit", muffle_warnings(lmFit(M, design), "lmFit"))
  report_memory("after lmFit")

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
  f_groups[["ALL"]] <- unlist(f_groups, use.names = FALSE)

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
  # `contrast_groups` is already in reporting order: full first, then each condition with its
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
                     dataset_name, nrow(results), ncol(results)))

  # --- report --------------------------------------------------------------------------
  cat("\nsites at FDR < 0.05 per contrast:\n")
  for (label in labels) {
    fdr_col <- paste0(cell_line, "_log2:FDR_", label)
    cat(sprintf("  %-16s %4d / %d\n", label,
                sum(results[[fdr_col]] < 0.05, na.rm = TRUE), nrow(results)))
  }
  cat("sites at F-test FDR < 0.05:\n")
  for (name in names(f_fits)) {
    fdr_col <- paste0(cell_line, "_log2:FFDR_", name, "_omnibus")
    cat(sprintf("  %-16s %4d / %d\n", name,
                sum(results[[fdr_col]] < 0.05, na.rm = TRUE), nrow(results)))
  }

  results
}


verify_within_plex <- function(df,
                               results,
                               cell_line,
                               control = "starve",
                               checks = c("EGF_5", "INS_10", "EGFnINS_90")) {
  # Check that limma's logFC equals the mean of the per-plex differences against the control.
  #
  # Args:
  #   df: the dataset that was fitted, as returned by read_dataset() — reused rather than
  #     re-read from disk.
  #   results: results data.frame returned by run_limma_dataset().
  #   cell_line: cell line column prefix.
  #   control: control timepoint used as the contrast reference.
  #   checks: contrasts to verify, each named "{condition}_{timepoint}".
  #
  # Returns:
  #   Invisibly, the maximum absolute discrepancy found across the checked sites.

  complete <- df[df[["n:reps"]] == 4, , drop = FALSE]
  complete <- complete[complete$site %in% results$site, , drop = FALSE]
  if (nrow(complete) == 0) {
    cat("no complete (4-plex) sites available to verify\n")
    return(invisible(NA_real_))
  }

  plexes <- c("r1", "r2", "r3", "r4")
  worst  <- 0

  for (check in checks) {
    parts     <- strsplit(check, "_")[[1]]
    condition <- parts[1]
    timepoint <- parts[2]

    fc_col <- paste0(cell_line, "_log2:limmaFC_", check)
    if (!fc_col %in% colnames(results)) next

    # vapply + an explicit matrix(), not sapply(): with exactly one complete site sapply
    # simplifies the four per-plex differences to a length-4 vector and rowMeans() then fails
    # with "'x' must be an array of at least two dimensions".
    per_plex <- vapply(plexes,
                       function(plex) {
                         complete[[paste0(cell_line, "_log2:abs_", condition, "_", timepoint, "_", plex)]] -
                         complete[[paste0(cell_line, "_log2:abs_", condition, "_", control, "_", plex)]]
                       },
                       numeric(nrow(complete)))
    per_plex <- matrix(per_plex, nrow = nrow(complete), dimnames = list(NULL, plexes))
    manual <- rowMeans(per_plex)

    from_limma <- results[[fc_col]][match(complete$site, results$site)]
    discrepancy <- max(abs(manual - from_limma), na.rm = TRUE)
    worst <- max(worst, discrepancy)

    cat(sprintf("  %-12s n=%d  max |limma - within-plex mean| = %.3e\n",
                check, nrow(complete), discrepancy))
  }

  invisible(worst)
}
