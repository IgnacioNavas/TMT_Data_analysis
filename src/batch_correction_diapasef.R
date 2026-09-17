# ─────────────────────────────────────────────────────────────────────────────────────────
# Batch correction (diaPASEF) — helper functions.
#
# Shared by the two drivers, which differ only in configuration:
#   notebooks/01_preprocessing/batch_correction_diapasef.rmd  (notebook, for reading the
#                                                              diagnostics as they appear)
#   notebooks/01_preprocessing/batch_correction_diapasef.R    (script, for Rscript / cron)
#
# Before this file existed the two carried their own copies and had drifted apart
# (`plot_reference_qc` gained an inline-drawing branch in the notebook only). One
# implementation now serves both.
#
#   source(file.path(PROJECT_ROOT, "src", "r_utils.R"))                  # say()
#   source(file.path(PROJECT_ROOT, "src", "batch_correction_diapasef.R"))
#
# Every function is pure except plot_reference_qc() (draws) and the say() calls (print).
# The correction itself is a single limma::removeBatchEffect call in the driver — nothing here
# re-implements it.
#
# ⚠️ Arguments that were notebook globals in the .rmd (OUT_DIR, OVERWRITE,
# ID_COLUMN_CANDIDATES, REQUIRED_SHEET_COLUMNS, NUMERIC_ANNOTATION_COLUMNS,
# LOG2_MAX_PLAUSIBLE, VERBOSE) are now explicit arguments with literal defaults matching the
# notebook's values, and the driver passes its configuration in.
#
# ⚠️ `read_tsv()` shadows readr::read_tsv. This file keeps the project's own name, so do not
# attach readr in a session that sources it.
#
# Requires: limma (called as limma::), and src/r_utils.R sourced first for say().
# ─────────────────────────────────────────────────────────────────────────────────────────

output_path <- function(input_path,
                        suffix,
                        out_dir = dirname(input_path),
                        overwrite = FALSE) {
  # Build an output path from the input filename plus a descriptive suffix.
  #
  # The input extension is stripped and the suffix appended, so the input file can never be
  # the output file. Stops if the target exists and OVERWRITE is FALSE (project rule: never
  # overwrite a processed data file, always version the name).
  #
  # Args:
  #   input_path: path of the abundance matrix the output is derived from.
  #   suffix: string appended to the input stem, including its own extension.
  #   out_dir: directory to write into; defaults to the input's own folder.
  #   overwrite: FALSE (default) refuses to write over an existing file.
  #
  # Returns:
  #   Absolute path of the file to write.

  stem <- sub("\\.[A-Za-z0-9]+$", "", basename(input_path))
  path <- file.path(out_dir, paste0(stem, suffix))

  if (file.exists(path) && !overwrite) {
    stop("output_path: '", path, "' already exists and overwrite is FALSE. ",
         "Move it aside or add a version tag to the suffix.")
  }
  if (normalizePath(path, mustWork = FALSE) == normalizePath(input_path, mustWork = FALSE)) {
    stop("output_path: refusing to write over the input file '", input_path, "'.")
  }
  path
}


read_tsv <- function(path,
                     what) {
  # Read a project TSV, keeping column names verbatim and missing values as NA.
  #
  # `check.names = FALSE` is essential: the sample column names contain ':' and the header
  # would otherwise be mangled into `WT_raw.abs_EGF_full_r1`, breaking the join against the
  # sample sheet. Empty strings and the usual sentinels become NA — nothing is ever
  # zero-filled (project rule).
  #
  # Args:
  #   path: file to read.
  #   what: short human-readable description used in error messages.
  #
  # Returns:
  #   A data.frame with the file's columns in file order.

  if (!file.exists(path)) {
    stop("read_tsv: ", what, " not found at '", path, "'.")
  }
  df <- utils::read.delim(path,
                          sep = "\t",
                          header = TRUE,
                          check.names = FALSE,
                          stringsAsFactors = FALSE,
                          na.strings = c("", "NA", "NaN", "nan", "None"))
  say("  loaded ", what, ": ", nrow(df), " rows x ", ncol(df), " columns")
  df
}


canonicalise_sheet <- function(sheet,
                               id_candidates = c("sample_id", "sample_Id", "sample_ID",
                                                 "sampleId"),
                               required_columns = c("cell_line", "condition", "timepoint",
                                                    "batch", "is_reference")) {
  # Validate the sample sheet and normalise its column names and types.
  #
  # Renames whichever id spelling is present to `sample_id`, checks the required columns
  # exist, coerces `is_reference` from R logicals / "TRUE"/"FALSE" / 0/1 to a logical
  # vector, and forces `batch` to a factor. Stops loudly on anything missing or ambiguous
  # rather than guessing.
  #
  # Args:
  #   sheet: data.frame as read from the sample-sheet TSV.
  #   id_candidates: accepted spellings of the id column; exactly one must be present.
  #   required_columns: columns the sheet must carry.
  #
  # Returns:
  #   The same data.frame with `sample_id` (character), `batch` (factor), `is_reference`
  #   (logical), and the remaining required columns present.

  id_present <- intersect(id_candidates, colnames(sheet))
  if (length(id_present) == 0L) {
    stop("canonicalise_sheet: the sample sheet has no id column. Expected one of: ",
         paste(id_candidates, collapse = ", "),
         ". Found: ", paste(colnames(sheet), collapse = ", "))
  }
  if (length(id_present) > 1L) {
    stop("canonicalise_sheet: the sample sheet has more than one id column (",
         paste(id_present, collapse = ", "), "). Keep exactly one.")
  }
  colnames(sheet)[colnames(sheet) == id_present[[1L]]] <- "sample_id"

  missing_cols <- setdiff(required_columns, colnames(sheet))
  if (length(missing_cols) > 0L) {
    stop("canonicalise_sheet: sample sheet is missing required column(s): ",
         paste(missing_cols, collapse = ", "))
  }

  sheet$sample_id <- as.character(sheet$sample_id)
  if (anyDuplicated(sheet$sample_id) > 0L) {
    dups <- unique(sheet$sample_id[duplicated(sheet$sample_id)])
    stop("canonicalise_sheet: duplicated sample_id in the sheet: ",
         paste(dups, collapse = ", "))
  }

  # is_reference: accept logical, "TRUE"/"FALSE"/"True"/"true", or 0/1.
  ref_raw <- sheet$is_reference
  ref <- if (is.logical(ref_raw)) {
    ref_raw
  } else if (is.numeric(ref_raw)) {
    ref_raw != 0
  } else {
    tolower(trimws(as.character(ref_raw))) %in% c("true", "t", "yes", "1")
  }
  if (anyNA(ref)) {
    stop("canonicalise_sheet: `is_reference` could not be read as a logical for ",
         sum(is.na(ref)), " row(s).")
  }
  sheet$is_reference <- ref

  sheet$batch <- factor(as.character(sheet$batch))
  if (nlevels(sheet$batch) < 2L) {
    stop("canonicalise_sheet: `batch` has ", nlevels(sheet$batch),
         " level(s). Nothing to correct — check the sample sheet.")
  }

  say("  sample sheet: ", nrow(sheet), " injections, ",
      nlevels(sheet$batch), " batches (", paste(levels(sheet$batch), collapse = "/"), "), ",
      sum(sheet$is_reference), " pooled references")
  sheet
}


split_matrix_columns <- function(abundance,
                                 sheet,
                                 numeric_annotation_columns = character(0)) {
  # Split the abundance table into sample columns and annotation columns.
  #
  # Sample columns are identified ONLY by matching the header against the sheet's
  # `sample_id` — never by pattern-matching on the column name. Two failures are fatal: a
  # `sample_id` with no matching column, and a numeric column that is in neither the sheet
  # nor NUMERIC_ANNOTATION_COLUMNS (which almost always means an injection was left out of
  # the sheet and would silently escape correction).
  #
  # Args:
  #   abundance: data.frame as read from the abundance TSV.
  #   sheet: canonicalised sample sheet.
  #   numeric_annotation_columns: annotation columns that are legitimately numeric and
  #     must not be mistaken for an injection missing from the sheet.
  #
  # Returns:
  #   A list with `sample_cols` (character, in sheet order) and `annot_cols` (character, in
  #   file order).

  sample_cols <- sheet$sample_id
  missing_in_matrix <- setdiff(sample_cols, colnames(abundance))
  if (length(missing_in_matrix) > 0L) {
    stop("split_matrix_columns: ", length(missing_in_matrix),
         " sample_id(s) in the sheet have no column in the abundance matrix: ",
         paste(utils::head(missing_in_matrix, 20L), collapse = ", "),
         if (length(missing_in_matrix) > 20L) " ..." else "")
  }

  annot_cols <- setdiff(colnames(abundance), sample_cols)

  unexplained <- annot_cols[vapply(abundance[annot_cols], is.numeric, logical(1))]
  unexplained <- setdiff(unexplained, numeric_annotation_columns)
  if (length(unexplained) > 0L) {
    stop("split_matrix_columns: ", length(unexplained),
         " numeric column(s) are neither a sample_id in the sheet nor a known annotation: ",
         paste(unexplained, collapse = ", "),
         ". If these are injections, add them to the sample sheet; if they are annotations, ",
         "add them to NUMERIC_ANNOTATION_COLUMNS.")
  }

  say("  columns: ", length(sample_cols), " samples, ", length(annot_cols), " annotations")
  list(sample_cols = sample_cols, annot_cols = annot_cols)
}


to_log2_if_needed <- function(x,
                              log2_max_plausible = 64) {
  # Put the intensity matrix on the log2 scale, exactly once.
  #
  # `removeBatchEffect` fits a linear model, so an additive batch offset only *is* additive
  # on the log scale. Linear intensities are detected by their range (see
  # log2_max_plausible) and transformed; values that are already log2 are left alone so the
  # matrix is never double-logged. Non-positive values become NA, NOT zero — a zero
  # intensity in DIA is a non-detection, and log2(0) = -Inf would drag the linear fit. No
  # normalisation of any kind is applied.
  #
  # Args:
  #   x: numeric matrix of intensities, injections in columns.
  #     log2_max_plausible: values above this are unambiguously linear. Real log2 phospho
  #     intensities live in roughly [5, 35]; linear detector counts run to 1e5-1e9.
  #
  # Returns:
  #   A list with `x` (the log2-scale matrix) and `was_linear` (logical, which branch ran).

  finite_vals <- x[is.finite(x)]
  if (length(finite_vals) == 0L) {
    stop("to_log2_if_needed: the sample matrix contains no finite values.")
  }

  rng <- range(finite_vals)
  was_linear <- rng[[2L]] > log2_max_plausible

  say("  value range: [", signif(rng[[1L]], 4), ", ", signif(rng[[2L]], 4), "], ",
      "median ", signif(stats::median(finite_vals), 4))

  if (was_linear) {
    say("  scale check -> LINEAR intensities: applying log2 (non-positive -> NA)")
    x[!is.na(x) & x <= 0] <- NA_real_
    x <- log2(x)
  } else {
    say("  scale check -> already LOG2: no transformation applied (not double-logging)")
    x[!is.finite(x)] <- NA_real_
  }

  say("  missing values: ", sum(is.na(x)), " / ", length(x),
      " (", sprintf("%.1f%%", 100 * mean(is.na(x))), ") — kept as NA, never zero-filled")

  list(x = x, was_linear = was_linear)
}


build_bio_group <- function(sheet) {
  # Build the biological factor that the correction must PROTECT.
  #
  # One level per (cell_line, condition, timepoint) combination — the finest biological unit
  # in the experiment, so nothing biological is pooled and therefore nothing biological can
  # be absorbed into the batch term. The pooled references are overridden to a single
  # "reference" level: they are the same material in all three batches, so giving them their
  # own free mean stops that material's overall abundance from being read as a biological
  # group, while still letting them ride along and be corrected.
  #
  # BRAF biological replicates 1 and 2 are distinct `cell_line` levels in the sheet, so they
  # stay distinct here — they are not pooled.
  #
  # Args:
  #   sheet: canonicalised sample sheet, rows in the order the matrix columns will be in.
  #
  # Returns:
  #   A factor with one entry per injection, levels made syntactically valid.

  grp <- paste(sheet$cell_line, sheet$condition, sheet$timepoint, sep = ".")
  grp[sheet$is_reference] <- "reference"
  factor(make.names(grp))
}


check_batch_orthogonality <- function(bio_group,
                                      batch) {
  # Report how batch and biology are laid out against each other.
  #
  # The design of this experiment puts every biological group in every batch exactly once,
  # which makes `batch` orthogonal to `bio_group`. That is the reason the correction cannot
  # erode cell-line / condition / time effects: there is no biological contrast that is even
  # partly a batch contrast, so subtracting the batch offsets removes a direction the
  # biology does not live in. This function checks that assumption instead of assuming it,
  # and warns (rather than stops) if the layout is imperfect — a few groups missing a batch
  # is a missingness problem, not a design failure.
  #
  # Args:
  #   bio_group: factor of biological groups, one entry per injection.
  #   batch: factor of batches, one entry per injection.
  #
  # Returns:
  #   The group x batch contingency table, invisibly.

  tab <- table(bio_group, batch)
  balanced <- all(tab == 1L)

  if (balanced) {
    say("  design: each biological group appears exactly once per batch -> ",
        "batch is orthogonal to bio_group; the correction cannot touch the biology")
  } else {
    n_unbal <- sum(apply(tab, 1L, function(r) length(unique(r)) > 1L))
    warning("check_batch_orthogonality: ", n_unbal, " of ", nrow(tab),
            " biological groups are not evenly spread across batches. ",
            "Batch and biology are partially confounded for those groups; ",
            "the correction there is a compromise, not a clean removal.",
            call. = FALSE)
    say("  design: NOT fully balanced — ", n_unbal, " group(s) unevenly spread over batches")
  }

  invisible(tab)
}


count_nonestimable <- function(x,
                               design,
                               batch) {
  # Count, per site, how many batch coefficients could not be estimated.
  #
  # A site too sparsely quantified in some batch leaves that batch's offset undetermined;
  # limma returns NA and `removeBatchEffect` silently substitutes 0, i.e. that part of the
  # correction is simply not applied. That is expected under DIA missingness and acceptable
  # — but it should be visible rather than silent, which limma's single "Partial NA
  # coefficients" warning does not make it (that warning also fires for NA *biological*
  # coefficients, which are harmless here, so it cannot be read as a batch diagnostic).
  #
  # This re-fits the exact model `removeBatchEffect` fits internally (the batch factor in
  # sum-to-zero coding, projected onto the orthogonal complement of `design`) purely to read
  # the NA pattern out of the coefficients. It does NOT produce the corrected values — those
  # come from `removeBatchEffect` itself, so there is one implementation of the correction,
  # not two.
  #
  # Args:
  #   x: log2-scale matrix, sites in rows, injections in columns.
  #   design: biological design matrix passed to removeBatchEffect.
  #   batch: factor of batches, one entry per injection.
  #
  # Returns:
  #   Integer vector, one entry per site: the number of NA batch coefficients (0 = fully
  #   corrected, ncol(batch contrasts) = not corrected at all).

  b <- as.factor(batch)
  stats::contrasts(b) <- stats::contr.sum(levels(b))
  batch_mat <- stats::model.matrix(~ b)[, -1L, drop = FALSE]
  batch_mat <- qr.resid(qr(design), batch_mat)

  fit <- suppressWarnings(limma::lmFit(x, cbind(design, batch_mat)))
  beta <- fit$coefficients[, -seq_len(ncol(design)), drop = FALSE]
  rowSums(is.na(beta))
}


reference_spread <- function(x,
                             ref_cols) {
  # Per-site technical spread across the pooled reference injections.
  #
  # The references are the same physical material injected once per batch, so every
  # difference between them is technical. Only sites quantified in ALL references are
  # evaluable — under DIA missingness that is a subset, which is fine: this is a spot-check,
  # not a per-site gate.
  #
  # CV is reported on the log2 scale as sd/|mean| * 100, which is a ratio of log2 units, not
  # the linear-scale CV. SD in log2 units is the primary number; the CV column is kept
  # because it is scale-free across sites.
  #
  # Args:
  #   x: log2-scale matrix, sites in rows.
  #   ref_cols: column names of the reference injections.
  #
  # Returns:
  #   A data.frame with one row per site: `n_obs`, `mean`, `sd`, `cv` (NA where the site is
  #   not observed in every reference).

  m <- x[, ref_cols, drop = FALSE]
  n_obs <- rowSums(!is.na(m))
  complete <- n_obs == length(ref_cols)

  mu <- rep(NA_real_, nrow(m))
  sd <- rep(NA_real_, nrow(m))
  mu[complete] <- rowMeans(m[complete, , drop = FALSE])
  sd[complete] <- apply(m[complete, , drop = FALSE], 1L, stats::sd)

  data.frame(n_obs = n_obs,
             mean = mu,
             sd = sd,
             cv = 100 * sd / abs(mu),
             row.names = rownames(m),
             stringsAsFactors = FALSE)
}


plot_reference_qc <- function(before,
                              after,
                              path = NULL) {
  # Draw the before/after reference-spread diagnostic, optionally saving it.
  #
  # Left: overlaid histograms of per-site SD across the references. Right: scatter of before
  # vs after with the identity line — points below the line are sites whose technical spread
  # shrank. Success is the whole cloud sitting below the diagonal.
  #
  # Args:
  #   before: data.frame from reference_spread() on the uncorrected values.
  #   after: data.frame from reference_spread() on the corrected values.
  #   path: PNG file to write; NULL draws to the current device (inline in the notebook).
  #
  # Returns:
  #   `path`, invisibly (NULL when drawing inline).

  ok <- !is.na(before$sd) & !is.na(after$sd)
  b <- before$sd[ok]
  a <- after$sd[ok]

  if (!is.null(path)) {
    grDevices::png(path, width = 1600, height = 750, res = 140)
    on.exit(grDevices::dev.off(), add = TRUE)
  }
  old_par <- graphics::par(mfrow = c(1L, 2L), mar = c(4.5, 4.5, 3, 1))
  # `after = FALSE` puts this handler FIRST, so par() is restored while the png device is
  # still current and dev.off() closes it afterwards. Registered the other way round,
  # dev.off() runs first and par(old_par) then lands on the null device — which opens a
  # stray graphics device (and an Rplots.pdf under Rscript) as a side effect.
  on.exit(graphics::par(old_par), add = TRUE, after = FALSE)

  brks <- graphics::hist(c(b, a), breaks = 40L, plot = FALSE)$breaks
  hb <- graphics::hist(b, breaks = brks, plot = FALSE)
  ha <- graphics::hist(a, breaks = brks, plot = FALSE)
  graphics::plot(hb, col = grDevices::adjustcolor("grey40", 0.6), border = NA,
                 ylim = c(0, max(hb$counts, ha$counts)),
                 main = "Reference spread per site",
                 xlab = "SD across mix / mixb / mixc (log2)", ylab = "sites")
  graphics::plot(ha, col = grDevices::adjustcolor("#1f77b4", 0.6), border = NA, add = TRUE)
  graphics::abline(v = stats::median(b), col = "grey20", lwd = 2, lty = 2)
  graphics::abline(v = stats::median(a), col = "#1f77b4", lwd = 2, lty = 2)
  graphics::legend("topright", bty = "n",
                   fill = c(grDevices::adjustcolor("grey40", 0.6),
                            grDevices::adjustcolor("#1f77b4", 0.6)),
                   legend = c(sprintf("before (median %.3f)", stats::median(b)),
                              sprintf("after  (median %.3f)", stats::median(a))))

  lim <- range(c(b, a), finite = TRUE)
  graphics::plot(b, a, pch = 16, cex = 0.35,
                 col = grDevices::adjustcolor("black", 0.25),
                 xlim = lim, ylim = lim,
                 main = sprintf("%.1f%% of sites improved", 100 * mean(a < b)),
                 xlab = "SD before (log2)", ylab = "SD after (log2)")
  graphics::abline(0, 1, col = "red", lwd = 2)

  invisible(path)
}
