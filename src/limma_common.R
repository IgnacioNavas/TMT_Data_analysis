# ─────────────────────────────────────────────────────────────────────────────────────────
# limma helpers shared by both statistics notebooks.
#
#   notebooks/01_preprocessing/limma_for_pvalues.rmd           (TMT, blocked on plex)
#   notebooks/01_preprocessing/limma_for_pvalues_diapasef.rmd  (diaPASEF, per cell line)
#
# Only the functions whose bodies are IDENTICAL in both notebooks live here. Anything that
# differs between the two platforms — column parsing (`plex` vs `replicate`), control
# de-duplication, the fit itself — stays in src/limma_tmt.R / src/limma_diapasef.R.
#
# Requires: limma (called with the limma:: prefix, so the package need not be attached).
#
#   source(file.path(PROJECT_ROOT, "src", "r_utils.R"))
#   source(file.path(PROJECT_ROOT, "src", "limma_common.R"))
# ─────────────────────────────────────────────────────────────────────────────────────────


sort_timepoints <- function(timepoints) {
  # Sort timepoint labels in biological order: named controls first, then numerically.
  #
  # Numeric sorting rather than a fixed list is what puts the diaPASEF grid
  # {full, starve, 2, 5, 10, 15, 20, 30, 90} in experimental order — a hard-coded TMT ordering
  # has no 20 or 30 min and would emit them after 90.
  #
  # Args:
  #   timepoints: character vector of timepoint labels, e.g. c("10", "full", "2", "starve").
  #
  # Returns:
  #   The same labels, ordered with "full" and "starve" first and the numeric points ascending.

  named   <- c("full", "starve")
  present <- named[named %in% timepoints]
  numeric_points <- setdiff(timepoints, named)
  c(present, numeric_points[order(as.numeric(numeric_points))])
}


normalize_matrix <- function(M,
                             method) {
  # Apply between-sample normalisation to the expression matrix.
  #
  # Args:
  #   M: numeric matrix, sites in rows and samples in columns, on the log2 scale.
  #   method: "none" (return unchanged), "median" (centre every column on the grand median),
  #     or "quantile" (limma::normalizeBetweenArrays).
  #
  # Returns:
  #   The normalised matrix, same dimensions and dimnames as the input.

  if (method == "none") {
    return(M)
  }
  if (method == "median") {
    column_medians <- apply(M, 2, median, na.rm = TRUE)
    return(sweep(M, 2, column_medians) + mean(column_medians))
  }
  if (method == "quantile") {
    return(limma::normalizeBetweenArrays(M, method = "quantile"))
  }
  stop("normalize_matrix: unknown method '", method, "'.")
}


build_contrasts <- function(groups,
                            control,
                            design) {
  # Build the contrast matrix of every group against the control group.
  #
  # Args:
  #   groups: character vector of group names to contrast against the control.
  #   control: name of the control group, e.g. "starve".
  #   design: design matrix whose column names define the coefficient names.
  #
  # Returns:
  #   Contrast matrix with one column per group, named after the group.

  expressions <- paste0("group", groups, " - group", control)
  contrast_matrix <- limma::makeContrasts(contrasts = expressions, levels = design)
  colnames(contrast_matrix) <- groups
  contrast_matrix
}


order_by_datatype <- function(results,
                              cell_line,
                              labels,
                              f_names) {
  # Reorder the result columns so each data type forms one contiguous block.
  #
  # The natural assembly order is per contrast (all four statistics of EGF_2, then all four of
  # EGF_5, ...). This flips it to per data type: every limmaFC column with its timepoints in
  # order, then every pvalue column, then FDR, then adjustedFDR, then the F-test blocks.
  #
  # Args:
  #   results: assembled results data.frame, keyed by `site`.
  #   cell_line: cell line column prefix, e.g. "WT".
  #   labels: contrast labels ("EGF_full", "EGF_2", ...) in the desired timepoint order.
  #   f_names: names of the F-tests, e.g. c("EGF", "INS", "EGFnINS", "ALL") or c("EGF", "ALL").
  #
  # Returns:
  #   The same data.frame with its columns reordered. Raises if the reordering would drop or
  #   invent a column.

  contrast_stats <- c("limmaFC", "pvalue", "FDR", "adjustedFDR")
  f_stats        <- c("Fpvalue", "FFDR", "adjustedFFDR")

  ordered <- "site"
  for (statistic in contrast_stats) {
    ordered <- c(ordered, paste0(cell_line, "_log2:", statistic, "_", labels))
  }
  for (statistic in f_stats) {
    ordered <- c(ordered, paste0(cell_line, "_log2:", statistic, "_", f_names, "_omnibus"))
  }

  if (!setequal(ordered, colnames(results))) {
    stop("order_by_datatype: the reordered column list does not match the assembled one.")
  }

  results[, ordered, drop = FALSE]
}


add_test_columns <- function(results,
                             p_values,
                             prefix) {
  # Append p-value, BH-FDR and adjusted p-value columns for one test to the results frame.
  #
  # The adjusted p-value is -log10(FDR), matching the project's `adjustedFDR` convention.
  # An FDR of exactly 0 would give +Inf and is set to NA instead.
  #
  # BH is applied over the non-NA p-values only (p.adjust's default n is the number of
  # non-missing values), which is the intended behaviour here: a site whose contrast was
  # inestimable — it never appeared at that timepoint — was not tested and must not enlarge the
  # multiple-testing burden.
  #
  # Args:
  #   results: data.frame being assembled, one row per site.
  #   p_values: numeric vector of raw p-values, in the row order of `results`.
  #   prefix: named list of the three column names to write (pvalue / fdr / adjustedfdr).
  #
  # Returns:
  #   The results data.frame with three new columns appended.

  fdr <- p.adjust(p_values, method = "BH")
  adjusted <- -log10(fdr)
  adjusted[is.infinite(adjusted)] <- NA_real_

  results[[prefix$pvalue]]      <- p_values
  results[[prefix$fdr]]         <- fdr
  results[[prefix$adjustedfdr]] <- adjusted
  results
}
