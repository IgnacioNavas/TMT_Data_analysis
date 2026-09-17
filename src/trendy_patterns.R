# ─────────────────────────────────────────────────────────────────────────────────────────
# Helper functions for the Trendy segmented-regression analysis.
#
# Extracted from notebooks/01_preprocessing/trendy_temporal_patterns.rmd so the same helpers
# can be reused by other R notebooks (e.g. a diaPASEF / mutant version) and tested on their
# own. Source it from a notebook with:
#
#   source(file.path(PROJECT_ROOT, "src", "r_utils.R"))        # find_project_root, stage
#   source(file.path(PROJECT_ROOT, "src", "trendy_patterns.R"))
#
# Nothing here calls Trendy itself — `trendy_result_table()` only reshapes the list that
# `Trendy::results()` returns — so the file can be sourced without the package installed.
#
# ⚠️ Difference from the in-notebook originals: every function that used to default an
# argument to a notebook global (CONTROL, CELL_LINE, TIME_AXIS, ...) now takes that value as
# an explicit argument with a literal default. A sourced file must not depend on variables
# defined by whoever sourced it.
# ─────────────────────────────────────────────────────────────────────────────────────────


# ── Locating files ────────────────────────────────────────────────────────────────────────

resolve_input_file <- function(dataset,
                               candidates,
                               data_dir,
                               subdir = file.path("Data", "Processed")) {
  # Pick the first candidate processed file that exists for this dataset.
  #
  # Existence is the only test applied — the caller is responsible for checking that the file
  # actually carries the columns it needs (the notebook does this with
  # `parse_condition_columns`, which raises if no log2:abs column matches).
  #
  # Args:
  #   dataset: dataset key, e.g. "hme1_2"; also the folder name under Experiment/.
  #   candidates: character vector of file names to try, in priority order.
  #   data_dir: the Experiment/ directory.
  #   subdir: path of the processed folder inside the dataset folder.
  #
  # Returns:
  #   Absolute path to the first existing candidate. Raises listing the folder if none exists.

  folder <- file.path(data_dir, dataset, subdir)
  if (!dir.exists(folder)) {
    stop("resolve_input_file: '", folder, "' does not exist.")
  }

  for (candidate in candidates) {
    path <- file.path(folder, candidate)
    if (file.exists(path)) {
      return(path)
    }
  }

  stop("resolve_input_file: none of the candidates exist in '", folder, "'.\n",
       "  tried   : ", paste(candidates, collapse = ", "), "\n",
       "  present : ", paste(list.files(folder, pattern = "\\.tsv$"), collapse = ", "))
}


trendy_output_path <- function(input_path,
                               out_suffix) {
  # Build the output path next to the input, reusing the input's date-prefixed stem.
  #
  # Args:
  #   input_path: path of the processed .tsv that was read.
  #   out_suffix: suffix appended to the stem, including the extension,
  #               e.g. "_trendy_EGF_patterns.tsv".
  #
  # Returns:
  #   Absolute path of the .tsv to write. Raises if it would collide with the input.

  stem <- basename(input_path)
  stem <- sub("_processed_phPlus\\.tsv$", "", stem)
  stem <- sub("_log2_processed\\.tsv$", "", stem)
  stem <- sub("\\.tsv$", "", stem)

  output_path <- file.path(dirname(input_path), paste0(stem, out_suffix))
  if (identical(output_path, input_path)) {
    stop("trendy_output_path: refusing to overwrite the input file '", input_path, "'.")
  }
  output_path
}


# ── Timepoints and sample columns ────────────────────────────────────────────────────────

sort_timepoints <- function(timepoints,
                            control = "starve") {
  # Order timepoint labels biologically: the control first, then the numeric points ascending.
  #
  # Args:
  #   timepoints: character vector of timepoint labels, e.g. c("10", "2", "starve").
  #   control: label of the control timepoint, placed first.
  #
  # Returns:
  #   The labels in plotting / fitting order.

  numeric_points <- setdiff(timepoints, control)
  c(control[control %in% timepoints],
    numeric_points[order(as.numeric(numeric_points))])
}


timepoint_minutes <- function(labels,
                              control = "starve") {
  # Translate timepoint labels into minutes, with the control at 0.
  #
  # Args:
  #   labels: character vector of timepoint labels.
  #   control: label mapped to 0.
  #
  # Returns:
  #   Numeric vector of minutes, same length and order as `labels`.

  minutes <- suppressWarnings(as.numeric(labels))
  minutes[labels == control] <- 0
  if (anyNA(minutes)) {
    stop("timepoint_minutes: cannot convert to minutes: ",
         paste(labels[is.na(minutes)], collapse = ", "))
  }
  minutes
}


parse_condition_columns <- function(column_names,
                                    cell_line,
                                    condition,
                                    exclude = "full",
                                    control = "starve") {
  # Find and parse the log2:abs replicate columns of one cell line and condition.
  #
  # The `full` timepoint is dropped here rather than downstream, so nothing later has to
  # remember to exclude it. Columns are returned sorted onto the minute axis (control first) —
  # the on-disk order is not trusted, because Trendy assumes `tVectIn` is ascending.
  #
  # Args:
  #   column_names: all column names of the dataset.
  #   cell_line: cell line prefix, e.g. "WT".
  #   condition: single condition name, e.g. "EGF".
  #   exclude: timepoint label(s) to drop entirely, e.g. "full".
  #   control: timepoint label kept as t = 0.
  #
  # Returns:
  #   data.frame with one row per sample column, ordered by time then replicate, and the
  #   columns `column`, `timepoint`, `plex`, `minutes`.

  pattern <- paste0("^", cell_line, "_log2:abs_", condition, "_([^_]+)_(r[0-9]+)$")
  matched <- grep(pattern, column_names, value = TRUE)
  if (length(matched) == 0) {
    stop("parse_condition_columns: no log2:abs columns matched '", pattern, "'.")
  }

  fields <- regmatches(matched, regexec(pattern, matched))
  info <- data.frame(column    = matched,
                     timepoint = vapply(fields, `[`, character(1), 2),
                     plex      = vapply(fields, `[`, character(1), 3),
                     stringsAsFactors = FALSE)

  info <- info[!info$timepoint %in% exclude, , drop = FALSE]
  info$minutes <- timepoint_minutes(info$timepoint, control)
  info <- info[order(info$minutes, info$plex), , drop = FALSE]
  rownames(info) <- NULL
  info
}


# ── Pre-processing the matrix Trendy fits ────────────────────────────────────────────────

normalize_matrix <- function(M,
                             method) {
  # Apply between-sample (column) normalisation to the log2 matrix.
  #
  # Args:
  #   M: numeric matrix, sites in rows and samples in columns, on the log2 scale.
  #   method: "none" (unchanged) or "median" (centre every column on the grand median).
  #
  # Returns:
  #   The normalised matrix, same dimensions and dimnames.

  if (method == "none") {
    return(M)
  }
  if (method == "median") {
    column_medians <- apply(M, 2, median, na.rm = TRUE)
    return(sweep(M, 2, column_medians) + mean(column_medians))
  }
  stop("normalize_matrix: unknown method '", method, "'.")
}


center_plex <- function(M,
                        plex) {
  # Remove each replicate's per-site offset — the segmented-regression stand-in for limma's
  # `plex` blocking term.
  #
  # Each TMT plex has its own overall intensity offset. limma absorbs it with a `plex` factor;
  # Trendy has no blocking term, so a plex that ran 0.4 log2 units high adds a step to the
  # residuals of every site and flattens real slopes. For each site and plex the mean over that
  # plex's timepoints is subtracted and the site's grand mean added back, so the level of each
  # site is preserved while the between-plex offsets are removed.
  #
  # Costs 3 degrees of freedom per site (four plex means constrained to the site mean); with 24
  # samples that is affordable, and it is reported in the output as `plex_centered`.
  #
  # Args:
  #   M: numeric matrix, sites in rows, samples in columns, no NA.
  #   plex: character or factor vector giving the plex of each column of M.
  #
  # Returns:
  #   The centred matrix, same dimensions and dimnames.

  plex <- as.character(plex)
  grand <- rowMeans(M)
  out <- M

  for (level in unique(plex)) {
    columns <- which(plex == level)
    offset  <- rowMeans(M[, columns, drop = FALSE]) - grand
    out[, columns] <- M[, columns, drop = FALSE] - offset
  }
  out
}


build_time_vector <- function(sample_info,
                              axis = "index") {
  # Build the numeric time vector Trendy fits against, on the requested axis.
  #
  # Args:
  #   sample_info: data.frame from parse_condition_columns(), already time-ordered.
  #   axis: "index" (1..T by timepoint order), "minutes" (raw), or "log" (log10(minutes + 1)).
  #
  # Returns:
  #   Named numeric vector, one entry per sample column, names = column names.

  minutes_levels <- sort(unique(sample_info$minutes))

  values <- switch(axis,
                   index   = match(sample_info$minutes, minutes_levels),
                   minutes = sample_info$minutes,
                   log     = log10(sample_info$minutes + 1),
                   stop("build_time_vector: unknown axis '", axis, "'."))

  values <- as.numeric(values)
  names(values) <- sample_info$column
  values
}


axis_to_minutes <- function(x,
                            minutes_levels,
                            axis = "index",
                            interp = "log") {
  # Convert a fitted breakpoint from the fitting axis back to minutes.
  #
  # On the "index" axis a breakpoint at, say, 3.4 sits 40% of the way from the 3rd to the 4th
  # timepoint. "40% of the way" needs a scale: `interp = "log"` interpolates in log10(t + 1),
  # which is consistent with the reason the index axis was chosen (uniform index spacing is
  # near-equivalent to log-time on this grid), while "linear" interpolates in raw minutes and
  # will place the same breakpoint much later.
  #
  # Args:
  #   x: numeric vector of breakpoints on the fitting axis (may contain NA).
  #   minutes_levels: sorted unique timepoints in minutes, e.g. c(0, 2, 5, 10, 15, 90).
  #   axis: the axis the fit used, "index" / "minutes" / "log".
  #   interp: "log" or "linear", used only for the "index" axis.
  #
  # Returns:
  #   Numeric vector of breakpoint times in minutes, same length as x.

  if (axis == "minutes") {
    return(x)
  }
  if (axis == "log") {
    return(10 ^ x - 1)
  }
  if (axis != "index") {
    stop("axis_to_minutes: unknown axis '", axis, "'.")
  }

  index_levels <- seq_along(minutes_levels)
  if (interp == "linear") {
    return(approx(index_levels, minutes_levels, xout = x, rule = 2)$y)
  }
  if (interp == "log") {
    logged <- approx(index_levels, log10(minutes_levels + 1), xout = x, rule = 2)$y
    return(10 ^ logged - 1)
  }
  stop("axis_to_minutes: unknown interp '", interp, "'.")
}


# ── Reshaping the Trendy output ──────────────────────────────────────────────────────────

trend_labels <- function(trends) {
  # Translate Trendy's numeric trend codes into words.
  #
  # Args:
  #   trends: numeric vector of -1 / 0 / 1, possibly padded with NA.
  #
  # Returns:
  #   Character vector of "down" / "flat" / "up", with NA preserved.

  out <- rep(NA_character_, length(trends))
  out[which(trends ==  1)] <- "up"
  out[which(trends == -1)] <- "down"
  out[which(trends ==  0)] <- "flat"
  out
}


trendy_result_table <- function(fit_list,
                                minutes_levels,
                                max_k,
                                adj_r2_cut = 0.5,
                                axis = "index",
                                interp = "log") {
  # Flatten Trendy's per-site list into one tidy row per fitted site.
  #
  # Built from the raw fit list rather than from topTrendy() on purpose: topTrendy drops every
  # site below the adjusted-R2 cutoff, and a dropped site is not the same thing as a site with
  # no pattern. Here every fitted site gets a row and the cutoff becomes a flag, matching how
  # the sigmoid notebook reports `fit_ok` alongside the sites it excludes.
  #
  # Trend / slope / p-value vectors are NA-trimmed before use: some Trendy versions pad them to
  # maxK + 1, and an NA there would silently turn `responsive` into NA and inflate
  # `n_segments`.
  #
  # Args:
  #   fit_list: the list returned by results(trendy(...)), one element per site.
  #   minutes_levels: sorted unique timepoints in minutes, for the breakpoint conversion.
  #   max_k: maximum number of breakpoints, fixing the width of the output table.
  #   adj_r2_cut: adjusted-R2 threshold recorded in `r2_ok`.
  #   axis: the axis the fit used, passed to axis_to_minutes().
  #   interp: index-axis interpolation, passed to axis_to_minutes().
  #
  # Returns:
  #   data.frame with one row per site: identity, pattern, breakpoints (axis units and
  #   minutes), per-segment slopes / trends / p-values, adjusted R2 and quality flags.

  n_seg_max <- max_k + 1

  rows <- lapply(names(fit_list), function(site) {
    fit <- fit_list[[site]]

    trends <- as.numeric(fit$Segment.Trends)
    slopes <- as.numeric(fit$Segment.Slopes)
    pvals  <- as.numeric(fit$Segment.Pvalues)
    trends <- trends[!is.na(trends)]
    slopes <- slopes[seq_along(trends)]
    pvals  <- pvals[seq_along(trends)]

    breaks <- fit$Breakpoints
    breaks <- if (all(is.na(breaks))) numeric(0) else as.numeric(breaks[!is.na(breaks)])

    words <- trend_labels(trends)
    row <- data.frame(site           = site,
                      n_segments     = length(trends),
                      n_breakpoints  = length(breaks),
                      pattern        = paste(words, collapse = "-"),
                      first_trend    = words[1],
                      last_trend     = words[length(words)],
                      responsive     = any(trends != 0),
                      adj_r2         = as.numeric(fit$AdjustedR2),
                      stringsAsFactors = FALSE)

    # Fixed-width columns, padded with NA, so every site has the same schema regardless of how
    # many segments its own fit ended up with.
    for (i in seq_len(n_seg_max)) {
      row[[paste0("segment", i, "_slope")]]  <- if (i <= length(slopes)) slopes[i] else NA_real_
      row[[paste0("segment", i, "_trend")]]  <- if (i <= length(words))  words[i]  else NA_character_
      row[[paste0("segment", i, "_pvalue")]] <- if (i <= length(pvals))  pvals[i]  else NA_real_
    }
    for (i in seq_len(max_k)) {
      value <- if (i <= length(breaks)) breaks[i] else NA_real_
      row[[paste0("breakpoint", i, "_axis")]] <- value
      row[[paste0("breakpoint", i, "_min")]]  <- axis_to_minutes(value,
                                                                 minutes_levels,
                                                                 axis,
                                                                 interp)
    }
    row
  })

  out <- do.call(rbind, rows)
  out$r2_ok      <- out$adj_r2 >= adj_r2_cut
  out$pattern_ok <- out$r2_ok & out$responsive
  rownames(out) <- NULL
  out
}


pattern_census <- function(patterns,
                           groups = NULL) {
  # Tabulate how many sites follow each pattern, and what fraction that is.
  #
  # The direct analogue of `shape_census` in src/response_shapes.py, so the two summaries can
  # be read side by side.
  #
  # Args:
  #   patterns: character vector of pattern strings, one per site.
  #   groups: optional named list of logical vectors defining subsets to report separately.
  #
  # Returns:
  #   data.frame with one row per pattern: counts and percentages, most frequent first.

  counts <- sort(table(patterns), decreasing = TRUE)
  out <- data.frame(pattern = names(counts),
                    n       = as.integer(counts),
                    pct     = round(100 * as.integer(counts) / length(patterns), 1),
                    stringsAsFactors = FALSE)

  if (!is.null(groups)) {
    for (group_name in names(groups)) {
      subset_counts <- table(patterns[groups[[group_name]]])
      out[[paste0("n_", group_name)]] <- as.integer(subset_counts[out$pattern])
      out[[paste0("pct_", group_name)]] <-
        round(100 * as.integer(subset_counts[out$pattern]) / sum(groups[[group_name]]), 1)
    }
  }
  out[is.na(out)] <- 0
  out
}
