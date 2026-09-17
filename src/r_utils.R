# ─────────────────────────────────────────────────────────────────────────────────────────
# Shared R infrastructure for the .rmd notebooks in notebooks/01_preprocessing/.
#
# Nothing here is specific to a dataset, a platform or a statistical model: locating the
# project root, and reporting progress / timing / memory / warnings while a long fit runs.
# Extracted from limma_for_pvalues.rmd, limma_for_pvalues_diapasef.rmd,
# batch_correction_diapasef.rmd and trendy_temporal_patterns.rmd, which each carried a
# byte-identical copy.
#
#   source(file.path(PROJECT_ROOT, "src", "r_utils.R"))
#
# ⚠️ Verbosity. The originals read a `VERBOSE` global defined by the notebook. A sourced file
# must not depend on variables defined by whoever sourced it, so verbosity is now the
# `tmt.verbose` option. Set it once, next to the other parameters:
#
#   options(tmt.verbose = VERBOSE)
#
# Every function still takes `verbose` explicitly, so a single call can override it.
# ─────────────────────────────────────────────────────────────────────────────────────────


find_project_root <- function(start = getwd(),
                              markers = c("data", "src", "notebooks")) {
  # Locate the project root by walking up the directory tree from `start`.
  #
  # Makes a notebook runnable from any working directory — knitted (working directory =
  # the notebook's folder) or run chunk-by-chunk from a console rooted at the project.
  #
  # Args:
  #   start: directory to start searching from, default the current working directory.
  #   markers: directory names that must all be present for a folder to count as the root.
  #
  # Returns:
  #   Absolute path to the project root. Raises an error if no ancestor directory qualifies.

  path <- normalizePath(start, mustWork = TRUE)

  repeat {
    if (all(dir.exists(file.path(path, markers)))) {
      return(path)
    }
    parent <- dirname(path)
    if (identical(parent, path)) {
      stop("find_project_root: no ancestor of '", start, "' contains all of: ",
           paste(markers, collapse = ", "),
           ". Set PROJECT_ROOT manually to the TMT_Data_analysis folder.")
    }
    path <- parent
  }
}


# Module-local clock for checkpoint(). Not exported in any meaningful sense — the leading dot
# keeps it out of a bare ls().
.timing <- new.env(parent = emptyenv())


checkpoint <- function(label,
                       reset = FALSE,
                       verbose = getOption("tmt.verbose", TRUE)) {
  # Print a flushed, timestamped progress line with the time since the previous checkpoint.
  #
  # Args:
  #   label: description of the stage being reported.
  #   reset: TRUE to restart the total-elapsed clock (use at the start of a dataset or cell
  #     line).
  #   verbose: whether to print anything at all.
  #
  # Returns:
  #   Invisibly, seconds elapsed since the previous checkpoint.

  if (!verbose) {
    return(invisible(0))
  }

  now <- proc.time()[["elapsed"]]
  if (reset || is.null(.timing$start)) {
    .timing$start <- now
    .timing$last  <- now
  }
  since_last  <- now - .timing$last
  since_start <- now - .timing$start
  .timing$last <- now

  cat(sprintf("[%s | +%7.1fs | %7.1fs total] %s\n",
              format(Sys.time(), "%H:%M:%S"),
              since_last,
              since_start,
              label))
  flush.console()
  invisible(since_last)
}


stage <- function(label,
                  expr,
                  verbose = getOption("tmt.verbose", TRUE)) {
  # Run one stage, announcing it before evaluation and timing it after.
  #
  # `expr` is passed unevaluated and forced inside, so the "starting" line is guaranteed to be
  # printed and flushed before any work begins. A stage that hangs therefore leaves its own
  # name as the last thing on screen.
  #
  # Args:
  #   label: short description of the stage, e.g. "lmFit".
  #   expr: the expression to evaluate.
  #   verbose: whether to print the announcement and the timing.
  #
  # Returns:
  #   The value of `expr`.

  if (verbose) {
    cat(sprintf("[%s]   -> %-42s ", format(Sys.time(), "%H:%M:%S"), paste0(label, " ...")))
    flush.console()
  }

  started <- proc.time()[["elapsed"]]
  value   <- force(expr)
  elapsed <- proc.time()[["elapsed"]] - started

  if (verbose) {
    cat(sprintf("done in %7.1fs\n", elapsed))
    flush.console()
  }
  value
}


report_memory <- function(label = "",
                          verbose = getOption("tmt.verbose", TRUE)) {
  # Print the memory currently used by R, to expose swapping as a cause of a slow run.
  #
  # Args:
  #   label: optional tag printed alongside the figures.
  #   verbose: whether to print anything at all.
  #
  # Returns:
  #   Invisibly, total megabytes in use.

  if (!verbose) {
    return(invisible(0))
  }

  # gc() returns a variable number of columns — a "limit (Mb)" column appears on some platforms,
  # so the megabyte columns must be located by name rather than by position. The first "(Mb)"
  # column is current usage, the last is the peak since the session started.
  usage   <- gc(verbose = FALSE)
  mb_cols <- which(colnames(usage) == "(Mb)")
  used    <- sum(usage[, mb_cols[1]])
  peak    <- if (length(mb_cols) > 1) sum(usage[, mb_cols[length(mb_cols)]]) else NA_real_

  cat(sprintf("[%s]      memory: %6.0f MB in use%s  %s\n",
              format(Sys.time(), "%H:%M:%S"),
              used,
              if (is.na(peak)) "" else sprintf(", %6.0f MB peak", peak),
              label))
  flush.console()
  invisible(used)
}


muffle_warnings <- function(expr,
                            label = "stage",
                            verbose = getOption("tmt.verbose", TRUE)) {
  # Evaluate an expression, collecting warnings instead of emitting them one by one.
  #
  # limma emits one warning per problematic site; tens of thousands of them can dominate the
  # runtime of an IDE notebook even though the computation itself is fast. This counts them and
  # prints a summary of the distinct messages.
  #
  # Args:
  #   expr: expression to evaluate.
  #   label: tag used in the printed summary.
  #   verbose: whether to print the summary.
  #
  # Returns:
  #   The value of `expr`.

  collected <- character(0)

  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      collected <<- c(collected, conditionMessage(w))
      invokeRestart("muffleWarning")
    })

  if (length(collected) > 0 && verbose) {
    counts <- sort(table(collected), decreasing = TRUE)
    cat(sprintf("[%s]      %d warning(s) in %s:\n",
                format(Sys.time(), "%H:%M:%S"), length(collected), label))
    for (i in seq_len(min(5, length(counts)))) {
      cat(sprintf("           %6d x  %s\n", counts[[i]], names(counts)[i]))
    }
    flush.console()
  }
  value
}


say <- function(...,
                verbose = getOption("tmt.verbose", TRUE)) {
  # Print a progress line, honouring the verbosity switch.
  #
  # Args:
  #   ...: objects pasted together (with no separator) to form the message.
  #   verbose: whether to print anything at all.
  #
  # Returns:
  #   NULL, invisibly. Called for the side effect of writing to stdout.

  if (verbose) cat(..., "\n", sep = "")
  invisible(NULL)
}
