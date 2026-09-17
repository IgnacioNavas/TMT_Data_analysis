# =====================================================================================
# batch_correction_diapasef.R
#
# Batch correction of the hme1_lfq (hme1_diaPASEF) single-site abundance matrix, for
# CLUSTERING / PCA / CURVE VISUALISATION ONLY.
#
# -------------------------------------------------------------------------------------
# The two-track logic: SUBTRACT for the eyes, ADJUST for the test
# -------------------------------------------------------------------------------------
# There are two legitimate ways to deal with an additive batch offset, and this project
# uses both — for different consumers, on different matrices:
#
#   * ADJUST for the test.  The limma differential-analysis path
#     (`limma_for_pvalues_diapasef.rmd`) handles batch by putting it in its own design
#     matrix (`~ ... + batch`) and fitting the *uncorrected* values. The batch term is
#     estimated jointly with the biology, so the residual degrees of freedom, the
#     moderated variance and therefore the p-values are all correct.
#
#   * SUBTRACT for the eyes.  Clustering, PCA and profile plots have no design matrix to
#     put a nuisance term into — they see whatever numbers they are handed. For those,
#     and ONLY for those, the batch offset is subtracted from the data up front. That is
#     what this script produces.
#
# The corrected matrix written here MUST NOT be fed into limma, or into any other
# inference step. Its batch offsets have already been spent, so any test run on it would
# be anti-conservative (the residual df no longer match the model that produced the
# numbers). If inference needs changing, that is a change to the limma notebook, made
# deliberately — not a change of input file.
#
# -------------------------------------------------------------------------------------
# Why limma::removeBatchEffect, location-only
# -------------------------------------------------------------------------------------
#   * It is the SAME additive batch model the limma inference uses. The matrix we cluster
#     and the matrix we test then differ only in *where* the batch term was accounted for,
#     not in what the batch term is. ComBat/sva would fit a different (empirical-Bayes,
#     and for ComBat also multiplicative) model, so the picture and the statistics would
#     start telling slightly different stories.
#   * The batch effect here is an enrichment-run + LC-session SHIFT — a location effect.
#     A scale/variance correction would be fitting something we have no evidence for, and
#     on DIA data with per-run missingness a per-batch variance estimate is exactly the
#     kind of quantity that is dominated by which sites happened to be quantified.
#   * `batch` is a plain FIXED effect with 3 levels. Three levels is far too few for
#     empirical-Bayes shrinkage of the batch term to mean anything; a fixed term gets the
#     bookkeeping right and nothing is gained by being clever.
#   * `design` carries the biology, which tells removeBatchEffect what to leave alone.
#
# NOT done here, deliberately: no normalisation (median / quantile / scaling — the
# `_None` unnormalised values are intentional), no zero-filling or imputation, no
# injection-order term (see INJECTION ORDER hook near the bottom), and the pooled
# reference injections are never written into the clustering matrix.
#
# -------------------------------------------------------------------------------------
# Usage
# -------------------------------------------------------------------------------------
#   Rscript notebooks/01_preprocessing/batch_correction_diapasef.R
#   Rscript notebooks/01_preprocessing/batch_correction_diapasef.R <abundance.tsv> <sample_sheet.tsv> [out_dir]
#
# All paths are variables in the CONFIGURATION block below; the command-line arguments
# only override them. Nothing is hard-coded to an absolute path.
# =====================================================================================

suppressPackageStartupMessages(library(limma))


# =====================================================================================
# PROJECT ROOT AND HELPERS
# =====================================================================================
# The helper functions live in src/, shared verbatim with the notebook form of this script
# (notebooks/01_preprocessing/batch_correction_diapasef.rmd). src/ has to be found before
# find_project_root() is available (it is defined in r_utils.R), hence the candidate loop:
# this script may be run from the repo root or from its own folder.

SRC_DIR <- NULL
for (candidate in c("src",
                    file.path("..", "..", "src"),
                    file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE),
                                                               value = TRUE)[1])), "..", "..", "src"))) {
  if (!is.na(candidate) && file.exists(file.path(candidate, "r_utils.R"))) {
    SRC_DIR <- normalizePath(candidate)
    break
  }
}
if (is.null(SRC_DIR)) {
  stop("cannot find src/ from '", getwd(),
       "'. Run this from the repo root, or from notebooks/01_preprocessing/.")
}
source(file.path(SRC_DIR, "r_utils.R"))                    # find_project_root, say
source(file.path(SRC_DIR, "batch_correction_diapasef.R"))  # output_path, read_tsv, ...

PROJECT_ROOT <- find_project_root()


# =====================================================================================
# CONFIGURATION — every path and switch lives here
# =====================================================================================

# --- inputs --------------------------------------------------------------------------
# Input 1: the filtered single-site diaPASEF abundance matrix, sites in rows, one column
#          per injection plus annotation columns. Deliberately the UNNORMALISED `_None`
#          variant — do not swap it for `_Norm`.
ABUNDANCE_FILE <- file.path(PROJECT_ROOT, "Experiment", "hme1_diaPASEF", "Data", "Processed",
                            "20260821_peptide_MS2quant_None_renamed.tsv")

# Input 2: the sample sheet, one row per injection. Written by
#          notebooks/01_preprocessing/LFQ_diaPASEF_batch_effect_correction.ipynb.
SAMPLE_SHEET_FILE <- file.path(PROJECT_ROOT, "Experiment", "hme1_diaPASEF", "Data",
                               "batch_info.tsv")

# --- outputs -------------------------------------------------------------------------
# Default: next to the abundance file. Never overwrites the input (suffixes below).
OUT_DIR <- dirname(ABUNDANCE_FILE)

OUT_SUFFIX_MATRIX  <- "_batchcorrected.tsv"        # the clustering matrix (references dropped)
OUT_SUFFIX_REF_QC  <- "_batchcorrected_refQC.tsv"  # per-site reference spread, before vs after
OUT_SUFFIX_REF_PNG <- "_batchcorrected_refQC.png"  # before/after diagnostic figure

# Refuse to clobber anything that already exists (project rule: never overwrite).
OVERWRITE <- FALSE

# --- sample sheet schema -------------------------------------------------------------
# The join key. Both spellings are accepted because the generating notebook writes
# `sample_Id` while the specification calls it `sample_id`; whichever is present is
# canonicalised to `sample_id` on load.
ID_COLUMN_CANDIDATES <- c("sample_id", "sample_Id", "sample_ID", "sampleId")

REQUIRED_SHEET_COLUMNS <- c("cell_line", "condition", "timepoint", "batch", "is_reference")

# --- scale detection -----------------------------------------------------------------
# removeBatchEffect operates on log-scale data. Real log2 phospho intensities live in
# roughly [5, 35]; linear detector intensities run to 1e5–1e9. Anything above this
# threshold is therefore unambiguously linear and gets log2-transformed once.
LOG2_MAX_PLAUSIBLE <- 64

# --- output scale --------------------------------------------------------------------
# "log2"   -> write the corrected values on the log2 scale the correction was done on,
#             and rename the DataType token of every sample column `raw:abs` -> `log2:abs`
#             (see RENAME_DATA_TYPE) so the header does not lie about what it holds.
# "linear" -> back-transform with 2^x so the file can be fed straight into
#             `run_diapasef_transformations()` in src/transformations.py, which expects
#             `raw:abs` linear intensities and does its own log2.
# Correction happens on log2 either way; this only chooses what is written to disk.
OUTPUT_SCALE <- "log2"

RENAME_DATA_TYPE <- c(from = "raw:abs", to = "log2:abs")

# --- annotation columns that are legitimately numeric --------------------------------
# Sample columns are identified ONLY by matching the header against the sheet's
# `sample_id` — never by string pattern. This list exists purely for the safety check
# below: a numeric column that is neither a known sample nor a known numeric annotation
# is almost certainly a sample missing from the sheet, and the script stops rather than
# silently leaving it uncorrected.
NUMERIC_ANNOTATION_COLUMNS <- c("Index", "peptide_index", "Start", "End", "Peptide Length",
                                "Probability", "Multiplicity", "Precursor Count",
                                "Best Localization", "Best Scan for Localization",
                                "protein_length", "nrPeptides", "nr_tryptic_peptides",
                                "NumPhos", "LocalizedNumPhos", "MaxPepProb", "AScore",
                                "n:reps", "ReferenceIntensity", "startModSite", "endModSite")

# --- injection-order drift hook (DISABLED — see the block at the bottom) --------------
USE_INJECTION_ORDER <- FALSE

# Re-fit the same linear model once more, purely to count how many sites had a batch
# coefficient that could not be estimated (see count_nonestimable). Costs one extra lmFit
# over the whole matrix; set FALSE to skip it on a very large file.
REPORT_NONESTIMABLE <- TRUE

VERBOSE <- TRUE
options(tmt.verbose = VERBOSE)   # read by say() in src/r_utils.R


# --- command-line overrides ----------------------------------------------------------
.cli <- commandArgs(trailingOnly = TRUE)
if (length(.cli) >= 1L) ABUNDANCE_FILE    <- .cli[[1L]]
if (length(.cli) >= 2L) SAMPLE_SHEET_FILE <- .cli[[2L]]
if (length(.cli) >= 3L) OUT_DIR           <- .cli[[3L]] else if (length(.cli) >= 1L) OUT_DIR <- dirname(ABUNDANCE_FILE)


# =====================================================================================
# HELPERS — MOVED to src/batch_correction_diapasef.R (and say() to src/r_utils.R)
# =====================================================================================
# Sourced above. The originals are kept below, commented out, so the two versions can
# be compared. Arguments that used to read a global (OUT_DIR, OVERWRITE,
# ID_COLUMN_CANDIDATES, REQUIRED_SHEET_COLUMNS, NUMERIC_ANNOTATION_COLUMNS,
# LOG2_MAX_PLAUSIBLE, VERBOSE) are now explicit; the call sites below pass them in.
#
# say <- function(...) {
#   # Print a progress line, honouring the VERBOSE switch.
#   #
#   # Args:
#   #   ...: objects pasted together to form the message.
#   #
#   # Returns:
#   #   NULL, invisibly. Called for the side effect of writing to stdout.
#
#   if (VERBOSE) cat(..., "\n", sep = "")
#   invisible(NULL)
# }
#
#
# output_path <- function(input_path,
#                         suffix,
#                         out_dir = OUT_DIR ) {
#   # Build an output path from the input filename plus a descriptive suffix.
#   #
#   # The input extension is stripped and the suffix appended, so the input file can never
#   # be the output file. Stops if the target exists and OVERWRITE is FALSE (project rule:
#   # never overwrite a processed data file, always version the name).
#   #
#   # Args:
#   #   input_path: path of the abundance matrix the output is derived from.
#   #   suffix: string appended to the input stem, including its own extension.
#   #   out_dir: directory to write into.
#   #
#   # Returns:
#   #   Absolute path of the file to write.
#
#   stem <- sub("\\.[A-Za-z0-9]+$", "", basename(input_path))
#   path <- file.path(out_dir, paste0(stem, suffix))
#
#   if (file.exists(path) && !OVERWRITE) {
#     stop("output_path: '", path, "' already exists and OVERWRITE is FALSE. ",
#          "Move it aside or add a version tag to the suffix.")
#   }
#   if (normalizePath(path, mustWork = FALSE) == normalizePath(input_path, mustWork = FALSE)) {
#     stop("output_path: refusing to write over the input file '", input_path, "'.")
#   }
#   path
# }
#
#
# read_tsv <- function(path,
#                      what ) {
#   # Read a project TSV, keeping column names verbatim and missing values as NA.
#   #
#   # `check.names = FALSE` is essential: the sample column names contain ':' and the
#   # header would otherwise be mangled into `WT_raw.abs_EGF_full_r1`, breaking the join
#   # against the sample sheet. Empty strings and the usual sentinels become NA — nothing
#   # is ever zero-filled (project rule).
#   #
#   # Args:
#   #   path: file to read.
#   #   what: short human-readable description used in error messages.
#   #
#   # Returns:
#   #   A data.frame with the file's columns in file order.
#
#   if (!file.exists(path)) {
#     stop("read_tsv: ", what, " not found at '", path, "'.")
#   }
#   df <- utils::read.delim(path,
#                           sep = "\t",
#                           header = TRUE,
#                           check.names = FALSE,
#                           stringsAsFactors = FALSE,
#                           na.strings = c("", "NA", "NaN", "nan", "None") )
#   say("  loaded ", what, ": ", nrow(df), " rows x ", ncol(df), " columns")
#   df
# }
#
#
# canonicalise_sheet <- function(sheet ) {
#   # Validate the sample sheet and normalise its column names and types.
#   #
#   # Renames whichever id spelling is present to `sample_id`, checks the required columns
#   # exist, coerces `is_reference` from R logicals / "TRUE"/"FALSE" / 0/1 to a logical
#   # vector, and forces `batch` to a factor. Stops loudly on anything missing or ambiguous
#   # rather than guessing.
#   #
#   # Args:
#   #   sheet: data.frame as read from the sample-sheet TSV.
#   #
#   # Returns:
#   #   The same data.frame with `sample_id` (character), `batch` (factor),
#   #   `is_reference` (logical), and the remaining required columns present.
#
#   id_present <- intersect(ID_COLUMN_CANDIDATES, colnames(sheet))
#   if (length(id_present) == 0L) {
#     stop("canonicalise_sheet: the sample sheet has no id column. Expected one of: ",
#          paste(ID_COLUMN_CANDIDATES, collapse = ", "),
#          ". Found: ", paste(colnames(sheet), collapse = ", "))
#   }
#   if (length(id_present) > 1L) {
#     stop("canonicalise_sheet: the sample sheet has more than one id column (",
#          paste(id_present, collapse = ", "), "). Keep exactly one.")
#   }
#   colnames(sheet)[colnames(sheet) == id_present[[1L]]] <- "sample_id"
#
#   missing_cols <- setdiff(REQUIRED_SHEET_COLUMNS, colnames(sheet))
#   if (length(missing_cols) > 0L) {
#     stop("canonicalise_sheet: sample sheet is missing required column(s): ",
#          paste(missing_cols, collapse = ", "))
#   }
#
#   sheet$sample_id <- as.character(sheet$sample_id)
#   if (anyDuplicated(sheet$sample_id) > 0L) {
#     dups <- unique(sheet$sample_id[duplicated(sheet$sample_id)])
#     stop("canonicalise_sheet: duplicated sample_id in the sheet: ",
#          paste(dups, collapse = ", "))
#   }
#
#   # is_reference: accept logical, "TRUE"/"FALSE"/"True"/"true", or 0/1.
#   ref_raw <- sheet$is_reference
#   ref <- if (is.logical(ref_raw)) {
#     ref_raw
#   } else if (is.numeric(ref_raw)) {
#     ref_raw != 0
#   } else {
#     tolower(trimws(as.character(ref_raw))) %in% c("true", "t", "yes", "1")
#   }
#   if (anyNA(ref)) {
#     stop("canonicalise_sheet: `is_reference` could not be read as a logical for ",
#          sum(is.na(ref)), " row(s).")
#   }
#   sheet$is_reference <- ref
#
#   sheet$batch <- factor(as.character(sheet$batch))
#   if (nlevels(sheet$batch) < 2L) {
#     stop("canonicalise_sheet: `batch` has ", nlevels(sheet$batch),
#          " level(s). Nothing to correct — check the sample sheet.")
#   }
#
#   say("  sample sheet: ", nrow(sheet), " injections, ",
#       nlevels(sheet$batch), " batches (", paste(levels(sheet$batch), collapse = "/"), "), ",
#       sum(sheet$is_reference), " pooled references")
#   sheet
# }
#
#
# split_matrix_columns <- function(abundance,
#                                  sheet ) {
#   # Split the abundance table into sample columns and annotation columns.
#   #
#   # Sample columns are identified ONLY by matching the header against the sheet's
#   # `sample_id` — never by pattern-matching on the column name. Two failures are fatal:
#   # a `sample_id` with no matching column, and a numeric column that is in neither the
#   # sheet nor NUMERIC_ANNOTATION_COLUMNS (which almost always means a sample was left out
#   # of the sheet and would silently escape correction).
#   #
#   # Args:
#   #   abundance: data.frame as read from the abundance TSV.
#   #   sheet: canonicalised sample sheet.
#   #
#   # Returns:
#   #   A list with `sample_cols` (character, in sheet order) and `annot_cols` (character,
#   #   in file order).
#
#   sample_cols <- sheet$sample_id
#   missing_in_matrix <- setdiff(sample_cols, colnames(abundance))
#   if (length(missing_in_matrix) > 0L) {
#     stop("split_matrix_columns: ", length(missing_in_matrix),
#          " sample_id(s) in the sheet have no column in the abundance matrix: ",
#          paste(utils::head(missing_in_matrix, 20L), collapse = ", "),
#          if (length(missing_in_matrix) > 20L) " ..." else "")
#   }
#
#   annot_cols <- setdiff(colnames(abundance), sample_cols)
#
#   unexplained <- annot_cols[vapply(abundance[annot_cols], is.numeric, logical(1))]
#   unexplained <- setdiff(unexplained, NUMERIC_ANNOTATION_COLUMNS)
#   if (length(unexplained) > 0L) {
#     stop("split_matrix_columns: ", length(unexplained),
#          " numeric column(s) are neither a sample_id in the sheet nor a known annotation: ",
#          paste(unexplained, collapse = ", "),
#          ". If these are injections, add them to the sample sheet; if they are annotations, ",
#          "add them to NUMERIC_ANNOTATION_COLUMNS.")
#   }
#
#   say("  columns: ", length(sample_cols), " samples, ", length(annot_cols), " annotations")
#   list(sample_cols = sample_cols, annot_cols = annot_cols )
# }
#
#
# to_log2_if_needed <- function(x ) {
#   # Put the intensity matrix on the log2 scale, exactly once.
#   #
#   # `removeBatchEffect` fits a linear model, so an additive batch offset only *is*
#   # additive on the log scale. Linear intensities are detected by their range (see
#   # LOG2_MAX_PLAUSIBLE) and transformed; values that are already log2 are left alone so
#   # the matrix is never double-logged. Non-positive values become NA, NOT zero — a zero
#   # intensity in DIA is a non-detection, and log2(0) = -Inf would drag the linear fit.
#   # No normalisation of any kind is applied.
#   #
#   # Args:
#   #   x: numeric matrix of intensities, samples in columns.
#   #
#   # Returns:
#   #   A list with `x` (the log2-scale matrix) and `was_linear` (logical, which branch ran).
#
#   finite_vals <- x[is.finite(x)]
#   if (length(finite_vals) == 0L) {
#     stop("to_log2_if_needed: the sample matrix contains no finite values.")
#   }
#
#   rng <- range(finite_vals)
#   was_linear <- rng[[2L]] > LOG2_MAX_PLAUSIBLE
#
#   say("  value range: [", signif(rng[[1L]], 4), ", ", signif(rng[[2L]], 4), "], ",
#       "median ", signif(stats::median(finite_vals), 4))
#
#   if (was_linear) {
#     say("  scale check -> LINEAR intensities: applying log2 (non-positive -> NA)")
#     x[!is.na(x) & x <= 0] <- NA_real_
#     x <- log2(x)
#   } else {
#     say("  scale check -> already LOG2: no transformation applied (not double-logging)")
#     x[!is.finite(x)] <- NA_real_
#   }
#
#   say("  missing values: ", sum(is.na(x)), " / ", length(x),
#       " (", sprintf("%.1f%%", 100 * mean(is.na(x))), ") — kept as NA, never zero-filled")
#
#   list(x = x, was_linear = was_linear )
# }
#
#
# build_bio_group <- function(sheet ) {
#   # Build the biological factor that the correction must PROTECT.
#   #
#   # One level per (cell_line, condition, timepoint) combination — the finest biological
#   # unit in the experiment, so nothing biological is pooled and therefore nothing
#   # biological can be absorbed into the batch term. The pooled references are overridden
#   # to a single "reference" level: they are the same material in all three batches, so
#   # giving them their own free mean stops that material's overall abundance from being
#   # read as a biological group, while still letting them ride along and be corrected.
#   #
#   # Note that BRAF biological replicates 1 and 2 are distinct `cell_line` levels in the
#   # sheet, so they stay distinct here — they are not pooled.
#   #
#   # Args:
#   #   sheet: canonicalised sample sheet, rows in the order the matrix columns will be in.
#   #
#   # Returns:
#   #   A factor with one entry per injection, levels made syntactically valid.
#
#   grp <- paste(sheet$cell_line, sheet$condition, sheet$timepoint, sep = ".")
#   grp[sheet$is_reference] <- "reference"
#   factor(make.names(grp))
# }
#
#
# check_batch_orthogonality <- function(bio_group,
#                                       batch ) {
#   # Report how batch and biology are laid out against each other.
#   #
#   # The design of this experiment puts every biological group in every batch exactly once,
#   # which makes `batch` orthogonal to `bio_group`. That is the reason the correction
#   # cannot erode cell-line / condition / time effects: there is no biological contrast
#   # that is even partly a batch contrast, so subtracting the batch offsets removes a
#   # direction the biology does not live in. This function checks that assumption instead
#   # of assuming it, and warns (rather than stops) if the layout is imperfect — a few
#   # groups missing a batch is a missingness problem, not a design failure.
#   #
#   # Args:
#   #   bio_group: factor of biological groups, one entry per injection.
#   #   batch: factor of batches, one entry per injection.
#   #
#   # Returns:
#   #   The group x batch contingency table, invisibly.
#
#   tab <- table(bio_group, batch)
#   balanced <- all(tab == 1L)
#
#   if (balanced) {
#     say("  design: each biological group appears exactly once per batch -> ",
#         "batch is orthogonal to bio_group; the correction cannot touch the biology")
#   } else {
#     n_unbal <- sum(apply(tab, 1L, function(r) length(unique(r)) > 1L))
#     warning("check_batch_orthogonality: ", n_unbal, " of ", nrow(tab),
#             " biological groups are not evenly spread across batches. ",
#             "Batch and biology are partially confounded for those groups; ",
#             "the correction there is a compromise, not a clean removal.",
#             call. = FALSE)
#     say("  design: NOT fully balanced — ", n_unbal, " group(s) unevenly spread over batches")
#   }
#
#   invisible(tab)
# }
#
#
# count_nonestimable <- function(x,
#                                design,
#                                batch ) {
#   # Count, per site, how many batch coefficients could not be estimated.
#   #
#   # A site that is too sparsely quantified in some batch leaves that batch's offset
#   # undetermined; limma returns NA and `removeBatchEffect` silently substitutes 0, i.e.
#   # that part of the correction is simply not applied. That is expected under DIA
#   # missingness and acceptable — but it should be visible rather than silent, which the
#   # single "Partial NA coefficients" warning does not make it.
#   #
#   # This re-fits the exact model `removeBatchEffect` fits internally (the batch factor in
#   # sum-to-zero coding, projected onto the orthogonal complement of `design`) purely to
#   # read the NA pattern out of the coefficients. It does NOT produce the corrected values
#   # — those come from `removeBatchEffect` itself, so there is one implementation of the
#   # correction, not two.
#   #
#   # Args:
#   #   x: log2-scale matrix, sites in rows, injections in columns.
#   #   design: biological design matrix passed to removeBatchEffect.
#   #   batch: factor of batches, one entry per injection.
#   #
#   # Returns:
#   #   Integer vector, one entry per site: the number of NA batch coefficients (0 = fully
#   #   corrected, ncol(batch contrasts) = not corrected at all).
#
#   b <- as.factor(batch)
#   stats::contrasts(b) <- stats::contr.sum(levels(b))
#   batch_mat <- stats::model.matrix(~ b)[, -1L, drop = FALSE]
#   batch_mat <- qr.resid(qr(design), batch_mat)
#
#   fit <- suppressWarnings(limma::lmFit(x, cbind(design, batch_mat)))
#   beta <- fit$coefficients[, -seq_len(ncol(design)), drop = FALSE]
#   rowSums(is.na(beta))
# }
#
#
# reference_spread <- function(x,
#                              ref_cols ) {
#   # Per-site technical spread across the pooled reference injections.
#   #
#   # The references are the same physical material injected once per batch, so every
#   # difference between them is technical. Only sites quantified in ALL references are
#   # evaluable — under DIA missingness that is a subset, which is fine: this is a
#   # spot-check, not a per-site gate.
#   #
#   # CV is reported on the log2 scale as sd/|mean| * 100, which is a ratio of log2 units,
#   # not the linear-scale CV. SD in log2 units is the primary number; the CV column is
#   # kept because it is what the instruction asks for and it is scale-free across sites.
#   #
#   # Args:
#   #   x: log2-scale matrix, sites in rows.
#   #   ref_cols: column names of the reference injections.
#   #
#   # Returns:
#   #   A data.frame with one row per site: `n_obs`, `mean`, `sd`, `cv` (NA where the site
#   #   is not observed in every reference).
#
#   m <- x[, ref_cols, drop = FALSE]
#   n_obs <- rowSums(!is.na(m))
#   complete <- n_obs == length(ref_cols)
#
#   mu <- rep(NA_real_, nrow(m))
#   sd <- rep(NA_real_, nrow(m))
#   mu[complete] <- rowMeans(m[complete, , drop = FALSE])
#   sd[complete] <- apply(m[complete, , drop = FALSE], 1L, stats::sd)
#
#   data.frame(n_obs = n_obs,
#              mean = mu,
#              sd = sd,
#              cv = 100 * sd / abs(mu),
#              row.names = rownames(m),
#              stringsAsFactors = FALSE )
# }
#
#
# plot_reference_qc <- function(before,
#                               after,
#                               path ) {
#   # Save the before/after reference-spread diagnostic.
#   #
#   # Left: overlaid histograms of per-site SD across the references. Right: scatter of
#   # before vs after with the identity line — points below the line are sites whose
#   # technical spread shrank. Success is the whole cloud sitting below the diagonal.
#   #
#   # Args:
#   #   before: data.frame from reference_spread() on the uncorrected values.
#   #   after: data.frame from reference_spread() on the corrected values.
#   #   path: PNG file to write.
#   #
#   # Returns:
#   #   `path`, invisibly.
#
#   ok <- !is.na(before$sd) & !is.na(after$sd)
#   b <- before$sd[ok]
#   a <- after$sd[ok]
#
#   grDevices::png(path, width = 1600, height = 750, res = 140)
#   on.exit(grDevices::dev.off(), add = TRUE)
#   graphics::par(mfrow = c(1L, 2L), mar = c(4.5, 4.5, 3, 1))
#
#   brks <- graphics::hist(c(b, a), breaks = 40L, plot = FALSE)$breaks
#   hb <- graphics::hist(b, breaks = brks, plot = FALSE)
#   ha <- graphics::hist(a, breaks = brks, plot = FALSE)
#   graphics::plot(hb, col = grDevices::adjustcolor("grey40", 0.6), border = NA,
#                  ylim = c(0, max(hb$counts, ha$counts)),
#                  main = "Reference spread per site",
#                  xlab = "SD across mix / mixb / mixc (log2)", ylab = "sites" )
#   graphics::plot(ha, col = grDevices::adjustcolor("#1f77b4", 0.6), border = NA, add = TRUE)
#   graphics::abline(v = stats::median(b), col = "grey20", lwd = 2, lty = 2)
#   graphics::abline(v = stats::median(a), col = "#1f77b4", lwd = 2, lty = 2)
#   graphics::legend("topright", bty = "n",
#                    fill = c(grDevices::adjustcolor("grey40", 0.6),
#                             grDevices::adjustcolor("#1f77b4", 0.6)),
#                    legend = c(sprintf("before (median %.3f)", stats::median(b)),
#                               sprintf("after  (median %.3f)", stats::median(a))) )
#
#   lim <- range(c(b, a), finite = TRUE)
#   graphics::plot(b, a, pch = 16, cex = 0.35,
#                  col = grDevices::adjustcolor("black", 0.25),
#                  xlim = lim, ylim = lim,
#                  main = sprintf("%.1f%% of sites improved", 100 * mean(a < b)),
#                  xlab = "SD before (log2)", ylab = "SD after (log2)" )
#   graphics::abline(0, 1, col = "red", lwd = 2)
#
#   invisible(path)
# }


# =====================================================================================
# 1. LOAD
# =====================================================================================

say("\n=== 1. Load ===")
say("  abundance : ", ABUNDANCE_FILE)
say("  sheet     : ", SAMPLE_SHEET_FILE)

abundance <- read_tsv(ABUNDANCE_FILE, "abundance matrix")
sheet     <- canonicalise_sheet(read_tsv(SAMPLE_SHEET_FILE, "sample sheet"),
                                id_candidates    = ID_COLUMN_CANDIDATES,
                                required_columns = REQUIRED_SHEET_COLUMNS)

cols <- split_matrix_columns(abundance,
                             sheet,
                             numeric_annotation_columns = NUMERIC_ANNOTATION_COLUMNS)
sample_cols <- cols$sample_cols
annot_cols  <- cols$annot_cols

# Order the matrix columns and the sheet rows identically before fitting. Everything
# downstream indexes positionally, so this is the single point where the two inputs are
# aligned — and it is done by name, so it cannot silently mis-pair.
x <- as.matrix(abundance[, sample_cols, drop = FALSE])
storage.mode(x) <- "double"
rownames(x) <- if ("site" %in% annot_cols) abundance[["site"]] else rownames(abundance)
stopifnot(identical(colnames(x), sheet$sample_id))


# =====================================================================================
# 2. SCALE CHECK -> log2 IF NEEDED
# =====================================================================================

say("\n=== 2. Scale check ===")
scaled <- to_log2_if_needed(x, log2_max_plausible = LOG2_MAX_PLAUSIBLE)
x <- scaled$x


# =====================================================================================
# 3. MODEL
# =====================================================================================

say("\n=== 3. Model ===")

bio_group <- build_bio_group(sheet)
batch     <- sheet$batch

say("  bio_group: ", nlevels(bio_group), " levels (incl. 'reference'), ",
    "batch: ", nlevels(batch), " levels")

check_batch_orthogonality(bio_group, batch)

design <- stats::model.matrix(~ 0 + bio_group)
colnames(design) <- levels(bio_group)

rank_full <- qr(design)$rank
if (rank_full < ncol(design)) {
  stop("model: the biological design matrix is rank deficient (rank ", rank_full,
       " < ", ncol(design), " columns). Check for empty bio_group levels in the sheet.")
}
say("  design: ", nrow(design), " injections x ", ncol(design),
    " biological coefficients (full rank), plus ", nlevels(batch) - 1L, " batch coefficient(s)")

# -------------------------------------------------------------------------------------
# INJECTION ORDER — DISABLED HOOK, do not enable without evidence
# -------------------------------------------------------------------------------------
# Injection order was randomised WITHIN each batch, so LC drift is orthogonal to the
# biology: it does not bias any contrast, it only inflates within-batch variance. Adding
# an order term would spend degrees of freedom to buy nothing, and on sparse DIA sites it
# would start fitting the missingness pattern.
#
# Wire this in ONLY IF a QC plot of total phospho-signal vs injection order shows a clear
# within-batch monotonic drift. It would then go in as a `covariates` term, NOT as a
# second batch factor:
#
# if (USE_INJECTION_ORDER) {
#   stopifnot("injection_order" %in% colnames(sheet))
#   # centre within batch so the term captures drift, not a second batch offset
#   drift <- unlist(lapply(split(sheet$injection_order, sheet$batch),
#                          function(v) scale(v, center = TRUE, scale = FALSE)))
#   drift <- drift[order(unlist(split(seq_len(nrow(sheet)), sheet$batch)))]
#   covariates <- matrix(drift, ncol = 1L, dimnames = list(NULL, "injection_order"))
# } else {
#   covariates <- NULL
# }
# -------------------------------------------------------------------------------------


# =====================================================================================
# 4. CORRECT
# =====================================================================================

say("\n=== 4. removeBatchEffect ===")

# `batch`  — the axis to remove (location only; 3 levels, fixed, unshrunk).
# `design` — the biology to leave alone. removeBatchEffect projects the batch columns onto
#            the orthogonal complement of `design` before fitting, so anything a batch
#            column shares with a biological group is credited to the biology, not removed.
#
# Sites too sparse in a batch to estimate its offset get an NA coefficient, which
# removeBatchEffect sets to 0 — i.e. those sites are left UNCORRECTED. That is expected
# and acceptable under DIA missingness, not an error; the count is reported below.
corrected <- limma::removeBatchEffect(x,
                                      batch = batch,
                                      design = design )

say("  corrected ", nrow(corrected), " sites x ", ncol(corrected), " injections")

n_untouched <- sum(rowSums(abs(corrected - x), na.rm = TRUE) == 0)
say("  sites left completely unchanged (no batch coefficient estimable): ",
    n_untouched, " (", sprintf("%.1f%%", 100 * n_untouched / nrow(corrected)), ")")

if (REPORT_NONESTIMABLE) {
  na_counts <- count_nonestimable(x, design, batch)
  say("  sites with a non-estimable batch coefficient (NA -> 0, partial correction): ",
      sum(na_counts > 0), " (", sprintf("%.1f%%", 100 * mean(na_counts > 0)), ")",
      " — expected under DIA missingness, not an error")
}

# Sanity: the correction is location-only, so nothing may appear or disappear.
stopifnot(identical(dim(corrected), dim(x)))
stopifnot(identical(is.na(corrected), is.na(x)))
say("  missingness pattern unchanged (nothing imputed)")


# =====================================================================================
# 5. SPLIT AND WRITE
# =====================================================================================

say("\n=== 5. Split and write ===")

ref_cols <- sheet$sample_id[sheet$is_reference]
bio_cols <- sheet$sample_id[!sheet$is_reference]

if (length(ref_cols) < 2L) {
  warning("write: fewer than 2 reference injections found — the reference QC below ",
          "will be uninformative.", call. = FALSE)
}
say("  biological samples: ", length(bio_cols), " | references held out for QC: ",
    length(ref_cols))

out_matrix <- corrected[, bio_cols, drop = FALSE]

out_names <- bio_cols
if (identical(OUTPUT_SCALE, "linear")) {
  say("  OUTPUT_SCALE = 'linear': back-transforming with 2^x (correction was done on log2)")
  out_matrix <- 2 ^ out_matrix
} else if (identical(OUTPUT_SCALE, "log2")) {
  if (scaled$was_linear) {
    # The header would otherwise claim `raw:abs` while holding log2 values.
    out_names <- sub(RENAME_DATA_TYPE[["from"]], RENAME_DATA_TYPE[["to"]],
                     out_names, fixed = TRUE )
    say("  OUTPUT_SCALE = 'log2': DataType token renamed '",
        RENAME_DATA_TYPE[["from"]], "' -> '", RENAME_DATA_TYPE[["to"]], "' in ",
        sum(out_names != bio_cols), " column name(s)")
  }
} else {
  stop("OUTPUT_SCALE must be 'log2' or 'linear', got '", OUTPUT_SCALE, "'.")
}

# Annotation columns re-attached unchanged, in their original file order.
out_df <- cbind(abundance[, annot_cols, drop = FALSE],
                as.data.frame(out_matrix, check.names = FALSE ) )
colnames(out_df) <- c(annot_cols, out_names)

matrix_path <- output_path(ABUNDANCE_FILE, OUT_SUFFIX_MATRIX,
                           out_dir = OUT_DIR, overwrite = OVERWRITE)
utils::write.table(out_df, matrix_path,
                   sep = "\t", quote = FALSE, row.names = FALSE, na = "" )
say("  wrote clustering matrix: ", matrix_path)


# =====================================================================================
# 6. REFERENCE QC — did the correction work?
# =====================================================================================
#
# The three references are the same physical material injected once per batch, so every
# difference between them is technical. If the batch offsets are real and were estimated
# well, the references must collapse toward each other after correction.
#
# CAVEAT (the reason this is a near-independent validation rather than a circular one):
# the references are 3 of ~219 injections, so they contribute almost nothing to the batch
# estimate — that estimate is driven by the ~216 biological samples. The correction is
# therefore not "fitted to" the references in any meaningful sense, and their convergence
# is close to out-of-sample evidence. It is still only a spot-check: only sites quantified
# in all three references are evaluable, and that subset is biased toward abundant sites.
# =====================================================================================

say("\n=== 6. Reference QC ===")

if (length(ref_cols) >= 2L) {
  before <- reference_spread(x, ref_cols)
  after  <- reference_spread(corrected, ref_cols)

  evaluable <- !is.na(before$sd)
  say("  evaluable sites (quantified in all ", length(ref_cols), " references): ",
      sum(evaluable), " / ", nrow(x),
      " (", sprintf("%.1f%%", 100 * mean(evaluable)), ")")

  if (sum(evaluable) > 0L) {
    med_sd_before <- stats::median(before$sd[evaluable])
    med_sd_after  <- stats::median(after$sd[evaluable])
    med_cv_before <- stats::median(before$cv[evaluable], na.rm = TRUE)
    med_cv_after  <- stats::median(after$cv[evaluable], na.rm = TRUE)

    say("  median SD across references (log2):  ",
        sprintf("%.4f -> %.4f  (%+.1f%%)", med_sd_before, med_sd_after,
                100 * (med_sd_after / med_sd_before - 1)))
    say("  median CV across references (log2 units, %): ",
        sprintf("%.3f -> %.3f", med_cv_before, med_cv_after))
    say("  sites whose spread shrank: ",
        sprintf("%.1f%%", 100 * mean(after$sd[evaluable] < before$sd[evaluable])))

    if (med_sd_after >= med_sd_before) {
      warning("Reference QC: the references did NOT collapse (median SD ",
              sprintf("%.4f -> %.4f", med_sd_before, med_sd_after),
              "). Either the batch effect is not an additive shift, or the sample sheet's ",
              "`batch` assignment is wrong. Do not use the corrected matrix until this ",
              "is understood.", call. = FALSE)
    }

    qc_df <- data.frame(site = rownames(x),
                        n_refs_observed = before$n_obs,
                        sd_before = before$sd,
                        sd_after = after$sd,
                        cv_before = before$cv,
                        cv_after = after$cv,
                        stringsAsFactors = FALSE )
    qc_path <- output_path(ABUNDANCE_FILE, OUT_SUFFIX_REF_QC,
                           out_dir = OUT_DIR, overwrite = OVERWRITE)
    utils::write.table(qc_df, qc_path,
                       sep = "\t", quote = FALSE, row.names = FALSE, na = "" )
    say("  wrote reference QC table: ", qc_path)

    png_path <- output_path(ABUNDANCE_FILE, OUT_SUFFIX_REF_PNG,
                            out_dir = OUT_DIR, overwrite = OVERWRITE)
    plot_reference_qc(before, after, png_path)
    say("  wrote reference QC figure: ", png_path)
  } else {
    warning("Reference QC: no site is quantified in every reference — QC skipped.",
            call. = FALSE)
  }
} else {
  say("  skipped: fewer than 2 reference injections in the sheet")
}


say("\n=== Done ===")
say("The written matrix is for clustering / PCA / visualisation ONLY.")
say("Statistical testing stays on the UNCORRECTED values with `batch` in the design.")

invisible(NULL)
