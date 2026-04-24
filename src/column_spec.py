"""
ColumnSpec: helper for selecting columns that follow the phosphoproteomics naming convention.

Expected column format:
    {CellLine}_{datatype}:{subtype}_{condition}_{timepoint}_{replicate}

Examples:
    WT_log2:FC_EGF_2            — WT cell line, log2 fold-change, EGF condition, 2-min timepoint
    WT_log2:FC_EGF_starve       — starvation control
    BRAFS151A_log2:FC_EGF_2_r1  — mutant cell line, replicate 1
    WT_log2:abs_INS_full_r3     — raw log2 absorbance replicate

Usage:
    # Select all log2:FC columns for WT, EGF condition, excluding the 'full' timepoint
    cols = ColumnSpec.select(df, cell_lines=["WT"], data_type="log2:FC",
                             conditions=["_EGF_"], exclude_full=True)

    # Check whether a single column matches a spec
    spec = ColumnSpec(cell_line="WT", data_type="log2:FC", condition="_EGF_")
    spec.matches("WT_log2:FC_EGF_2")   # True
    spec.matches("WT_log2:abs_EGF_2")  # False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


@dataclass
class ColumnSpec:
    """Represents the naming-convention constraints for a single set of columns."""

    cell_line: str
    data_type: str        # e.g. "log2:FC", "log2:abs", "raw:abs"
    condition: str = ""   # e.g. "_EGF_", "_INS_", "_EGFnINS_"  (with underscores as delimiters)
    timepoint: str = ""   # e.g. "2", "starve", "full"; empty = match all timepoints
    replicate: str = ""   # e.g. "r1", "r2"; empty = match all replicates

    def matches(self, col: str) -> bool:
        """Return True if *col* satisfies all constraints in this ColumnSpec."""
        if not col.startswith(self.cell_line):
            return False
        parts = col.split("_")
        # parts[1] is the datatype field (e.g. "log2:FC")
        if len(parts) < 2 or parts[1] != self.data_type:
            return False
        if self.condition and self.condition not in col:
            return False
        if self.timepoint and self.timepoint not in col:
            return False
        if self.replicate and self.replicate not in col:
            return False
        return True

    # ------------------------------------------------------------------
    # Class-level helper — the main entry point for most use cases
    # ------------------------------------------------------------------

    @classmethod
    def select(
        cls,
        df: pd.DataFrame,
        cell_lines: List[str],
        data_type: str,
        conditions: List[str],
        exclude_full: bool = False,
        exclude_replicate_cols: bool = False,
    ) -> List[str]:
        """
        Return a list of column names from *df* that match any combination of
        (cell_line, data_type, condition).

        Args:
            df: DataFrame whose columns follow the naming convention.
            cell_lines: e.g. ["WT", "BRAFS151A"].
            data_type: e.g. "log2:FC".
            conditions: e.g. ["_EGF_", "_INS_"].
            exclude_full: if True, drop columns whose name contains 'full'.
            exclude_replicate_cols: if True, drop columns that end in _r1, _r2, etc.
        """
        selected: List[str] = []
        for cell in cell_lines:
            for cond in conditions:
                spec = cls(cell_line=cell, data_type=data_type, condition=cond)
                matched = [c for c in df.columns if spec.matches(c)]
                if exclude_full:
                    matched = [c for c in matched if "full" not in c]
                if exclude_replicate_cols:
                    matched = [c for c in matched if not _ends_with_replicate(c)]
                selected.extend(matched)
        return selected

    @classmethod
    def timepoints_from(
        cls,
        df: pd.DataFrame,
        cell_line: str,
        data_type: str,
        condition: str,
    ) -> List[str]:
        """
        Infer the ordered list of timepoint labels present in *df* for the given spec.

        Returns labels extracted from the 4th underscore-delimited field of matching
        column names (index 3), preserving column order.
        """
        spec = cls(cell_line=cell_line, data_type=data_type, condition=condition)
        matched = [c for c in df.columns if spec.matches(c)]
        seen: dict = {}
        for col in matched:
            parts = col.split("_")
            if len(parts) >= 4:
                tp = parts[3]
                seen[tp] = None  # use dict to preserve insertion order & deduplicate
        return list(seen.keys())


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _ends_with_replicate(col: str) -> bool:
    """Return True if the column name ends with a replicate suffix like '_r1'."""
    import re
    return bool(re.search(r"_r\d+$", col))
