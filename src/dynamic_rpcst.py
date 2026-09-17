"""
Dynamic EGF kinase-phosphosite network optimisation (dynamic RPCST).

Helper functions for `notebooks/07_dynamics_RPCST/dynamic_network_optimization_tutorial.ipynb`.
The notebook keeps the narrative, the parameters and the calls; everything it defines lives here.

The pipeline is:

  prepare_peak_fc_table   per-peptide EGF table  -> one row per phosphosite + its peak response
  load_kinsub             pickled kinase-substrate edges (optionally filtered by resource)
  load_tf_protein_ids     the transcription factors that define the allowed output terminals
  build_candidate_graph   ROOT -> EGFR -> (kinase -> site -> kinase)* -> TF site -> SINK
  prepare_solver_inputs   prizes, measured times, receptors, TF terminals, latent-time kinases
  dynamic_rpcst_selection the ILP: a temporally ordered, rooted, prize-collecting subnetwork
  temporal_violations     post-hoc check that no selected edge runs backwards in time
  write_selected_outputs  pickle + node/edge CSVs + summary JSON
  write_graphviz_dot      DOT rendering of the selected network

⚠️ Moved verbatim from the notebook: behaviour is unchanged, including the sharp edges recorded
in the docstrings below. The only deliberate differences are that values which used to be read
from notebook globals (TOP_FRACTION, NODE_COST, SOLVER, MIP_GAP, TIME_LIMIT_S, SOURCE_PARQUET,
TIME_COLORS) are now explicit arguments with module-level defaults, so a sourced module never
depends on variables defined by whoever imported it.

Requires: pandas, numpy, networkx, cvxpy, and a MILP solver (GUROBI preferred; SCIPY/HiGHS works
for small graphs). Graphviz `dot` is needed only for rendering.
"""

from __future__ import annotations

import html
import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd

# ── Defaults, formerly notebook globals ──────────────────────────────────────────────────
ROOT = "ROOT"
SINK = "SINK"
EGFR_UNIPROT = "P00533"

# The experimental EGF time grid a latent kinase activation time may be drawn from. 0 is included
# so a kinase may sit upstream of the earliest measured site; the measured grid itself is
# {2, 5, 10, 15, 90}.
ALLOWED_ACTIVATION_TIMES = (0, 2, 5, 10, 15, 90)

DEFAULT_TOP_FRACTION = 0.10
DEFAULT_NODE_COST = 1.2
DEFAULT_MIP_GAP = 0.05
DEFAULT_TIME_LIMIT_S = 180

# Fill colour of a phosphosite node, keyed by its measured peak timepoint in minutes.
TIME_COLORS = {
    2: "#2563eb",
    5: "#16a34a",
    10: "#f59e0b",
    15: "#dc2626",
    90: "#7c3aed",
}


def default_solver() -> str:
    """
    Pick the best mixed-integer solver CVXPY can actually see in this environment.

    GUROBI is strongly preferred for the full top-10% graph; GLPK_MI and SCIPY (HiGHS) solve
    small examples but are usually too slow for it.

    Args:
      None.

    Returns:
      Name of the first available solver among GUROBI, GLPK_MI and SCIPY, falling back to the
      string "SCIPY" when none of them is installed.
    """
    return next((s for s in ("GUROBI", "GLPK_MI", "SCIPY") if s in cp.installed_solvers()), "SCIPY")


# ── Paths ────────────────────────────────────────────────────────────────────────────────


def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the project root by walking upwards until a directory containing a data/ folder is found.

    Args:
      start: Directory to start the upward search from. Defaults to the current working
        directory when None.

    Returns:
      Path of the first directory, searching from start upwards, that contains a data/
      subdirectory. Raises RuntimeError if no such directory exists on the path to the
      filesystem root.
    """
    start = (start or Path.cwd()).resolve()
    for path in [start, *start.parents]:
        if (path / "data").exists():
            return path
    raise RuntimeError("Could not find a project root containing data/")


# ── 1. The dynamic phosphosite table ─────────────────────────────────────────────────────


def make_protein_site_id(df: pd.DataFrame) -> pd.Series:
    """
    Build the human-readable phosphosite identifier used to label nodes in the network.

    The identifier is {protein_name}_{PhosSites}, with the ';' separating multiple sites on the
    same peptide replaced by '_', so a two-site peptide becomes e.g. 'EGFR_T1191_Y1197'.

    Args:
      df: DataFrame carrying the FragPipe columns protein_name, protein_Id and PhosSites.
        protein_Id is used as a fallback wherever protein_name is missing.

    Returns:
      Series of string identifiers aligned to the index of df.
    """
    protein = df["protein_name"].fillna(df["protein_Id"]).astype(str)
    phosphosite = df["PhosSites"].astype(str).str.replace(";", "_", regex=False)
    return protein + "_" + phosphosite


def prepare_peak_fc_table(source_parquet: Path) -> pd.DataFrame:
    """
    Collapse the per-peptide EGF table into one row per phosphosite carrying its peak response.

    Keeps only localized rows (PhosSites not null), reads the pre-computed peak timepoint from
    WT_peak:FC_EGF, and picks the log2 fold change at that timepoint. Where several peptide rows
    map to the same phosphosite the row with the largest absolute peak fold change is kept, with
    n:reps and then ReferenceIntensity as tie-breakers.

    Args:
      source_parquet: Path to the processed hme1_2 EGF parquet file, which must contain
        protein_name, protein_Id, PhosSites, WT_peak:FC_EGF, n:reps, ReferenceIntensity and the
        WT_log2:FC_EGF_{2,5,10,15,90} columns.

    Returns:
      DataFrame with one row per protein_site_id, sorted by descending
      peak_abs_log2_fc_vs_starve, holding the identifier columns (protein_site_id,
      kinsub_site_id, protein_name, protein_Id, PhosSites), the response columns
      (peak_fc_time_min, peak_log2_fc_vs_starve, peak_abs_log2_fc_vs_starve,
      peak_linear_fc_vs_starve) and the provenance columns (n_source_rows, n:reps, site,
      peptide_seq, SequenceWindow).
    """
    df = pd.read_parquet(source_parquet)
    localized = df[df["PhosSites"].notna()].copy()
    localized["protein_site_id"] = make_protein_site_id(localized)
    localized["kinsub_site_id"] = localized["protein_Id"].astype(str) + "_" + localized["PhosSites"].astype(str)

    peak_time = localized["WT_peak:FC_EGF"].astype(str)
    localized["peak_fc_time_min"] = peak_time.astype("Int64")
    localized["peak_log2_fc_vs_starve"] = [
        row[f"WT_log2:FC_EGF_{time}"] for time, (_, row) in zip(peak_time, localized.iterrows())
    ]
    localized["peak_abs_log2_fc_vs_starve"] = localized["peak_log2_fc_vs_starve"].abs()
    localized["peak_linear_fc_vs_starve"] = 2.0 ** localized["peak_log2_fc_vs_starve"]

    source_counts = localized.groupby("protein_site_id").size().rename("n_source_rows")
    localized = localized.join(source_counts, on="protein_site_id")
    localized = localized.sort_values(
        ["protein_site_id", "peak_abs_log2_fc_vs_starve", "n:reps", "ReferenceIntensity"],
        ascending=[True, False, False, False],
        na_position="last",
    )
    site_level = localized.drop_duplicates("protein_site_id", keep="first")

    keep = [
        "protein_site_id", "kinsub_site_id", "protein_name", "protein_Id", "PhosSites",
        "peak_fc_time_min", "peak_log2_fc_vs_starve", "peak_abs_log2_fc_vs_starve",
        "peak_linear_fc_vs_starve", "n_source_rows", "n:reps", "site", "peptide_seq",
        "SequenceWindow",
    ]
    return site_level[keep].sort_values("peak_abs_log2_fc_vs_starve", ascending=False)


# ── 2. Kinase-substrate edges and transcription-factor terminals ─────────────────────────


def load_kinsub(path: Path, resource_regex: str | None = None) -> pd.DataFrame:
    """
    Load the kinase-substrate interaction table from a pickled DataFrame and optionally filter it.

    Args:
      path: Path to a pickle file containing a pandas DataFrame with the columns source (kinase
        UniProt accession), target (phosphosite in UniProt_site format), score (interaction
        confidence) and resource (the database or predictor the edge came from).
      resource_regex: Optional regular expression matched against the resource column; only
        matching rows are kept. None keeps every row.

    Returns:
      DataFrame restricted to the four required columns, filtered by resource_regex when given.
      Raises TypeError if the pickle does not hold a DataFrame and ValueError if any required
      column is absent.
    """
    with path.open("rb") as fh:
        kinsub = pickle.load(fh)
    if not isinstance(kinsub, pd.DataFrame):
        raise TypeError(f"Expected {path} to contain a pandas DataFrame, got {type(kinsub)!r}")
    required = {"source", "target", "score", "resource"}
    missing = required.difference(kinsub.columns)
    if missing:
        raise ValueError(f"Kinase-substrate dataframe missing columns: {sorted(missing)}")
    kinsub = kinsub[["source", "target", "score", "resource"]].copy()
    if resource_regex:
        keep = kinsub["resource"].astype(str).str.contains(resource_regex, regex=True, na=False)
        kinsub = kinsub.loc[keep].copy()
    return kinsub


def load_curated_tf_protein_ids(path: Path) -> Set[str]:
    """
    Read a user-supplied list of transcription-factor UniProt accessions.

    Args:
      path: Path to the curated list. A .csv or .tsv file is read as a table and must contain a
        protein_Id column; any other extension is read as a plain text file with one accession
        per line, ignoring blank lines and lines starting with '#'.

    Returns:
      Set of transcription-factor UniProt accessions as strings. Raises ValueError if a
      csv/tsv file lacks the protein_Id column.
    """
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        table = pd.read_csv(path, sep=sep)
        if "protein_Id" not in table.columns:
            raise ValueError(f"{path} must contain a protein_Id column")
        return set(table["protein_Id"].dropna().astype(str))
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def infer_tf_protein_ids(metadata_path: Path) -> tuple[Set[str], pd.DataFrame]:
    """
    Infer transcription factors from the UniProt protein descriptions carried in the dataset.

    Matches the description text against the case-insensitive pattern 'transcription factor',
    'transcription regulator', 'DNA-binding protein' or 'DNA binding protein'. This is a text
    heuristic, not a curated annotation, and it determines every terminal the optimisation is
    allowed to end on.

    Args:
      metadata_path: Path to the parquet file holding the protein_Id, protein_name and
        description columns.

    Returns:
      Tuple of (set of transcription-factor UniProt accessions, DataFrame of the matched rows
      with their protein_Id, protein_name and description for inspection).
    """
    metadata = pd.read_parquet(metadata_path, columns=["protein_Id", "protein_name", "description"]).drop_duplicates("protein_Id")
    description = metadata["description"].fillna("").astype(str)
    tf_pattern = re.compile(
        r"\b(?:transcription factor|transcription regulator|DNA-binding protein|DNA binding protein)\b",
        flags=re.IGNORECASE,
    )
    tf_table = metadata.loc[description.str.contains(tf_pattern, regex=True), ["protein_Id", "protein_name", "description"]].copy()
    return set(tf_table["protein_Id"].astype(str)), tf_table


def load_tf_protein_ids(metadata_path: Path, curated_path: Path | None = None) -> tuple[Set[str], pd.DataFrame, str]:
    """
    Return the transcription-factor set, preferring a curated list over description inference.

    Args:
      metadata_path: Path to the parquet file used when the TF set has to be inferred from
        protein descriptions.
      curated_path: Optional path to a curated TF list. When given, inference is skipped.

    Returns:
      Tuple of (set of TF UniProt accessions, DataFrame describing them, source label). The
      source label is 'curated' when curated_path was used and 'description_regex' otherwise;
      for the curated route the DataFrame holds only the protein_Id column.
    """
    if curated_path is not None:
        tf_ids = load_curated_tf_protein_ids(Path(curated_path))
        return tf_ids, pd.DataFrame({"protein_Id": sorted(tf_ids)}), "curated"
    tf_ids, tf_table = infer_tf_protein_ids(metadata_path)
    return tf_ids, tf_table, "description_regex"


# ── 3. The candidate graph ───────────────────────────────────────────────────────────────


def add_kinase_node(graph: nx.DiGraph, kinase_id: str, root_kinase: str) -> None:
    """
    Add a kinase node to the graph, marking whether its activation time is a free variable.

    Every kinase except the root receives latent_time=True, meaning the optimiser chooses its
    activation time from ALLOWED_ACTIVATION_TIMES. The root kinase is given latent_time=False
    because its time is pinned externally (to 0 in prepare_solver_inputs).

    Args:
      graph: Directed graph to add the node to; modified in place.
      kinase_id: UniProt accession of the kinase, used as the node key.
      root_kinase: UniProt accession of the receptor kinase that seeds the network.

    Returns:
      None. The graph is modified in place.
    """
    graph.add_node(
        kinase_id,
        node_type="kinase",
        latent_time=kinase_id != root_kinase,
        dynamic_exempt=False,
    )


def build_candidate_graph(
    kinsub: pd.DataFrame,
    dynamic_sites: pd.DataFrame,
    tf_protein_ids: Set[str],
    top_fraction: float = DEFAULT_TOP_FRACTION,
    root_kinase: str = EGFR_UNIPROT,
    root_name: str = ROOT,
    sink_name: str = SINK,
) -> tuple[nx.DiGraph, dict]:
    """
    Build the candidate kinase-phosphosite graph the optimisation will select a subnetwork from.

    Keeps the top fraction of phosphosites by absolute peak fold change, restricts the
    kinase-substrate table to edges pointing at them, and assembles four kinds of edge:
    ROOT to the root kinase, kinase to phosphosite (weighted by the interaction score),
    phosphosite to kinase where the modified protein is itself a kinase, and phosphosite to SINK
    where the modified protein is a transcription factor. The graph is finally pruned to the
    nodes reachable from ROOT.

    Args:
      kinsub: Kinase-substrate DataFrame as returned by load_kinsub.
      dynamic_sites: Per-site table as returned by prepare_peak_fc_table.
      tf_protein_ids: Set of transcription-factor UniProt accessions defining the sink terminals.
      top_fraction: Fraction of phosphosites retained, ranked by peak_abs_log2_fc_vs_starve.
      root_kinase: UniProt accession of the receptor kinase the network is rooted at.
      root_name: Key used for the artificial root node.
      sink_name: Key used for the artificial sink node.

    Returns:
      Tuple of (pruned directed graph, statistics dict). Phosphosite nodes carry
      protein_site_id, protein_name, protein_id, phosphosite, activation_time (the measured peak
      timepoint) and peak_log2_fc_vs_starve; kinase nodes carry latent_time. The statistics dict
      reports the site and edge counts before and after pruning and whether SINK stayed reachable
      from ROOT.
    """
    n_top = math.ceil(len(dynamic_sites) * top_fraction)
    top_sites = dynamic_sites.nlargest(n_top, "peak_abs_log2_fc_vs_starve").copy()
    if "kinsub_site_id" not in top_sites.columns:
        top_sites["kinsub_site_id"] = top_sites["protein_Id"].astype(str) + "_" + top_sites["PhosSites"].astype(str)
    top_sites = top_sites.drop_duplicates("kinsub_site_id", keep="first")

    top_site_ids = set(top_sites["kinsub_site_id"])
    kinase_ids = set(kinsub["source"].astype(str).unique())
    filtered = kinsub.loc[kinsub["target"].isin(top_site_ids)].copy()
    filtered = (
        filtered.groupby(["source", "target"], as_index=False)
        .agg(score=("score", "max"), resources=("resource", lambda x: ";".join(sorted(set(map(str, x))))))
    )

    graph = nx.DiGraph()
    graph.add_node(root_name, node_type="root", dynamic_exempt=True)
    graph.add_node(sink_name, node_type="sink", dynamic_exempt=True)
    add_kinase_node(graph, root_kinase, root_kinase)
    graph.add_edge(root_name, root_kinase, edge_type="root_to_kinase", weight=1.0)

    site_metadata = top_sites.set_index("kinsub_site_id").to_dict("index")
    for site_id, row in site_metadata.items():
        graph.add_node(
            site_id,
            node_type="phosphosite",
            protein_site_id=row["protein_site_id"],
            protein_name=row["protein_name"],
            protein_id=row["protein_Id"],
            phosphosite=row["PhosSites"],
            activation_time=int(row["peak_fc_time_min"]),
            peak_log2_fc_vs_starve=float(row["peak_log2_fc_vs_starve"]),
            peak_abs_log2_fc_vs_starve=float(row["peak_abs_log2_fc_vs_starve"]),
            dynamic_exempt=False,
        )

    for kinase_id in filtered["source"].astype(str).unique():
        add_kinase_node(graph, kinase_id, root_kinase)

    for row in filtered.itertuples(index=False):
        graph.add_edge(
            row.source,
            row.target,
            edge_type="kinase_to_phosphosite",
            weight=float(row.score),
            resources=row.resources,
        )

    kinase_site_count = 0
    tf_site_count = 0
    for site_id, row in site_metadata.items():
        protein_id = str(row["protein_Id"])
        if protein_id in kinase_ids:
            add_kinase_node(graph, protein_id, root_kinase)
            graph.add_edge(site_id, protein_id, edge_type="phosphosite_to_kinase", weight=1.0)
            kinase_site_count += 1
        if protein_id in tf_protein_ids:
            graph.add_edge(site_id, sink_name, edge_type="tf_phosphosite_to_sink", weight=1.0)
            tf_site_count += 1

    reachable = nx.descendants(graph, root_name) | {root_name}
    pruned = graph.subgraph(reachable).copy()
    stats = {
        "n_top_dynamic_sites": len(top_sites),
        "n_kinsub_rows_targeting_top_sites": int(kinsub["target"].isin(top_site_ids).sum()),
        "n_unique_kinase_site_edges_targeting_top_sites": len(filtered),
        "n_top_sites_on_kinase_proteins": kinase_site_count,
        "n_top_sites_on_tf_proteins": tf_site_count,
        "pre_prune_nodes": graph.number_of_nodes(),
        "pre_prune_edges": graph.number_of_edges(),
        "post_prune_nodes": pruned.number_of_nodes(),
        "post_prune_edges": pruned.number_of_edges(),
        "sink_reachable_from_root": sink_name in pruned,
    }
    return pruned, stats


# ── 4. The ILP ───────────────────────────────────────────────────────────────────────────


def dynamic_rpcst_selection(
    network: nx.DiGraph,
    prizes: Dict[str, float],
    activation_times: Optional[Dict[str, float]] = None,
    receptors: Optional[Sequence[str]] = None,
    tfs: Optional[Sequence[str]] = None,
    root_name: str = ROOT,
    sink_name: str = SINK,
    n_edges: Optional[int] = None,
    node_penalty: float = DEFAULT_NODE_COST,
    edge_weight_attr: str = "weight",
    time_attr: str = "activation_time",
    allow_equal_time: bool = True,
    drop_missing_times: bool = False,
    dynamic_exempt_nodes: Optional[Sequence[str]] = None,
    latent_time_nodes: Optional[Sequence[str]] = None,
    latent_time_allowed_values: Optional[Sequence[float]] = ALLOWED_ACTIVATION_TIMES,
    latent_time_penalty: float = 1e-6,
    solver: Optional[str] = None,
    verbose: int = 0,
    mip: float = DEFAULT_MIP_GAP,
    time_limit: int = DEFAULT_TIME_LIMIT_S,
    include_artificial_terminals: bool = False,
    return_problem: bool = False,
    receptor_kinase: str = EGFR_UNIPROT,
) -> object:
    """
    Select a temporally coherent rooted subnetwork by solving a prize-collecting ILP.

    Formulates and solves a mixed-integer program over binary node and edge variables. The
    objective trades unrealised phosphosite prize against edge cost and a per-node penalty, so a
    larger node_penalty yields a smaller network. Selection is constrained so that every chosen
    node has both incoming and outgoing selected support, the result is acyclic and rooted, and
    no selected edge runs backwards in time. Kinases listed as latent have their activation time
    chosen by the solver from a discrete set; phosphosites keep their measured peak time.

    Edges that already violate the temporal order between two known times are removed before the
    ILP is built, rather than being left for the solver to reject.

    Args:
      network: Candidate directed graph. Any existing root or sink node is stripped and replaced
        by artificial terminals created inside this function.
      prizes: Mapping of node key to prize collected when that node is selected. Negative values
        are clamped to zero.
      activation_times: Mapping of node key to fixed activation time in minutes. When None, the
        times are read from the time_attr node attribute.
      receptors: Nodes the artificial root connects to. When None, the root kinase is detected
        from the node attributes.
      tfs: Nodes that connect to the artificial sink. When None, they are detected from the
        existing tf_phosphosite_to_sink edges.
      root_name: Key used for the artificial root node.
      sink_name: Key used for the artificial sink node.
      n_edges: Exact number of biological edges to select. None lets the objective choose the
        network size.
      node_penalty: Cost charged for every selected biological node.
      edge_weight_attr: Edge attribute holding the interaction confidence. Edge cost is
        1 - weight, or zero when every weight equals 1.
      time_attr: Node attribute holding the measured activation time.
      allow_equal_time: When True, an edge may connect two nodes with the same activation time;
        when False, the source must be strictly earlier.
      drop_missing_times: When True, edges whose endpoints have no activation time are dropped;
        when False, a missing time on a latent-adjacent edge raises ValueError.
      dynamic_exempt_nodes: Nodes exempt from the temporal constraints, in addition to any node
        already carrying dynamic_exempt=True.
      latent_time_nodes: Nodes whose activation time the solver chooses, in addition to any node
        already carrying latent_time=True.
      latent_time_allowed_values: Discrete set of activation times a latent node may take.
      latent_time_penalty: Small weight on the sum of latent times, breaking ties towards earlier
        activation.
      solver: CVXPY solver name. None resolves to default_solver().
      verbose: Passed to the solver as its verbosity flag.
      mip: Relative MIP optimality gap at which the solver may stop.
      time_limit: Solver time limit in seconds.
      include_artificial_terminals: When True, the artificial root and sink are kept in the
        returned subgraph; when False they are removed.
      return_problem: Selects the return type, see below.
      receptor_kinase: UniProt accession used to detect the root kinase when receptors is None.

    Returns:
      When return_problem is False, the selected subgraph as a DiGraph whose nodes carry
      assigned_activation_time. When True, a dict holding that subgraph under 'subgraph' plus the
      solver status, objective value, the selected edges, the full candidate node and edge lists,
      the raw variable values, the chosen latent times and the CVXPY problem object. Note that
      'nodes' and 'edges' are the candidate lists, not the selected ones. Raises ValueError if no
      receptor or no TF terminal survives in the graph.
    """
    if solver is None:
        solver = default_solver()
    if receptors is None:
        receptors = [n for n, d in network.nodes(data=True)
                     if d.get("node_type") == "kinase" and n == receptor_kinase]
    if tfs is None:
        tfs = [u for u, v, d in network.edges(data=True) if v == sink_name and d.get("edge_type") == "tf_phosphosite_to_sink"]
    receptors = [n for n in receptors if n in network]
    tfs = [n for n in tfs if n in network]
    if not receptors:
        raise ValueError("dynamic_rpcst_selection requires at least one receptor/root kinase node")
    if not tfs:
        raise ValueError("dynamic_rpcst_selection requires at least one TF phosphosite output node")

    if activation_times is None:
        activation_times = {n: data[time_attr] for n, data in network.nodes(data=True) if time_attr in data}

    dynamic_exempt_nodes = set(dynamic_exempt_nodes or [])
    dynamic_exempt_nodes.update(n for n, d in network.nodes(data=True) if d.get("dynamic_exempt", False))
    latent_time_nodes = set(latent_time_nodes or [])
    latent_time_nodes.update(n for n, d in network.nodes(data=True) if d.get("latent_time", False))
    dynamic_exempt_nodes -= latent_time_nodes

    ph_net = nx.DiGraph()
    for node, data in network.nodes(data=True):
        if node not in {root_name, sink_name}:
            ph_net.add_node(node, **data)
    ph_net.add_node(root_name, node_type="artificial_root")
    ph_net.add_node(sink_name, node_type="artificial_sink")

    original_edges: list[tuple[str, str]] = []
    for u, v, data in network.edges(data=True):
        if u in {root_name, sink_name} or v in {root_name, sink_name}:
            continue
        is_dynamic_exempt_edge = u in dynamic_exempt_nodes or v in dynamic_exempt_nodes
        has_latent_time_endpoint = u in latent_time_nodes or v in latent_time_nodes
        if not is_dynamic_exempt_edge and has_latent_time_endpoint:
            for node in (u, v):
                if node not in latent_time_nodes and node not in activation_times:
                    if drop_missing_times:
                        break
                    raise ValueError(
                        f"Edge {(u, v)!r} is incident to a latent-time node, "
                        f"but non-latent endpoint {node!r} has no activation time"
                    )
            else:
                ph_net.add_edge(u, v, **data, artificial=False)
                original_edges.append((u, v))
            continue
        if not is_dynamic_exempt_edge:
            u_time = activation_times.get(u)
            v_time = activation_times.get(v)
            if u_time is None or v_time is None:
                if drop_missing_times:
                    continue
            elif allow_equal_time:
                if u_time > v_time:
                    continue
            elif u_time >= v_time:
                continue
        ph_net.add_edge(u, v, **data, artificial=False)
        original_edges.append((u, v))

    for receptor in receptors:
        ph_net.add_edge(root_name, receptor, weight=1.0, edge_type="root_to_kinase", artificial=True)
    for tf in tfs:
        ph_net.add_edge(tf, sink_name, weight=1.0, edge_type="tf_phosphosite_to_sink", artificial=True)

    nodes = list(ph_net.nodes())
    edges = list(ph_net.edges())
    node_indices = {node: i for i, node in enumerate(nodes)}
    edge_indices = {edge: i for i, edge in enumerate(edges)}
    source_indices = [node_indices[u] for u, _ in edges]
    target_indices = [node_indices[v] for _, v in edges]
    original_edge_indices = [edge_indices[e] for e in original_edges]
    prize_indices = [node_indices[n] for n in prizes if n in node_indices]
    prize_values = np.array([max(float(prizes[n]), 0.0) for n in prizes if n in node_indices])

    edge_weights = np.array([float(ph_net[u][v].get(edge_weight_attr, 1.0)) for u, v in edges])
    edge_costs = 1 - edge_weights if not np.all(edge_weights == 1) else np.zeros(len(edges))

    node_vars = cp.Variable(len(nodes), boolean=True)
    edge_vars = cp.Variable(len(edges), boolean=True)
    distance_vars = cp.Variable(len(nodes), integer=True)

    latent_nodes = [node for node in nodes if node in latent_time_nodes]
    latent_time_indices = {node: i for i, node in enumerate(latent_nodes)}
    latent_time_vars = cp.Variable(len(latent_nodes), integer=True) if latent_nodes else None

    M = len(nodes) + 1
    constraints = []

    # C1. Edge-to-node coupling.
    constraints += [
        edge_vars <= node_vars[source_indices],
        edge_vars <= node_vars[target_indices],
    ]

    in_edges: Dict[str, list[int]] = {node: [] for node in nodes}
    out_edges: Dict[str, list[int]] = {node: [] for node in nodes}
    for edge_i, (u, v) in enumerate(edges):
        out_edges[u].append(edge_i)
        in_edges[v].append(edge_i)

    # C2. Every selected non-root node has incoming selected support.
    for node in nodes:
        if node == root_name:
            continue
        incoming = in_edges[node]
        constraints.append(node_vars[node_indices[node]] <= cp.sum(edge_vars[incoming]) if incoming else node_vars[node_indices[node]] <= 0)

    # C3. Every selected non-sink node has outgoing selected continuation.
    for node in nodes:
        if node == sink_name:
            continue
        outgoing = out_edges[node]
        constraints.append(node_vars[node_indices[node]] <= cp.sum(edge_vars[outgoing]) if outgoing else node_vars[node_indices[node]] <= 0)

    # C4. Rooted acyclic ordering.
    constraints += [
        node_vars[node_indices[root_name]] == 1,
        node_vars[node_indices[sink_name]] == 1,
        distance_vars >= 0,
        distance_vars <= M * node_vars,
        distance_vars[node_indices[root_name]] == 0,
    ]
    constraints.append(cp.sum(edge_vars[out_edges[root_name]]) >= 1)
    constraints.append(cp.sum(edge_vars[in_edges[sink_name]]) >= 1)
    constraints.append(distance_vars[target_indices] >= distance_vars[source_indices] + 1 - M * (1 - edge_vars))

    # C5. Temporal ordering with latent kinase activation times.
    if latent_nodes:
        allowed_times = np.array(sorted(set(map(float, latent_time_allowed_values))))
        if len(allowed_times) == 0:
            raise ValueError("latent_time_allowed_values cannot be empty")
        min_time, max_time = float(allowed_times.min()), float(allowed_times.max())
        time_M = (max_time - min_time) + 1
        choice_vars = cp.Variable((len(latent_nodes), len(allowed_times)), boolean=True)
        constraints += [
            cp.sum(choice_vars, axis=1) == 1,
            latent_time_vars == choice_vars @ allowed_times,
            latent_time_vars >= min_time,
            latent_time_vars <= max_time,
        ]

        def time_expr(node: str):
            """
            Return the activation time of a node as either a CVXPY expression or a constant.

            Args:
              node: Node key to resolve.

            Returns:
              The node's CVXPY time variable when its time is latent, a float when its time is fixed, or
              None when the node has no activation time at all, in which case the caller skips the
              temporal constraint for any edge touching it.
            """
            if node in latent_time_indices:
                return latent_time_vars[latent_time_indices[node]]
            if node in activation_times:
                return float(activation_times[node])
            return None

        for edge_i, (u, v) in enumerate(edges):
            if ph_net[u][v].get("artificial", False):
                continue
            u_time = time_expr(u)
            v_time = time_expr(v)
            if u_time is None or v_time is None:
                continue
            if allow_equal_time:
                constraints.append(u_time <= v_time + time_M * (1 - edge_vars[edge_i]))
            else:
                constraints.append(u_time + 1 <= v_time + time_M * (1 - edge_vars[edge_i]))

    # C6. Optional exact biological edge count.
    if n_edges is not None:
        selected_original_edges = cp.sum(edge_vars[original_edge_indices]) if original_edge_indices else 0
        constraints.append(selected_original_edges == n_edges)

    missed_prize = cp.sum(cp.multiply(prize_values, 1 - node_vars[prize_indices])) if len(prize_indices) else 0
    biological_node_indices = [node_indices[n] for n in nodes if n not in {root_name, sink_name}]
    selected_node_cost = node_penalty * cp.sum(node_vars[biological_node_indices])
    latent_time_cost = latent_time_penalty * cp.sum(latent_time_vars) if latent_time_vars is not None else 0
    objective = cp.Minimize(missed_prize + cp.sum(cp.multiply(edge_costs, edge_vars)) + selected_node_cost + latent_time_cost)

    problem = cp.Problem(objective, constraints)
    solve_kwargs = {"solver": solver, "verbose": bool(verbose)}
    if solver == "GUROBI":
        solve_kwargs.update({"time_limit": time_limit, "mipGap": mip})
    elif solver == "SCIPY":
        solve_kwargs["scipy_options"] = {"time_limit": float(time_limit), "mip_rel_gap": float(mip)}
    problem.solve(**solve_kwargs)

    selected_edges = [] if edge_vars.value is None else [edge for edge, value in zip(edges, edge_vars.value) if value > 0.5]
    subgraph = nx.edge_subgraph(ph_net, selected_edges).copy()
    if not include_artificial_terminals:
        subgraph.remove_nodes_from([root_name, sink_name])

    latent_time_values = {}
    if latent_time_vars is not None and latent_time_vars.value is not None:
        latent_time_values = {node: float(value) for node, value in zip(latent_nodes, latent_time_vars.value)}
    for node in subgraph.nodes():
        if node in activation_times:
            subgraph.nodes[node]["assigned_activation_time"] = float(activation_times[node])
        elif node in latent_time_values:
            subgraph.nodes[node]["assigned_activation_time"] = latent_time_values[node]

    result = {
        "subgraph": subgraph,
        "status": problem.status,
        "objective_value": problem.value,
        "selected_edges": selected_edges,
        "candidate_original_edges": original_edges,
        "nodes": nodes,
        "edges": edges,
        "node_values": node_vars.value,
        "edge_values": edge_vars.value,
        "distance_values": distance_vars.value,
        "latent_time_values": latent_time_values,
        "problem": problem,
    }
    return result if return_problem else subgraph


def prepare_solver_inputs(candidate_graph: nx.DiGraph,
                          root_kinase: str = EGFR_UNIPROT,
                          root_name: str = ROOT,
                          sink_name: str = SINK,):
    """
    Derive the arguments dynamic_rpcst_selection needs from the candidate graph.

    Strips the explicit ROOT and SINK so the optimiser can add its own artificial terminals,
    collects the phosphosite prizes and measured activation times, pins the root kinase at time
    zero, and lists the kinases whose activation time is left to the solver.

    Args:
      candidate_graph: Candidate graph as returned by build_candidate_graph, still carrying its
        ROOT and SINK nodes.
      root_kinase: UniProt accession of the receptor kinase the network is rooted at.
      root_name: Key of the artificial root node to strip.
      sink_name: Key of the artificial sink node to strip.

    Returns:
      Tuple of (solver_graph, prizes, receptors, tf_sites, latent_time_nodes, activation_times).
      solver_graph is the candidate graph without ROOT and SINK; prizes maps each phosphosite to
      its absolute peak log2 fold change; receptors is the single-element root kinase list;
      tf_sites are the phosphosites that fed the sink; latent_time_nodes are all kinases except
      the root; activation_times holds the measured site times plus the root kinase at 0.
    """
    solver_graph = candidate_graph.copy()
    tf_sites = sorted({
        source for source, target, data in solver_graph.edges(data=True)
        if target == sink_name and data.get("edge_type") == "tf_phosphosite_to_sink"
    })
    solver_graph.remove_nodes_from([root_name, sink_name])

    prizes = {
        node: float(data["peak_abs_log2_fc_vs_starve"])
        for node, data in solver_graph.nodes(data=True)
        if data.get("node_type") == "phosphosite" and "peak_abs_log2_fc_vs_starve" in data
    }
    activation_times = {
        node: int(data["activation_time"])
        for node, data in solver_graph.nodes(data=True)
        if data.get("node_type") == "phosphosite" and "activation_time" in data
    }
    activation_times[root_kinase] = 0
    latent_time_nodes = [
        node for node, data in solver_graph.nodes(data=True)
        if data.get("node_type") == "kinase" and node != root_kinase
    ]
    return solver_graph, prizes, [root_kinase], tf_sites, latent_time_nodes, activation_times


# ── 5. Validation ────────────────────────────────────────────────────────────────────────


def selected_with_artificial_terminals(result: dict) -> nx.DiGraph:
    """
    Rebuild the selected network with its artificial root and sink edges retained.

    Used only for the connectivity check, since the returned subgraph of the optimiser has the
    artificial terminals removed and therefore no longer shows a ROOT to SINK path.

    Args:
      result: Result dict from dynamic_rpcst_selection called with return_problem=True.

    Returns:
      DiGraph containing the selected edges. Note that the nodes are taken from result['nodes'],
      which is the full candidate node list, so the graph also holds unselected isolated nodes;
      only its edges describe the selection.
    """
    full = nx.DiGraph()
    full.add_nodes_from(result["nodes"])
    full.add_edges_from(result["selected_edges"])
    return full


def temporal_violations(graph: nx.DiGraph) -> list[tuple[str, str, float, float]]:
    """
    List selected edges that run backwards in time, which the constraints should make impossible.

    Args:
      graph: Selected network whose nodes carry assigned_activation_time. Edges flagged
        artificial and edges with a missing time at either end are skipped.

    Returns:
      List of (source, target, source_time, target_time) tuples for every edge whose source
      activates later than its target, using a 1e-6 minute tolerance. An empty list means the
      temporal ordering holds throughout.
    """
    violations = []
    for source, target, data in graph.edges(data=True):
        if data.get("artificial", False):
            continue
        source_time = graph.nodes[source].get("assigned_activation_time")
        target_time = graph.nodes[target].get("assigned_activation_time")
        if source_time is None or target_time is None:
            continue
        if float(source_time) > float(target_time) + 1e-6:
            violations.append((source, target, float(source_time), float(target_time)))
    return violations


# ── 6. Outputs ───────────────────────────────────────────────────────────────────────────


def write_selected_outputs(graph: nx.DiGraph, summary: dict, prefix: Path) -> dict:
    """
    Write the selected network to disk as a pickle, two CSV tables and a JSON summary.

    Args:
      graph: Selected network to serialise.
      summary: Validation and solver summary dict, written verbatim as JSON.
      prefix: Path prefix the four output paths are derived from by appending
        .selected_graph.pkl, .selected_nodes.csv, .selected_edges.csv and .summary.json. Parent
        directories are created if missing, and existing files at these paths are overwritten.

    Returns:
      Dict mapping the keys 'pickle', 'nodes_csv', 'edges_csv' and 'summary_json' to the written
      paths as strings.
    """
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "pickle": prefix.with_suffix(".selected_graph.pkl"),
        "nodes_csv": prefix.with_suffix(".selected_nodes.csv"),
        "edges_csv": prefix.with_suffix(".selected_edges.csv"),
        "summary_json": prefix.with_suffix(".summary.json"),
    }
    with paths["pickle"].open("wb") as fh:
        pickle.dump(graph, fh)

    node_rows = []
    for node, data in graph.nodes(data=True):
        row = {"node": node}
        row.update(data)
        node_rows.append(row)
    edge_rows = []
    for source, target, data in graph.edges(data=True):
        row = {"source": source, "target": target}
        row.update(data)
        edge_rows.append(row)

    pd.DataFrame(node_rows).to_csv(paths["nodes_csv"], index=False)
    pd.DataFrame(edge_rows).to_csv(paths["edges_csv"], index=False)
    with paths["summary_json"].open("w") as fh:
        json.dump(summary, fh, indent=2)
    return {key: str(value) for key, value in paths.items()}


# ── 7. Graphviz rendering ────────────────────────────────────────────────────────────────


def quote_dot(value: object) -> str:
    """
    Render a value as a quoted DOT string literal, escaping backslashes and double quotes.

    Args:
      value: Any object; converted with str() before escaping.

    Returns:
      The string surrounded by double quotes and safe to paste into a DOT attribute.
    """
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def html_escape(text: object) -> str:
    """
    Escape a value for use inside a Graphviz HTML-like label.

    Args:
      text: Any object; converted with str() before escaping.

    Returns:
      The string with the HTML special characters replaced by entities.
    """
    return html.escape(str(text))


def html_label(lines: list[str]) -> str:
    """
    Join several lines into a single Graphviz HTML-like label.

    Args:
      lines: Sequence of strings to stack, each escaped and separated by an HTML line break.

    Returns:
      The label wrapped in angle brackets, which is how DOT distinguishes an HTML-like label
      from a plain quoted string.
    """
    return "<" + "<BR/>".join(html_escape(line) for line in lines) + ">"


def uniprot_to_gene_map(metadata_path: Path) -> dict[str, str]:
    """
    Build the UniProt accession to gene-name lookup used to label kinase nodes.

    Args:
      metadata_path: Path to the parquet file holding the protein_Id and protein_name columns.
        Rows missing either value are dropped and the first row per accession is kept.

    Returns:
      Dict mapping UniProt accession to gene name. Kinases absent from the measured dataset will
      not appear, and are then drawn with their accession only.
    """
    metadata = pd.read_parquet(metadata_path, columns=["protein_Id", "protein_name"])
    metadata = metadata.dropna(subset=["protein_Id", "protein_name"]).drop_duplicates("protein_Id")
    return dict(zip(metadata["protein_Id"].astype(str), metadata["protein_name"].astype(str)))


def dot_node(node: str,
             data: dict,
             uniprot_to_gene: dict[str, str],
             time_colors: dict | None = None,) -> str:
    """
    Render one node as a DOT statement, styled by whether it is a kinase or a phosphosite.

    Kinases are drawn as rounded blue boxes labelled with gene name, accession and the
    activation time the solver assigned. Phosphosites are drawn as filled ellipses labelled with
    the site identifier, its measured peak time and its absolute log2 fold change, coloured by
    peak time through TIME_COLORS.

    Args:
      node: Node key, a UniProt accession for kinases and a UniProt_site string for phosphosites.
      data: Node attribute dict, read for node_type, assigned_activation_time, protein_site_id,
        activation_time and peak_abs_log2_fc_vs_starve.
      uniprot_to_gene: Accession to gene-name lookup as returned by uniprot_to_gene_map.
      time_colors: Peak-time to fill-colour map. None uses the module default TIME_COLORS.

    Returns:
      A single indented DOT node statement, terminated by a semicolon.
    """
    time_colors = TIME_COLORS if time_colors is None else time_colors
    if data.get("node_type") == "kinase":
        gene = uniprot_to_gene.get(node)
        assigned_time = data.get("assigned_activation_time")
        label = [gene, node, f"assigned t={assigned_time:g}" if assigned_time is not None else "latent time"] if gene else [node, "latent time"]
        attrs = {
            "label": html_label(label),
            "shape": "box",
            "style": "rounded,filled",
            "fillcolor": "#dbeafe",
            "color": "#1e3a8a",
            "penwidth": "1.6",
        }
    else:
        time = data.get("activation_time")
        label = [
            data.get("protein_site_id", node),
            f"t={time} min",
            f"|log2FC|={data.get('peak_abs_log2_fc_vs_starve', float('nan')):.2f}",
        ]
        attrs = {
            "label": html_label(label),
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": time_colors.get(time, "#6b7280"),
            "fontcolor": "white",
            "color": "white",
        }
    attr_text = ", ".join(f"{k}={v}" if k == "label" else f"{k}={quote_dot(v)}" for k, v in attrs.items())
    return f"  {quote_dot(node)} [{attr_text}];"


def dot_edge(source: str, target: str, data: dict) -> str:
    """
    Render one edge as a DOT statement, styled by edge type.

    Phosphosite to kinase edges are drawn as orange dashed arrows; all others are grey solid
    arrows, and kinase to phosphosite edges additionally carry their interaction score as a
    label.

    Args:
      source: Key of the edge's source node.
      target: Key of the edge's target node.
      data: Edge attribute dict, read for edge_type and weight.

    Returns:
      A single indented DOT edge statement, terminated by a semicolon.
    """
    if data.get("edge_type") == "phosphosite_to_kinase":
        attrs = {"color": "#f59e0b", "style": "dashed", "penwidth": "1.5"}
    else:
        attrs = {"color": "#64748b", "style": "solid", "penwidth": "1.5"}
        if data.get("edge_type") == "kinase_to_phosphosite" and "weight" in data:
            attrs["label"] = f"{float(data['weight']):.2f}"
            attrs["fontsize"] = "8"
            attrs["fontcolor"] = "#475569"
    attr_text = ", ".join(f"{k}={quote_dot(v)}" for k, v in attrs.items())
    return f"  {quote_dot(source)} -> {quote_dot(target)} [{attr_text}];"


def write_graphviz_dot(graph: nx.DiGraph,
                       path: Path,
                       title: str,
                       metadata_path: Path,
                       time_colors: dict | None = None,) -> None:
    """
    Write the selected network to a DOT file, including a legend subgraph.

    Emits the graph-level layout attributes, one statement per node and per edge, and a legend
    cluster showing the two node shapes, the two edge styles and the peak-time colour key. The
    file is written but not rendered; rendering to SVG or PNG is done separately by the Graphviz
    dot command.

    Args:
      graph: Selected network to draw.
      path: Destination path for the DOT file. An existing file is overwritten.
      title: Caption placed at the top of the figure.
      metadata_path: Parquet file holding protein_Id and protein_name, used to label kinase
        nodes with their gene name. In the notebook this is SOURCE_PARQUET.
      time_colors: Peak-time to fill-colour map used for the phosphosite nodes and the legend.
        None uses the module default TIME_COLORS.

    Returns:
      None. The DOT file is written as a side effect.
    """
    time_colors = TIME_COLORS if time_colors is None else time_colors
    uniprot_to_gene = uniprot_to_gene_map(metadata_path)
    lines = [
        "digraph G {",
        "  graph [rankdir=TB, overlap=false, splines=true, nodesep=0.35, ranksep=0.60, bgcolor=\"white\"];",
        "  node [fontname=\"Helvetica\", fontsize=10, margin=0.08];",
        "  edge [fontname=\"Helvetica\", arrowsize=0.7];",
        '  labelloc="t";',
        f"  label={quote_dot(title)};",
        "",
    ]
    for node, data in graph.nodes(data=True):
        lines.append(dot_node(node, data, uniprot_to_gene, time_colors))
    lines.append("")
    for source, target, data in graph.edges(data=True):
        lines.append(dot_edge(source, target, data))

    lines.extend([
        "",
        "  subgraph cluster_legend {",
        '    label="Legend";',
        '    color="#cbd5e1";',
        '    style="rounded";',
        '    "legend_kinase" [label=<Kinase<BR/>gene / UniProt<BR/>assigned t>, shape="box", style="rounded,filled", fillcolor="#dbeafe", color="#1e3a8a"];',
        '    "legend_site" [label=<Phosphosite<BR/>peak time color>, shape="ellipse", style="filled", fillcolor="#6b7280", fontcolor="white", color="white"];',
        '    "legend_kinase" -> "legend_site" [label=" kinase -> site", color="#64748b", penwidth="1.5", fontsize="9"];',
        '    "legend_site" -> "legend_kinase" [label=" site -> kinase", color="#f59e0b", style="dashed", penwidth="1.5", fontsize="9"];',
    ])
    previous = None
    for time, color in time_colors.items():
        node = f"legend_time_{time}"
        lines.append(f'    {quote_dot(node)} [label={quote_dot(str(time) + " min")}, shape="box", style="filled", fillcolor={quote_dot(color)}, fontcolor="white", color="white"];')
        if previous:
            lines.append(f"    {quote_dot(previous)} -> {quote_dot(node)} [style=\"invis\"];")
        previous = node
    lines.extend(["  }", "}"])
    path.write_text("\n".join(lines) + "\n")
