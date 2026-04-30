"""
to_training_data.py
====================
Converts ScholarHypergraph output → PyTorch Geometric training-ready files.

Output files:
  H.npz            - Incidence matrix (N nodes × M hyperedges), scipy sparse CSR
  X.npy            - Node feature matrix (N × F), float32
  X_scholar.npy    - Scholar-only feature matrix (n_scholars × F), float32
  node_id_map.json - node_id string ↔ matrix row index
  meta.json        - Dataset statistics and feature names
  splits/
    train_pairs.npy  - Positive (scholar_i, scholar_j) pairs for training
    val_pairs.npy    - Validation pairs
    test_pairs.npy   - Test pairs
    neg_pairs.npy    - Hard negative pairs (same keyword/venue but no award collab)

Usage:
  from hypergraph_builder import ScholarHypergraph
  from to_training_data import to_training_data

  hg = ScholarHypergraph()
  hg.add_scholar_from_file("scholar1.json")
  hg.build()
  to_training_data(hg, save_dir="training_data/")
"""

import os
import json
import math
import random
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from typing import Optional

# Optional: sentence-transformers for keyword embedding
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


# ─────────────────────────────────────────────────────────────
#  Positive-pair source configuration
#  Modify here to add / remove sources or change sampling caps.
# ─────────────────────────────────────────────────────────────

# Each entry: hedge_type_prefix → max pairs to keep (None = keep all)
# Priority order matters for deduplication: award > pub > prog
POSITIVE_SOURCES = {
    "award_team"              : None,   # ~1,736  — strongest signal, keep all
    "publication_coauthorship": None,   # ~51,766 — true co-authorship, keep all
    "nsf_program_community"   : 5_000,  # ~127,559 — sample to avoid dominating
}

# ─────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────

def to_training_data(
    hg,
    save_dir: str = "training_data",
    keyword_embed_dim: int = 32,
    use_sbert: bool = False,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    neg_ratio: float = 3,
    random_seed: int = 42,
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "splits"), exist_ok=True)

    print("=" * 55)
    print("  Converting hypergraph → training data")
    print("=" * 55)

    # ── 1. Index all nodes and hyperedges ──────────────────
    node_ids  = list(hg.nodes.keys())
    hedge_ids = list(hg.hedges.keys())
    node2idx  = {n: i for i, n in enumerate(node_ids)}
    hedge2idx = {h: i for i, h in enumerate(hedge_ids)}

    N = len(node_ids)
    M = len(hedge_ids)
    print(f"  Nodes : {N}  |  Hyperedges : {M}")

    scholar_ids  = [n for n in node_ids if hg.nodes[n]["type"] == "scholar"]
    scholar_idxs = [node2idx[n] for n in scholar_ids]
    print(f"  Scholar nodes: {len(scholar_ids)}")

    # ── 2. Build incidence matrix H ────────────────────────
    print("\n[1/5] Building incidence matrix H ...")
    H = _build_incidence_matrix(hg, node2idx, hedge2idx, N, M)
    sp.save_npz(os.path.join(save_dir, "H.npz"), H)
    print(f"  H shape: {H.shape}, nnz: {H.nnz}")

    # ── 3. Build node feature matrix X ────────────────────
    print("\n[2/5] Building node feature matrix X ...")
    X, feature_names = _build_node_features(
        hg, node_ids, scholar_ids, keyword_embed_dim, use_sbert
    )
    np.save(os.path.join(save_dir, "X.npy"), X)
    print(f"  X shape: {X.shape}  ({len(feature_names)} features)")

    X_scholar = X[scholar_idxs]
    np.save(os.path.join(save_dir, "X_scholar.npy"), X_scholar)

    # ── 4. Build hyperedge attribute matrix ───────────────
    print("\n[3/5] Building hyperedge attribute matrix ...")
    E_attr, edge_feat_names = _build_hedge_features(hg, hedge_ids)
    np.save(os.path.join(save_dir, "E_attr.npy"), E_attr)
    print(f"  E_attr shape: {E_attr.shape}")

    # ── 5. Build pos/neg pairs ─────────────────────────────
    print("\n[4/5] Building train/val/test splits ...")
    splits = _build_splits(
        hg, node2idx, scholar_ids, node_ids,
        val_ratio, test_ratio, neg_ratio, random_seed
    )
    for name, arr in splits.items():
        path = os.path.join(save_dir, "splits", f"{name}.npy")
        np.save(path, arr)
        print(f"  {name}: {len(arr)} pairs")

    # ── 6. Save mappings and metadata ─────────────────────
    print("\n[5/5] Saving mappings and metadata ...")
    id_map = {
        "node2idx"    : node2idx,
        "idx2node"    : {str(v): k for k, v in node2idx.items()},
        "hedge2idx"   : hedge2idx,
        "idx2hedge"   : {str(v): k for k, v in hedge2idx.items()},
        "scholar_ids" : scholar_ids,
        "scholar_idxs": scholar_idxs,
    }
    with open(os.path.join(save_dir, "node_id_map.json"), "w") as f:
        json.dump(id_map, f)

    meta = {
        "num_nodes"        : N,
        "num_hyperedges"   : M,
        "num_scholars"     : len(scholar_ids),
        "feature_dim"      : X.shape[1],
        "feature_names"    : feature_names,
        "edge_feat_dim"    : E_attr.shape[1],
        "edge_feat_names"  : edge_feat_names,
        "node_type_counts" : _count_types(hg.nodes),
        "hedge_type_counts": _count_types(hg.hedges),
        "splits"           : {k: len(v) for k, v in splits.items()},
        "keyword_embed_dim": keyword_embed_dim,
        "use_sbert"        : use_sbert,
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 55)
    print(f"  Done! Saved to: {save_dir}/")
    print("=" * 55)
    _print_pyg_snippet(save_dir, X.shape[1])
    return meta


# ─────────────────────────────────────────────────────────────
#  Step 1 – Incidence matrix
# ─────────────────────────────────────────────────────────────

def _build_incidence_matrix(hg, node2idx, hedge2idx, N, M):
    rows, cols, vals = [], [], []

    for hid, hedge in hg.hedges.items():
        j = hedge2idx[hid]
        members = [n for n in hedge["nodes"] if n in node2idx]

        type_weight = {
            "award_team"              : 1.0,
            "publication_coauthorship": 0.9,
            "keyword_cluster"         : 0.5,
            "venue_cluster"           : 0.4,
            "institution_affiliation" : 0.3,
            "nsf_program_community"   : 0.6,
            "nsf_division_community"  : 0.4,
            "citation_tier"           : 0.2,
            "temporal_activity"       : 0.3,
        }.get(hedge["type"], 0.5)

        for nid in members:
            rows.append(node2idx[nid])
            cols.append(j)
            vals.append(type_weight)

    H = sp.csr_matrix((vals, (rows, cols)), shape=(N, M), dtype=np.float32)
    return H


# ─────────────────────────────────────────────────────────────
#  Step 2 – Node feature matrix
# ─────────────────────────────────────────────────────────────

def _build_node_features(hg, node_ids, scholar_ids, keyword_embed_dim, use_sbert):
    all_keywords = []
    for nid in scholar_ids:
        kws = hg.nodes[nid]["attr"].get("top_keywords", [])
        all_keywords.extend(kws)

    kw_freq = defaultdict(int)
    for kw in all_keywords:
        kw_freq[kw] += 1
    vocab = [kw for kw, _ in sorted(kw_freq.items(), key=lambda x: -x[1])]
    vocab = vocab[:keyword_embed_dim]
    kw2idx = {kw: i for i, kw in enumerate(vocab)}

    scholar_scalar_dim = 11
    total_dim = scholar_scalar_dim + keyword_embed_dim

    feature_names = (
        _scholar_scalar_names() +
        [f"kw_{kw}" for kw in vocab]
    )

    X = np.zeros((len(node_ids), total_dim), dtype=np.float32)

    for i, nid in enumerate(node_ids):
        node  = hg.nodes[nid]
        ntype = node["type"]
        attr  = node["attr"]

        if ntype == "scholar":
            scalars = _scholar_features(attr)
            kw_vec  = _keyword_tfidf(attr.get("top_keywords", []), kw2idx, keyword_embed_dim)
            X[i, :scholar_scalar_dim]          = scalars
            X[i, scholar_scalar_dim:total_dim] = kw_vec

        elif ntype == "paper":
            X[i, 0] = _safe_log(attr.get("citation_count", 0))
            X[i, 1] = (attr.get("year") or 2000) / 2025.0
            X[i, 2] = _safe_log(attr.get("author_count", 1))
            X[i, 3] = float(attr.get("has_abstract", False))

        elif ntype == "award":
            X[i, 0] = _safe_log(attr.get("amount", 0))
            start = str(attr.get("start_date", "2020-01-01"))[:4]
            end   = str(attr.get("exp_date",   "2022-01-01"))[:4]
            try:
                X[i, 1] = (int(end) - int(start)) / 5.0
            except:
                X[i, 1] = 0.0

        elif ntype == "keyword":
            term = attr.get("term", "")
            if term in kw2idx:
                X[i, scholar_scalar_dim + kw2idx[term]] = 1.0

    scholar_mask = np.array([hg.nodes[n]["type"] == "scholar" for n in node_ids])
    if scholar_mask.sum() > 1:
        scholar_rows = X[scholar_mask, :scholar_scalar_dim]
        col_max = scholar_rows.max(axis=0)
        col_max[col_max == 0] = 1.0
        X[np.ix_(scholar_mask, np.arange(scholar_scalar_dim))] = scholar_rows / col_max

    return X, feature_names


def _scholar_features(attr: dict) -> np.ndarray:
    return np.array([
        _safe_log(attr.get("h_index", 0) + 1),
        _safe_log(attr.get("citation_count", 0) + 1),
        _safe_log(attr.get("paper_count", 0) + 1),
        attr.get("experience_years", 0) / 30.0,
        float(attr.get("leadership_score", 0)),
        _safe_log(attr.get("award_count", 0) + 1),
        _safe_log(attr.get("total_funding_usd", 0) + 1),
        _safe_log(attr.get("collab_breadth", 0) + 1),
        float(attr.get("venue_diversity", 0)) / 20.0,
        attr.get("avg_citations_per_paper", 0) / 100.0,
        float(attr.get("year_last_pub") or 2000) / 2025.0,
    ], dtype=np.float32)


def _scholar_scalar_names():
    return [
        "log_h_index", "log_citations", "log_papers",
        "experience_norm", "leadership", "log_awards",
        "log_funding", "log_collab_breadth", "venue_diversity_norm",
        "avg_cit_norm", "last_pub_year_norm",
    ]


def _keyword_tfidf(keywords: list, kw2idx: dict, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for kw in keywords:
        if kw in kw2idx:
            rank = keywords.index(kw)
            vec[kw2idx[kw]] = 1.0 / (1.0 + rank)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ─────────────────────────────────────────────────────────────
#  Step 3 – Hyperedge feature matrix
# ─────────────────────────────────────────────────────────────

HEDGE_TYPES = [
    "award_team", "publication_coauthorship", "keyword_cluster",
    "venue_cluster", "institution_affiliation", "nsf_program_community",
    "nsf_division_community", "citation_tier", "temporal_activity",
    "award_overlap_period",
]

def _build_hedge_features(hg, hedge_ids):
    type2idx = {t: i for i, t in enumerate(HEDGE_TYPES)}
    feat_dim = len(HEDGE_TYPES) + 2
    feat_names = HEDGE_TYPES + ["log_size", "type_scalar"]

    E = np.zeros((len(hedge_ids), feat_dim), dtype=np.float32)

    for j, hid in enumerate(hedge_ids):
        hedge = hg.hedges[hid]
        htype = hedge["type"]
        attr  = hedge["attr"]

        if htype in type2idx:
            E[j, type2idx[htype]] = 1.0

        E[j, len(HEDGE_TYPES)] = _safe_log(len(hedge["nodes"]))

        if htype == "award_team":
            E[j, len(HEDGE_TYPES) + 1] = _safe_log(attr.get("amount", 0) + 1)
        elif htype == "publication_coauthorship":
            E[j, len(HEDGE_TYPES) + 1] = _safe_log(attr.get("citation_count", 0) + 1)
        elif htype in ("keyword_cluster", "venue_cluster",
                       "nsf_program_community", "institution_affiliation"):
            E[j, len(HEDGE_TYPES) + 1] = _safe_log(attr.get("scholar_count", 0) + 1)

    return E, feat_names


# ─────────────────────────────────────────────────────────────
#  Step 4 – Train / val / test splits
# ─────────────────────────────────────────────────────────────

def _collect_pairs_by_source(hg, node2idx, scholar_set, seed):
    """
    Collect positive scholar-scholar pairs from each source in POSITIVE_SOURCES.

    Returns:
        source_pairs : dict[hedge_type → set of (int, int) sorted pairs]
        processed in POSITIVE_SOURCES order so priority-based dedup works correctly.
    """
    rng = random.Random(seed)

    # Map hedge type string → POSITIVE_SOURCES key
    # hedge["type"] uses full names like "publication_coauthorship"
    # POSITIVE_SOURCES uses the same full names
    source_pairs: dict[str, set] = {src: set() for src in POSITIVE_SOURCES}

    for hedge in hg.hedges.values():
        htype = hedge["type"]
        if htype not in POSITIVE_SOURCES:
            continue

        scholars_in = [n for n in hedge["nodes"] if n in scholar_set]
        if len(scholars_in) < 2:
            continue

        for i in range(len(scholars_in)):
            for j in range(i + 1, len(scholars_in)):
                a, b = sorted([node2idx[scholars_in[i]],
                                node2idx[scholars_in[j]]])
                source_pairs[htype].add((a, b))

    # Apply per-source sampling caps
    print("\n  Positive pair sources (before dedup):")
    for src in POSITIVE_SOURCES:
        pairs  = source_pairs[src]
        cap    = POSITIVE_SOURCES[src]
        before = len(pairs)
        if cap is not None and len(pairs) > cap:
            pairs = set(rng.sample(list(pairs), cap))
            source_pairs[src] = pairs
        print(f"    {src:35s}: {before:>7,} raw  →  {len(pairs):>7,} kept"
              + (f"  (capped at {cap:,})" if cap and before > cap else ""))

    return source_pairs


def _build_splits(hg, node2idx, scholar_ids, node_ids,
                  val_ratio, test_ratio, neg_ratio, seed):
    random.seed(seed)
    rng = random.Random(seed)

    scholar_set = set(scholar_ids)

    # ── Collect positive pairs per source ─────────────────
    source_pairs = _collect_pairs_by_source(hg, node2idx, scholar_set, seed)

    # ── Merge with priority dedup ──────────────────────────
    # award_team pairs take priority; pub pairs added only if not already
    # in award set; prog pairs added only if not in either of the above.
    # This prevents the same pair being counted twice across sources.
    all_pos: set = set()
    for src in POSITIVE_SOURCES:           # iteration order = priority order
        before = len(all_pos)
        all_pos.update(source_pairs[src])
        added = len(all_pos) - before
        print(f"    {src:35s}: +{added:,} new pairs after dedup")

    all_pos = list(all_pos)
    rng.shuffle(all_pos)

    # ── Coverage stats ────────────────────────────────────
    scholars_covered = set()
    for a, b in all_pos:
        scholars_covered.add(a)
        scholars_covered.add(b)
    print(f"\n  Total positive pairs  : {len(all_pos):,}")
    print(f"  Scholars with ≥1 pair : {len(scholars_covered):,} / {len(scholar_ids):,}"
          f"  ({100*len(scholars_covered)/max(len(scholar_ids),1):.1f}%)")

    # ── Train / val / test split ──────────────────────────
    n_total = len(all_pos)
    n_test  = max(1, int(n_total * test_ratio))
    n_val   = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    train_pos = all_pos[:n_train]
    val_pos   = all_pos[n_train:n_train + n_val]
    test_pos  = all_pos[n_train + n_val:]

    print(f"\n  Split → train: {len(train_pos):,}"
          f"  val: {len(val_pos):,}"
          f"  test: {len(test_pos):,}")

    # ── Hard negatives ────────────────────────────────────
    # Use keyword/venue/tier hedges (similar but not true collaborators)
    all_pos_set = set(map(tuple, all_pos))
    soft_connected: set = set()

    HARD_NEG_SOURCES = {"keyword_cluster", "venue_cluster", "citation_tier"}
    for hedge in hg.hedges.values():
        if hedge["type"] not in HARD_NEG_SOURCES:
            continue
        scholars_in = [n for n in hedge["nodes"] if n in scholar_set]
        for i in range(len(scholars_in)):
            for j in range(i + 1, len(scholars_in)):
                a, b = sorted([node2idx[scholars_in[i]],
                                node2idx[scholars_in[j]]])
                if (a, b) not in all_pos_set:
                    soft_connected.add((a, b))

    # Supplement with random negatives
    scholar_idxs_list = [node2idx[s] for s in scholar_ids]
    random_negs: set = set()
    target_neg = int(neg_ratio * n_total) * 2   # 2× buffer, trim later
    attempts   = 0
    while len(random_negs) < target_neg and attempts < 200_000:
        a = rng.choice(scholar_idxs_list)
        b = rng.choice(scholar_idxs_list)
        if a != b:
            pair = tuple(sorted([a, b]))
            if pair not in all_pos_set:
                random_negs.add(pair)
        attempts += 1

    neg_pool = list(soft_connected) + list(random_negs)
    rng.shuffle(neg_pool)
    n_neg    = int(neg_ratio * n_total)
    neg_pairs = neg_pool[:n_neg]
    print(f"  Neg pairs             : {len(neg_pairs):,}"
          f"  (hard: {min(len(soft_connected), n_neg):,}"
          f"  + random: {max(0, n_neg - len(soft_connected)):,})")

    return {
        "train_pairs": np.array(train_pos, dtype=np.int64),
        "val_pairs"  : np.array(val_pos,   dtype=np.int64),
        "test_pairs" : np.array(test_pos,  dtype=np.int64),
        "neg_pairs"  : np.array(neg_pairs, dtype=np.int64),
    }


# ─────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────

def _safe_log(x):
    return float(np.log1p(max(0, x)))

def _count_types(collection):
    counts = defaultdict(int)
    for item in collection.values():
        counts[item["type"]] += 1
    return dict(counts)

def _print_pyg_snippet(save_dir, feat_dim):
    print(f"""
─────────────────────────────────────────────
  PyTorch Geometric quickstart:
─────────────────────────────────────────────
  import torch, numpy as np, scipy.sparse as sp
  from torch_geometric.data import Data
  from torch_geometric.nn import HypergraphConv

  H = sp.load_npz("{save_dir}/H.npz").tocoo()
  X = np.load("{save_dir}/X.npy")

  hyperedge_index = torch.tensor([H.row, H.col], dtype=torch.long)
  x = torch.tensor(X, dtype=torch.float)

  data = Data(x=x, hyperedge_index=hyperedge_index)

  conv = HypergraphConv(in_channels={feat_dim}, out_channels=64)
  embeddings = conv(data.x, data.hyperedge_index)
─────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────
#  Load from saved hypergraph JSON
# ─────────────────────────────────────────────────────────────

def load_saved_hypergraph(saved_path: str):
    print(f"Loading saved hypergraph from {saved_path} ...")
    with open(saved_path, "r") as f:
        data = json.load(f)

    if "nodes" in data and "hedges" in data:
        nodes  = data["nodes"]
        hedges = data["hedges"]
    else:
        raise ValueError(
            f"Unexpected format in {saved_path}. "
            "Expected keys 'nodes' and 'hedges' from hg.save()."
        )

    print(f"  Nodes : {len(nodes)}")
    print(f"  Hedges: {len(hedges)}")

    class _HG:
        pass

    hg = _HG()
    hg.nodes  = nodes
    hg.hedges = hedges
    return hg


# ─────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    input_path = "graph/hypergraph_output.json"
    save_dir   = "training_data"

    if input_path.endswith(".json") and os.path.isfile(input_path):
        hg = load_saved_hypergraph(input_path)

    elif os.path.isdir(input_path):
        from hypergraph_builder import ScholarHypergraph
        hg = ScholarHypergraph()
        files = [f for f in os.listdir(input_path) if f.endswith(".json")]
        print(f"Loading {len(files)} scholar files from {input_path} ...")
        for i, f in enumerate(files):
            hg.add_scholar_from_file(os.path.join(input_path, f))
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(files)} loaded...")
        hg.build()

    else:
        raise FileNotFoundError(f"Not found: {input_path}")

    meta = to_training_data(hg, save_dir=save_dir)
    print(json.dumps(meta, indent=2))