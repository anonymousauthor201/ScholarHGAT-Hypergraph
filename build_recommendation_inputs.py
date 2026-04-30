"""
Generates all inputs needed by team_recommendation.py
from a trained scholarhgat model.

Produces:
  checkpoints/scores.npy               — (n_scholars,) global ranking scores
  checkpoints/node_embs_keyword.npz
  checkpoints/node_embs_venue.npz
  checkpoints/node_embs_nsf_program.npz
  checkpoints/node_embs_nsf_division.npz
  checkpoints/node_embs_institution.npz
  checkpoints/node_embeddings_index.json

Score computation:
  s_c = (1/|S_ref|) sum_{s in S_ref} score_all(emb, s)[c]
  Mean TransE score over n_ref sampled reference scholars.

"""

import os
import json
import argparse
import importlib.util
import numpy as np
import scipy.sparse as sp

import torch

NODE_TYPE_LIST = ["scholar", "paper", "award", "institution",
                  "keyword", "venue", "nsf_program", "nsf_division"]
NODE_TYPE_TO_ID   = {t: i for i, t in enumerate(NODE_TYPE_LIST)}
NODE_TYPE_UNKNOWN = len(NODE_TYPE_LIST)

HEDGE_TYPE_LIST = ["award_team", "publication_coauthorship", "keyword_cluster",
                   "venue_cluster", "institution_affiliation", "nsf_program_community",
                   "nsf_division_community", "citation_tier", "temporal_activity",
                   "award_overlap_period"]
HEDGE_TYPE_TO_ID   = {t: i for i, t in enumerate(HEDGE_TYPE_LIST)}
HEDGE_TYPE_UNKNOWN = len(HEDGE_TYPE_LIST)


# ─────────────────────────────────────────────────────────────
#  0. id_map normalisation helper
#     Supports BOTH formats:
#       Old: {"idx2node": {"0": "scholar::x", ...}, "idx2hedge": {...}, "scholar_idxs": [...]}
#       New: {"node2idx": {"scholar::x": 0,  ...}, "hedge2idx": {...}}   ← your current format
# ─────────────────────────────────────────────────────────────

def normalise_id_map(raw: dict) -> dict:
    """
    Returns a canonical id_map with keys:
      idx2node      : dict[str(int) -> str]   e.g. {"0": "scholar::nsf_000003742", ...}
      idx2hedge     : dict[str(int) -> str]   e.g. {"0": "h_award::...", ...}
      scholar_idxs  : list[int]
      node2idx      : dict[str -> int]        (always present, original or inverted)
      hedge2idx     : dict[str -> int]        (always present, original or inverted)
    """
    out = dict(raw)  # shallow copy

    # ── node direction ──────────────────────────────────────────────────────
    if "idx2node" not in out:
        if "node2idx" not in out:
            raise KeyError("id_map has neither 'idx2node' nor 'node2idx'.")
        # invert  node2idx  →  idx2node
        out["idx2node"] = {str(v): k for k, v in out["node2idx"].items()}
        print("  [id_map] Built idx2node by inverting node2idx "
              f"({len(out['idx2node'])} entries)")
    else:
        # ensure node2idx is also available for extract_node_texts
        if "node2idx" not in out:
            out["node2idx"] = {v: int(k) for k, v in out["idx2node"].items()}

    # ── hedge direction ─────────────────────────────────────────────────────
    if "idx2hedge" not in out:
        if "hedge2idx" in out:
            out["idx2hedge"] = {str(v): k for k, v in out["hedge2idx"].items()}
            print("  [id_map] Built idx2hedge by inverting hedge2idx "
                  f"({len(out['idx2hedge'])} entries)")
        else:
            # Some node_id_map.json files omit hedge info entirely;
            # we'll build a placeholder — hedge types will fall back to UNKNOWN.
            print("  [id_map] WARNING: no hedge id info found; "
                  "hedge types will all be UNKNOWN.")
            out["idx2hedge"] = {}

    if "hedge2idx" not in out:
        out["hedge2idx"] = {v: int(k) for k, v in out["idx2hedge"].items()}

    # ── scholar_idxs ────────────────────────────────────────────────────────
    if "scholar_idxs" not in out:
        # derive from node2idx: all entries whose key starts with "scholar::"
        scholar_idxs = sorted(
            idx for node_id, idx in out["node2idx"].items()
            if node_id.startswith("scholar::")
        )
        out["scholar_idxs"] = scholar_idxs
        print(f"  [id_map] Derived scholar_idxs from node2idx "
              f"({len(scholar_idxs)} scholars)")

    return out


# ─────────────────────────────────────────────────────────────
#  1. Data Loading
# ─────────────────────────────────────────────────────────────

def load_data_v2(data_dir, device, add_self_loops=False):
    """
    Returns data dict with v2 keys:
      node_type_ids   : LongTensor (N,)
      hedge_type_ids  : LongTensor (M,)
      scholar_idxs_t  : LongTensor
      global_to_local : LongTensor (N,)
    """
    H = sp.load_npz(os.path.join(data_dir, "H.npz")).tocoo()
    hyperedge_index = torch.tensor(
        np.vstack([H.row, H.col]), dtype=torch.long, device=device
    )
    X = torch.tensor(
        np.load(os.path.join(data_dir, "X.npy")),
        dtype=torch.float, device=device
    )
    with open(os.path.join(data_dir, "node_id_map.json")) as f:
        raw_id_map = json.load(f)
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)

    # ── normalise to canonical format ──────────────────────────────────────
    id_map = normalise_id_map(raw_id_map)

    # ── node_type_ids ──────────────────────────────────────────────────────
    idx2node   = id_map["idx2node"]
    node_types = [idx2node[str(i)].split("::")[0] for i in range(len(idx2node))]
    node_type_ids = torch.tensor(
        [NODE_TYPE_TO_ID.get(t, NODE_TYPE_UNKNOWN) for t in node_types],
        dtype=torch.long, device=device
    )

    # ── hedge_type_ids and years ───────────────────────────────────────────
    idx2hedge = id_map["idx2hedge"]
    type_map  = {
        "award": "award_team", "pub": "publication_coauthorship",
        "kw": "keyword_cluster", "venue": "venue_cluster",
        "inst": "institution_affiliation", "prog": "nsf_program_community",
        "div": "nsf_division_community", "tier": "citation_tier",
        "temporal": "temporal_activity", "award_overlap": "award_overlap_period",
    }
    hedge_types_str, hedge_years = [], []
    num_hedges_raw = meta["num_hyperedges"]
    for j in range(num_hedges_raw):
        hid    = idx2hedge.get(str(j), "")
        parts  = hid.split("::")
        prefix = parts[0].replace("h_", "") if hid else ""
        hedge_types_str.append(type_map.get(prefix, "unknown"))
        year = 0.0
        if prefix in ("temporal", "award_overlap") and len(parts) > 1:
            try:
                year = float(parts[1].split("_")[0])
            except Exception:
                pass
        hedge_years.append(year)

    hedge_type_ids = torch.tensor(
        [HEDGE_TYPE_TO_ID.get(t, HEDGE_TYPE_UNKNOWN) for t in hedge_types_str],
        dtype=torch.long, device=device
    )
    hedge_years_t = torch.tensor(hedge_years, dtype=torch.float, device=device)

    num_nodes  = meta["num_nodes"]
    num_hedges = meta["num_hyperedges"]

    if add_self_loops:
        self_node  = torch.arange(num_nodes, device=device, dtype=torch.long)
        self_hedge = torch.arange(num_hedges, num_hedges + num_nodes,
                                  device=device, dtype=torch.long)
        hyperedge_index = torch.cat(
            [hyperedge_index, torch.stack([self_node, self_hedge])], dim=1
        )
        hedge_type_ids = torch.cat([
            hedge_type_ids,
            torch.full((num_nodes,), HEDGE_TYPE_UNKNOWN, dtype=torch.long, device=device)
        ])
        hedge_years_t = torch.cat([
            hedge_years_t,
            torch.zeros(num_nodes, dtype=torch.float, device=device)
        ])
        num_hedges += num_nodes

    scholar_idxs   = id_map["scholar_idxs"]
    scholar_idxs_t = torch.tensor(scholar_idxs, dtype=torch.long, device=device)

    global_to_local = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    for local_idx, global_idx in enumerate(scholar_idxs):
        global_to_local[global_idx] = local_idx

    return {
        "X"              : X,
        "hyperedge_index": hyperedge_index,
        "node_type_ids"  : node_type_ids,
        "hedge_type_ids" : hedge_type_ids,
        "hedge_years"    : hedge_years_t,
        "scholar_idxs"   : scholar_idxs,
        "scholar_idxs_t" : scholar_idxs_t,
        "global_to_local": global_to_local,
        "num_nodes"      : num_nodes,
        "num_hedges"     : num_hedges,
        "input_dim"      : meta["feature_dim"],
        "id_map"         : id_map,   # normalised map
    }


# ─────────────────────────────────────────────────────────────
#  2. Model Loading
# ─────────────────────────────────────────────────────────────

def load_model_v2(hgat_path, model_path, input_dim, hidden_dim,
                  output_dim, num_layers, num_heads, scoring_mode, device):
    """
    Loads ScholarHGAT from scholar_hgat_v2.py.

    Important: scoring_mode must match what was used during training.
      - 'transe'   → checkpoint has collab_relation
      - 'bilinear' → checkpoint has W_bilinear
      - 'dot'      → neither
    """
    if not os.path.exists(hgat_path):
        raise FileNotFoundError(f"Not found: {hgat_path}")

    spec   = importlib.util.spec_from_file_location("scholar_hgat_v2", hgat_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.ScholarHGAT(
        input_dim    = input_dim,
        hidden_dim   = hidden_dim,
        output_dim   = output_dim,
        num_layers   = num_layers,
        num_heads    = num_heads,
        scoring_mode = scoring_mode,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if not missing and not unexpected:
        print("  Checkpoint loaded cleanly")
    else:
        if missing:
            print(f"  Missing  ({len(missing)}): {missing[:2]}"
                  f"{'...' if len(missing)>2 else ''}")
        if unexpected:
            print(f"  Unexpected ({len(unexpected)}): {unexpected[:2]}"
                  f"{'...' if len(unexpected)>2 else ''}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────
#  3. Global Score Computation
# ─────────────────────────────────────────────────────────────

def compute_global_scores(model, embeddings, activity_scores,
                           n_ref=200, device=None, verbose=True):
    """
    s_c = mean_{s in S_ref} score_all(emb, s)[c]

    Uses model.score_all() which internally dispatches to
    dot / transe / bilinear based on model.scoring_mode.
    """
    model.eval()
    n     = embeddings.size(0)
    n_ref = min(n_ref, n)
    refs  = torch.randperm(n)[:n_ref].tolist()
    acc   = torch.zeros(n, device=device)

    if verbose:
        print(f"  scoring_mode = {model.scoring_mode}, n_ref = {n_ref}")

    with torch.no_grad():
        for i, s in enumerate(refs):
            acc += model.score_all(embeddings, s, activity_scores)
            if verbose and (i + 1) % 50 == 0:
                print(f"    {i+1}/{n_ref}")

    scores = (acc / n_ref).cpu().numpy()
    if verbose:
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    return scores


# ─────────────────────────────────────────────────────────────
#  4. Node Text Extraction + SBERT Encoding
# ─────────────────────────────────────────────────────────────

def extract_node_texts(id_map):
    """
    Works with the normalised id_map (always has node2idx).
    Iterates node2idx {node_id -> global_idx} directly — no inversion needed.
    """
    targets = {"keyword", "venue", "nsf_program", "nsf_division", "institution"}
    result  = {t: {"global_idxs": [], "texts": []} for t in targets}

    # id_map is already normalised: node2idx is guaranteed to exist
    for node_id, global_idx in id_map["node2idx"].items():
        parts = node_id.split("::")
        if len(parts) < 2 or parts[0] not in targets:
            continue
        label = parts[1].replace("_", " ").strip()
        if not label:
            continue
        result[parts[0]]["global_idxs"].append(int(global_idx))
        result[parts[0]]["texts"].append(label)

    for t in result:
        result[t]["global_idxs"] = np.array(result[t]["global_idxs"], dtype=np.int64)

    print("  Node text labels:")
    for t, d in result.items():
        print(f"    {t:<15}: {len(d['texts'])} nodes")
    return result


def encode_node_texts(node_texts, batch_size=256,
                      model_name="all-MiniLM-L6-v2", verbose=True):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")

    sbert  = SentenceTransformer(model_name)
    result = {}
    for ntype, data in node_texts.items():
        texts = data["texts"]
        if not texts:
            result[ntype] = {**data, "embs": np.zeros((0, 384), dtype=np.float32)}
            continue
        if verbose:
            print(f"    {ntype:<15} ({len(texts)}) ...", end=" ", flush=True)
        embs = sbert.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        result[ntype] = {"global_idxs": data["global_idxs"],
                         "texts": texts, "embs": embs}
        if verbose:
            print(f"done {embs.shape}")
    return result


# ─────────────────────────────────────────────────────────────
#  5. Save / Load helpers
# ─────────────────────────────────────────────────────────────

def save_node_embeddings(node_embeddings, save_dir):
    index = {}
    for ntype, data in node_embeddings.items():
        if len(data["global_idxs"]) == 0:
            continue
        fname = os.path.join(save_dir, f"node_embs_{ntype}.npz")
        np.savez(fname, global_idxs=data["global_idxs"],
                 embs=data["embs"], texts=np.array(data["texts"]))
        index[ntype] = {"path": fname, "n_nodes": len(data["global_idxs"]),
                        "emb_dim": data["embs"].shape[1]}
        print(f"    Saved {ntype:<15} → {os.path.basename(fname)}")
    return index


def load_node_embeddings(save_dir):
    """Load pre-built node embeddings. Returns dict for node_embeddings_by_type."""
    index_path = os.path.join(save_dir, "node_embeddings_index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"node_embeddings_index.json not found in {save_dir}.\n"
            f"Run: python build_recommendation_inputs.py --save_dir {save_dir}"
        )
    with open(index_path) as f:
        index = json.load(f)
    result = {}
    for ntype, meta in index.items():
        d = np.load(meta["path"], allow_pickle=True)
        result[ntype] = {
            "global_idxs": d["global_idxs"],
            "embs"        : d["embs"].astype(np.float32),
            "texts"       : d["texts"].tolist(),
        }
        print(f"  Loaded {ntype:<15}: {meta['n_nodes']} nodes")
    return result


# ─────────────────────────────────────────────────────────────
#  6. Main
# ─────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    print("\n[1/4] Loading data (v2 format)...")
    data = load_data_v2(args.data_dir, device, add_self_loops=args.self_loops)
    print(f"  Nodes={data['num_nodes']}, Hedges={data['num_hedges']}, "
          f"Scholars={len(data['scholar_idxs'])}")

    scores_path = os.path.join(args.save_dir, "scores.npy")
    if os.path.exists(scores_path) and not args.force:
        print(f"\n[2/4] scores.npy exists → skip  (--force to recompute)")
        global_scores = np.load(scores_path)
    else:
        print(f"\n[2/4] Computing global scores...")
        print(f"  Loading model from {args.model_path}")
        model = load_model_v2(
            args.hgat_path, args.model_path,
            data["input_dim"], args.hidden_dim, args.output_dim,
            args.num_layers, args.num_heads, args.scoring, device
        )

        emb_path = args.emb_path
        if os.path.exists(emb_path) and not args.force:
            print(f"  Embeddings exist at {emb_path}, loading...")
            embeddings = torch.tensor(
                np.load(emb_path), dtype=torch.float, device=device
            )
        else:
            print("  Running forward pass...")
            with torch.no_grad():
                out = model(
                    data["X"],
                    data["hyperedge_index"],
                    data["node_type_ids"],
                    data["hedge_type_ids"],
                    data["hedge_years"],
                )
                embeddings = model.get_scholar_embeddings(
                    out, data["scholar_idxs_t"]
                )
            np.save(emb_path, embeddings.cpu().numpy())
            print(f"  Embeddings saved → {emb_path}  shape={embeddings.shape}")

        with torch.no_grad():
            X_scholar       = data["X"][data["scholar_idxs_t"]]
            activity_scores = model.compute_activity_scores(X_scholar)

        global_scores = compute_global_scores(
            model, embeddings, activity_scores,
            n_ref=args.n_ref, device=device, verbose=True
        )
        np.save(scores_path, global_scores)
        print(f"  Scores saved → {scores_path}")

    index_path = os.path.join(args.save_dir, "node_embeddings_index.json")

    # ── Detect scholar-only graph ─────────────────────────────────────────
    NON_SCHOLAR_TYPES = {"keyword", "venue", "nsf_program", "nsf_division", "institution"}
    non_scholar_count = sum(
        1 for node_id in data["id_map"]["node2idx"]
        if node_id.split("::")[0] in NON_SCHOLAR_TYPES
    )

    if non_scholar_count == 0:
        print("\n[3/4] Scholar-only graph detected — no keyword/venue/institution nodes.")
        print("      Skipping SBERT node-text encoding (nothing to encode).")
        print("[4/4] Skipped.")
        # Write an empty index so downstream code doesn't crash on missing file
        if not os.path.exists(index_path):
            with open(index_path, "w") as f:
                json.dump({}, f)
            print(f"      Empty index written → {index_path}")
    elif os.path.exists(index_path) and not args.force:
        print(f"\n[3/4] Node embeddings exist → skip  (--force to recompute)")
        print(f"[4/4] Skipped.")
    else:
        print("\n[3/4] Extracting node text labels...")
        node_texts = extract_node_texts(data["id_map"])
        print(f"\n[4/4] Encoding with SBERT ({args.sbert_model})...")
        node_embeddings = encode_node_texts(
            node_texts, batch_size=args.sbert_batch,
            model_name=args.sbert_model, verbose=True
        )
        print("\n  Saving...")
        index = save_node_embeddings(node_embeddings, args.save_dir)
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"  Index → {index_path}")

    print("\n" + "─" * 50)
    print("Done. Run team_recommendation.py with:")
    print(f"  --emb_path   {args.emb_path}")
    print(f"  --scores_path {scores_path}")
    print(f"  # load node_emb_dict via load_node_embeddings('{args.save_dir}')")
    print("─" * 50)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="../training_data_v2/")
    p.add_argument("--model_path",  default="../checkpoints_newdata_loss/best_model.pt")
    p.add_argument("--hgat_path",   default="../scholar_hgatv2.py")
    p.add_argument("--emb_path",    default="../checkpoints_newdata_loss/scholar_embeddings.npy")
    p.add_argument("--save_dir",    default="checkpoints_team/")
    p.add_argument("--n_ref",       type=int,   default=200)
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--output_dim",  type=int,   default=128)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--num_heads",   type=int,   default=4)
    p.add_argument("--scoring",     default="transe",
                   choices=["dot", "transe", "bilinear"],
                   help="Must match training --scoring flag")
    p.add_argument("--self_loops",  action="store_true", default=False,
                   help="Match training --self_loops flag")
    p.add_argument("--sbert_model", default="all-MiniLM-L6-v2")
    p.add_argument("--sbert_batch", type=int, default=256)
    p.add_argument("--force",       action="store_true")
    args = p.parse_args()
    main(args)
