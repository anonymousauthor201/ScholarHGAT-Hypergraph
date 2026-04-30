"""
Metrics:
  - Precision@k, Recall@k, F1@k
  - Jaccard similarity
  - Hit Rate @ {1, 2}
  - Quality   : mean pairwise cosine similarity within predicted team
  - Diversity : 1 − mean pairwise cosine similarity
  - Coverage  : fraction of GT members whose nearest neighbor is in pred team

Baselines:
  - Random    : k scholars sampled uniformly at random
  - TopK      : top-k by raw activity/embedding score (greedy, no diversity)
  - Ours      : QD-DASTF full pipeline

"""

import os, json, argparse, glob
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import torch
from team_eval import recommend_team, load_graph_data

# ─────────────────────────────────────────────────────────────
# 1. Ground truth from hypergraph
# ─────────────────────────────────────────────────────────────

def build_ground_truth_from_hypergraph(
    hyperedge_index: torch.Tensor,
    hedge_types: List[str],
    scholar_idxs: List[int],
) -> Dict[str, Set[int]]:
    """
    Extract award_team hyperedges as ground truth.
    Returns {str(hedge_idx): set of local scholar indices}.
    """
    global_to_local = {gid: li for li, gid in enumerate(scholar_idxs)}
    node_idx_cpu    = hyperedge_index[0].cpu().tolist()
    hedge_idx_cpu   = hyperedge_index[1].cpu().tolist()

    hedge_members = defaultdict(set)
    for nid, hid in zip(node_idx_cpu, hedge_idx_cpu):
        if hid < len(hedge_types) and hedge_types[hid] == "award_team":
            local = global_to_local.get(nid)
            if local is not None:
                hedge_members[hid].add(local)

    evaluable = {str(hid): members
                 for hid, members in hedge_members.items()
                 if len(members) >= 2}
    print(f"  Award hyperedges with ≥2 scholars: {len(evaluable)}")
    return evaluable


def build_award_titles(profiles_dir: str) -> Tuple[
    Dict[str, Set[str]], Dict[str, str]
]:
    """Read award titles from scholar profiles."""
    award_teams  = defaultdict(set)
    award_titles = {}

    for fp in glob.glob(os.path.join(profiles_dir, "*.json")):
        with open(fp) as f:
            profile = json.load(f)
        nsf   = profile.get("nsf_profile", {})
        pi_id = nsf.get("nsf_pi_id")
        if not pi_id:
            continue
        for award in nsf.get("awards", []):
            award_id = award.get("nsf_award_id")
            if not award_id:
                continue
            award_teams[award_id].add(pi_id)
            for co in award.get("co_pi_ids", []):
                award_teams[award_id].add(co)
            if award_id not in award_titles:
                title = award.get("title", "").strip()
                kws   = award.get("keywords", [])
                query = title + (". Keywords: " + ", ".join(kws[:8]) if kws else "")
                award_titles[award_id] = query

    return dict(award_teams), award_titles


def build_hedge_to_award(data_dir: str) -> Dict[str, str]:
    """Map str(hedge_idx) → award_id string."""
    with open(os.path.join(data_dir, "node_id_map.json")) as f:
        id_map = json.load(f)
    idx2hedge = id_map.get("idx2hedge", {})
    mapping = {}
    for str_hid, hid_str in idx2hedge.items():
        if "award" in str(hid_str):
            parts = str(hid_str).split("::")
            if len(parts) >= 2:
                mapping[str_hid] = parts[-1]
    return mapping


# ─────────────────────────────────────────────────────────────
# 2. Metrics
# ─────────────────────────────────────────────────────────────

def precision_at_k(pred: List[int], gt: Set[int]) -> float:
    return sum(1 for p in pred if p in gt) / len(pred) if pred else 0.0

def recall_at_k(pred: List[int], gt: Set[int]) -> float:
    if not gt: return 0.0
    return sum(1 for p in pred if p in gt) / len(gt)

def f1_at_k(pred: List[int], gt: Set[int]) -> float:
    p = precision_at_k(pred, gt)
    r = recall_at_k(pred, gt)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def jaccard(pred: Set[int], gt: Set[int]) -> float:
    if not pred and not gt: return 1.0
    return len(pred & gt) / len(pred | gt)

def hit_rate(pred: Set[int], gt: Set[int], threshold: int = 1) -> int:
    return int(len(pred & gt) >= threshold)

def quality(pred: List[int], embeddings: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity within predicted team.
    Higher = more coherent team (members work in similar areas).
    Embeddings are L2-normalised before computing similarity.
    """
    if len(pred) < 2: return 0.0
    vecs = embeddings[pred].copy()
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    vecs_n = vecs / norms                                # unit vectors
    sim_matrix = vecs_n @ vecs_n.T                      # (k, k) in [-1,1]
    k = len(pred)
    pairs = [(i, j) for i in range(k) for j in range(i+1, k)]
    raw = float(np.mean([sim_matrix[i, j] for i, j in pairs]))
    # rescale from [-1,1] to [0,1] for readability
    return float(np.clip((raw + 1) / 2, 0.0, 1.0))

def diversity(pred: List[int], embeddings: np.ndarray) -> float:
    """
    1 - quality (higher = more diverse team).
    Always in [0, 1] because quality is rescaled to [0, 1].
    """
    return float(np.clip(1.0 - quality(pred, embeddings), 0.0, 1.0))

def coverage(pred: List[int], gt: Set[int], embeddings: np.ndarray,
             threshold: float = 0.5) -> float:
    """
    Skill coverage: fraction of GT members whose nearest neighbor
    in the predicted team has cosine similarity >= threshold.
    Uses GT-anchored perspective (how well does pred team cover GT skills).
    Higher threshold = stricter coverage requirement.
    """
    if not gt: return 0.0
    if embeddings is None or len(pred) == 0:
        return len(set(pred) & gt) / len(gt)

    gt_vecs   = embeddings[list(gt)].copy()
    pred_vecs = embeddings[list(pred)].copy()

    gt_n   = gt_vecs   / np.linalg.norm(gt_vecs,   axis=1, keepdims=True).clip(min=1e-8)
    pred_n = pred_vecs / np.linalg.norm(pred_vecs, axis=1, keepdims=True).clip(min=1e-8)

    sim      = gt_n @ pred_n.T                           # (|GT|, k)
    max_sim  = sim.max(axis=1)                           # best match per GT member
    covered  = (max_sim >= threshold).sum()
    return float(covered) / len(gt)


# ─────────────────────────────────────────────────────────────
# 3. Baselines
# ─────────────────────────────────────────────────────────────

def baseline_random(k: int, n_scholars: int,
                    exclude: Set[int] = None, seed: int = 0) -> List[int]:
    rng = np.random.default_rng(seed)
    pool = [i for i in range(n_scholars)
            if exclude is None or i not in exclude]
    return rng.choice(pool, size=min(k, len(pool)), replace=False).tolist()

def baseline_topk(k: int, scores: np.ndarray,
                  exclude: Set[int] = None) -> List[int]:
    s = scores.copy()
    if exclude:
        for i in exclude:
            s[i] = -1e9
    return np.argsort(-s)[:k].tolist()


# ─────────────────────────────────────────────────────────────
# 4. Compute metrics for one prediction
# ─────────────────────────────────────────────────────────────

def compute_metrics(pred: List[int], gt: Set[int],
                    embeddings: np.ndarray) -> Dict:
    pred_set = set(pred)
    return {
        "precision":  round(precision_at_k(pred, gt), 4),
        "recall":     round(recall_at_k(pred, gt), 4),
        "f1":         round(f1_at_k(pred, gt), 4),
        "jaccard":    round(jaccard(pred_set, gt), 4),
        "hit@1":      hit_rate(pred_set, gt, 1),
        "hit@2":      hit_rate(pred_set, gt, 2),
        "quality":    round(quality(pred, embeddings), 4),
        "diversity":  round(diversity(pred, embeddings), 4),
        "coverage":   round(coverage(pred, gt, embeddings), 4),
    }

def aggregate(records: List[Dict]) -> Dict:
    if not records: return {}
    keys = [k for k in records[0] if k not in ("hedge_id","award_id","query")]
    return {k: round(float(np.mean([r[k] for r in records])), 4) for k in keys}


# ─────────────────────────────────────────────────────────────
# 5. Main evaluation
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    profiles_path: str,
    emb_path:      str,
    scores_path:   str,
    data_dir:      str,
    save_dir:      str,
    k:             int  = 5,
    n_eval:        int  = 200,
    use_llm:       bool = False,
    output_path:   str  = "eval_results.json",
    seed:          int  = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load embeddings ────────────────────────────────────────
    embeddings = np.load(emb_path)                           # (S, D)
    scores = (np.load(scores_path)
              if scores_path and os.path.exists(scores_path)
              else np.linalg.norm(embeddings, axis=1))
    print(f"  Embeddings: {embeddings.shape}  Scores: {scores.shape}")

    # ── Load graph ─────────────────────────────────────────────
    graph        = load_graph_data(data_dir, device)
    scholar_idxs = graph["scholar_idxs"]
    S            = len(scholar_idxs)
    print(f"  Scholars: {S}")

    # ── Ground truth ───────────────────────────────────────────
    print("\nBuilding ground truth ...")
    evaluable        = build_ground_truth_from_hypergraph(
        graph["hyperedge_index"], graph["hedge_types"], scholar_idxs)
    hedge_to_award   = build_hedge_to_award(data_dir)
    _, award_titles  = build_award_titles(profiles_path)
    print(f"  Award titles loaded: {len(award_titles)}")

    # ── Sample evaluation set ──────────────────────────────────
    rng      = np.random.default_rng(seed)
    eval_ids = list(evaluable.keys())
    if n_eval < len(eval_ids):
        eval_ids = rng.choice(eval_ids, size=n_eval, replace=False).tolist()
    print(f"  Evaluating on {len(eval_ids)} awards (k={k})\n")

    # ── Results containers ─────────────────────────────────────
    ours_records   = []
    topk_records   = []
    random_records = []

    # ── Evaluation loop ────────────────────────────────────────
    for i, hid_str in enumerate(eval_ids):
        gt_local = evaluable[hid_str]

        # Get query from award title
        award_id = hedge_to_award.get(hid_str, "")
        query    = award_titles.get(award_id, "")
        if not query:
            query = "collaborative research in computer science and engineering"

        # ── Ours: QD-DASTF ────────────────────────────────────
        try:
            team_local, info = recommend_team(
                query                   = query,
                embeddings              = embeddings,
                scores                  = scores,
                hyperedge_index         = graph["hyperedge_index"],
                hedge_types             = graph["hedge_types"],
                hedge_years             = graph["hedge_years"],
                scholar_idxs            = scholar_idxs,
                node_embeddings_by_type = None,   # coverage uses embeddings directly
                k                       = k,
                use_llm                 = use_llm,
                top_m                   = S,
                verbose                 = False,
            )
        except Exception as e:
            print(f"  [SKIP] hedge {hid_str}: {e}")
            continue

        # ── Baselines ──────────────────────────────────────────
        topk_team  = baseline_topk(k, scores)
        rand_team  = baseline_random(k, S, seed=i)

        # ── Metrics ───────────────────────────────────────────
        base = {"hedge_id": hid_str, "award_id": award_id,
                "query": query[:100]}

        ours_records.append({
            **base,
            **compute_metrics(team_local, gt_local, embeddings)
        })
        topk_records.append({
            **base,
            **compute_metrics(topk_team, gt_local, embeddings)
        })
        random_records.append({
            **base,
            **compute_metrics(rand_team, gt_local, embeddings)
        })

        if (i + 1) % 20 == 0:
            avg = np.mean([r["f1"] for r in ours_records])
            print(f"  [{i+1}/{len(eval_ids)}] Ours F1={avg:.4f}")

    # ── Aggregate ──────────────────────────────────────────────
    ours_agg   = aggregate(ours_records)
    topk_agg   = aggregate(topk_records)
    random_agg = aggregate(random_records)

    # ── Print table ────────────────────────────────────────────
    metrics_order = ["precision","recall","f1","jaccard",
                     "hit@1","hit@2","quality","diversity","coverage"]
    pct_metrics   = {"precision","recall","f1","jaccard"}

    print("\n" + "="*70)
    print(f"Team Evaluation Results  (k={k}, n={len(ours_records)})")
    print("  P/R/F1/Jaccard shown as % — exact team recovery is naturally sparse")
    print("="*70)
    print(f"  {'Metric':<14} {'Random':>10} {'TopK':>10} {'Ours':>12}")
    print("-"*70)
    for m in metrics_order:
        r = random_agg.get(m, 0)
        t = topk_agg.get(m, 0)
        o = ours_agg.get(m, 0)
        if m in pct_metrics:
            print(f"  {m:<14} {r*100:>9.2f}% {t*100:>9.2f}% {o*100:>11.2f}%")
        else:
            print(f"  {m:<14} {r:>10.4f} {t:>10.4f} {o:>12.4f}")
    print("="*70)
    print("  Quality  ∈ [0,1]: coherence within predicted team")
    print("  Diversity∈ [0,1]: skill spread within predicted team")
    print("  Coverage : fraction of GT members semantically covered (cosine ≥ 0.5)")

    # ── LaTeX table ────────────────────────────────────────────
    # Bold the best result per row (ignoring hit@2 if all zero)
    def best_bold(r, t, o, fmt=".4f"):
        vals = [r, t, o]
        best = max(vals)
        def fmt_val(v):
            s = f"{v:{fmt}}"
            return f"\\textbf{{{s}}}" if v == best else s
        return fmt_val(r), fmt_val(t), fmt_val(o)

    print("\n% LaTeX table rows")
    print(r"\midrule")
    print(r"\multicolumn{4}{l}{\textit{Team Formation (NSF, $k=" + str(k) + r"$)}} \\")
    print(r"\midrule")
    metric_display = [
        ("Precision@$k$", "precision", ".2f", True),
        ("Recall@$k$",    "recall",    ".2f", True),
        ("F1@$k$",        "f1",        ".2f", True),
        ("Jaccard",       "jaccard",   ".4f", False),
        ("Hit@1",         "hit@1",     ".4f", False),
        ("Quality",       "quality",   ".4f", False),
        ("Diversity",     "diversity", ".4f", False),
        ("Coverage",      "coverage",  ".4f", False),
    ]
    for display_name, key, fmt, as_pct in metric_display:
        r = random_agg.get(key, 0)
        t = topk_agg.get(key, 0)
        o = ours_agg.get(key, 0)
        if as_pct:
            rb, tb, ob = best_bold(r*100, t*100, o*100, fmt=fmt)
            suffix = r"\%"
            print(f"{display_name:<20} & {rb}{suffix} & {tb}{suffix} & {ob}{suffix} \\\\")
        else:
            rb, tb, ob = best_bold(r, t, o, fmt=fmt)
            print(f"{display_name:<20} & {rb} & {tb} & {ob} \\\\")

    # ── Save ───────────────────────────────────────────────────
    output = {
        "config":  {"k": k, "n_eval": len(ours_records), "seed": seed},
        "summary": {
            "Random":        random_agg,
            "TopK":          topk_agg,
            "Ours_QD_DASTF": ours_agg,
        },
        "per_award": {
            "ours":   ours_records,
            "topk":   topk_records,
            "random": random_records,
        }
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {output_path}")
    return output


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles_path",
                        default="../../../combined_author_data_2025")
    parser.add_argument("--emb_path",
                        default="../checkpoints_newdata_no_norm/scholar_embeddings.npy")
    parser.add_argument("--scores_path",   default="checkpoints_team/scores.npy")
    parser.add_argument("--data_dir",      default="../training_data_v2/")
    parser.add_argument("--save_dir",      default="checkpoints_team/")
    parser.add_argument("--k",             type=int,  default=5)
    parser.add_argument("--n_eval",        type=int,  default=200)
    parser.add_argument("--use_llm",       action="store_true")
    parser.add_argument("--output",        default="eval_results.json")
    parser.add_argument("--seed",          type=int,  default=42)
    args = parser.parse_args()

    run_evaluation(
        profiles_path = args.profiles_path,
        emb_path      = args.emb_path,
        scores_path   = args.scores_path,
        data_dir      = args.data_dir,
        save_dir      = args.save_dir,
        k             = args.k,
        n_eval        = args.n_eval,
        use_llm       = args.use_llm,
        output_path   = args.output,
        seed          = args.seed,
    )
