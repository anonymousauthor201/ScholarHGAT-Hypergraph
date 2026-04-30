"""
scholar_hgat.py
===============
HGAT + Type-specific Attention + Temporal Decay + Scholar Attribute Encoder

Architecture:
  1. ScholarAttrEncoder  : per-node-type MLP, maps raw features → dense embedding
  2. HGATConv            : one hypergraph attention layer (node→hedge→node)
                           with type-specific attention weights
  3. TemporalDecay       : reweights hedge contributions by recency
  4. ScholarHGAT         : full model stacking the above

Training:
  - Self-supervised via BPR loss on award_team / publication_coauthorship pairs
  - No manual labels needed

Usage:
  python scholar_hgat.py --data_dir training_data/ --epochs 100
"""

import os
import json
import math
import time
import argparse
import numpy as np
import scipy.sparse as sp
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# ─────────────────────────────────────────────────────────────
#  1. Scholar Attribute Encoder
#     Different node types get different MLP projections
# ─────────────────────────────────────────────────────────────

class ScholarAttrEncoder(nn.Module):
    """
    Maps raw node features → uniform hidden_dim embedding.

    For scholar nodes: uses full feature vector (h_index, citations, keywords...)
    For other nodes  : uses available features, zero-pads the rest
    Each node type gets its own linear projection so the model can learn
    type-specific feature importance.
    """

    NODE_TYPES = ["scholar", "paper", "award", "institution",
                  "keyword", "venue", "nsf_program", "nsf_division"]

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # One linear layer per node type
        self.type_projections = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for t in self.NODE_TYPES
        })

        # Fallback for unknown types
        self.default_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, node_types: list[str]) -> torch.Tensor:
        """
        Args:
            x          : (N, input_dim) raw node features
            node_types : list of N strings, one per node

        Returns:
            h : (N, hidden_dim) encoded embeddings
        """
        h = torch.zeros(x.size(0), self._get_hidden_dim(), device=x.device)

        for ntype in self.NODE_TYPES:
            mask = torch.tensor(
                [i for i, t in enumerate(node_types) if t == ntype],
                dtype=torch.long, device=x.device
            )
            if len(mask) == 0:
                continue
            proj = self.type_projections[ntype]
            h[mask] = proj(x[mask])

        # Handle any remaining nodes with default projection
        handled = set(i for i, t in enumerate(node_types) if t in self.NODE_TYPES)
        rest = torch.tensor(
            [i for i in range(len(node_types)) if i not in handled],
            dtype=torch.long, device=x.device
        )
        if len(rest) > 0:
            h[rest] = self.default_projection(x[rest])

        return h

    def _get_hidden_dim(self):
        # Read hidden_dim from first projection
        proj = next(iter(self.type_projections.values()))
        return proj[-1].out_features


# ─────────────────────────────────────────────────────────────
#  2. Temporal Decay
#     More recent hyperedges get higher attention weight
# ─────────────────────────────────────────────────────────────

class TemporalDecay(nn.Module):
    """
    Computes a scalar decay weight for each hyperedge based on recency.

    decay(t) = exp(-λ * (current_year - hedge_year))

    λ is a learnable parameter — the model decides how fast to decay.
    hedge_year is taken from edge attributes:
      - publication_coauthorship → paper year
      - award_team / award_overlap_period → award start year
      - others → current year (no decay)
    """

    def __init__(self, init_lambda: float = 0.1):
        super().__init__()
        # log(lambda) to keep lambda > 0
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda)))

    def forward(self, hedge_years: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hedge_years : (M,) float tensor of hedge years (0 = unknown/no decay)

        Returns:
            weights : (M,) decay weights in (0, 1]
        """
        current_year = datetime.now().year
        lam = torch.exp(self.log_lambda)

        # age = how many years ago; clip to [0, 50]
        age = (current_year - hedge_years).clamp(0, 50)

        # No decay for hedges with unknown year (year == 0)
        decay = torch.where(
            hedge_years > 0,
            torch.exp(-lam * age),
            torch.ones_like(age)
        )
        return decay


# ─────────────────────────────────────────────────────────────
#  3. HGAT Convolution Layer
#     node → hedge aggregation (with type-specific attention)
#     hedge → node aggregation (with temporal decay)
# ─────────────────────────────────────────────────────────────

HEDGE_TYPES = [
    "award_team", "publication_coauthorship", "keyword_cluster",
    "venue_cluster", "institution_affiliation", "nsf_program_community",
    "nsf_division_community", "citation_tier", "temporal_activity",
    "award_overlap_period",
]

class HGATConv(nn.Module):
    """
    One layer of Hypergraph Attention Convolution.

    Two-stage message passing:
      Stage 1 (node → hedge):
        e_j = Σ_{i ∈ N(j)} α_ij * W_type(j) * h_i
        α_ij = softmax over hedge j's members, using type-specific attention

      Stage 2 (hedge → node):
        h_i' = Σ_{j ∈ N(i)} β_ij * temporal_decay(j) * e_j
        β_ij = softmax over node i's hedges
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads

        # Type-specific W matrices for node→hedge (one per hedge type)
        self.W_type = nn.ModuleDict({
            t: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for t in HEDGE_TYPES
        })
        self.W_default = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Attention vectors (node→hedge direction)
        self.attn_node2hedge = nn.Parameter(torch.randn(num_heads, self.head_dim * 2))

        # Attention vectors (hedge→node direction)
        self.attn_hedge2node = nn.Parameter(torch.randn(num_heads, self.head_dim * 2))

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.dropout  = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_node2hedge)
        nn.init.xavier_uniform_(self.attn_hedge2node)

    def forward(
        self,
        h_nodes: torch.Tensor,          # (N, hidden_dim)
        hyperedge_index: torch.Tensor,  # (2, nnz): [node_idx, hedge_idx]
        hedge_types: list[str],         # (M,) hedge type strings
        hedge_weights: torch.Tensor,    # (M,) temporal decay weights
    ) -> torch.Tensor:
        """Returns updated node embeddings (N, hidden_dim)."""

        N   = h_nodes.size(0)
        M   = len(hedge_types)
        D   = h_nodes.size(1)

        node_idx  = hyperedge_index[0]   # (nnz,)
        hedge_idx = hyperedge_index[1]   # (nnz,)
        nnz = node_idx.size(0)

        # ── Stage 1: node → hedge ─────────────────────────
        h_edge_contrib = self._type_transform(h_nodes, node_idx, hedge_idx, hedge_types, M, nnz)
        # h_edge_contrib: (nnz, D)

        # Attention: node→hedge (softmax within each hedge)
        hedge_mean = self._scatter_mean_from_edges(h_edge_contrib, hedge_idx, M)
        h_dst_n2h  = hedge_mean[hedge_idx]
        attn_n2h   = self._attn_softmax(h_edge_contrib, h_dst_n2h,
                                         self.attn_node2hedge, hedge_idx, M)

        # Aggregate into hedges
        h_hedges = self._weighted_scatter(h_edge_contrib, attn_n2h, hedge_idx, M)

        # Apply temporal decay
        h_hedges = h_hedges * hedge_weights.unsqueeze(-1)   # (M, D)

        # ── Stage 2: hedge → node ─────────────────────────
        h_hedge_at_edge = h_hedges[hedge_idx]
        h_node_at_edge  = h_nodes[node_idx]

        attn_h2n = self._attn_softmax(h_hedge_at_edge, h_node_at_edge,
                                       self.attn_hedge2node, node_idx, N)

        h_new = self._weighted_scatter(h_hedge_at_edge, attn_h2n, node_idx, N)

        # Residual + norm
        h_new = self.norm(h_nodes + self.dropout(self.out_proj(h_new)))
        return h_new

    # ── Helpers ───────────────────────────────────────────

    def _type_transform(self, h_nodes, node_idx, hedge_idx, hedge_types, M, nnz):
        D   = h_nodes.size(1)
        dev = h_nodes.device
        out = torch.zeros(nnz, D, device=dev)

        type2idx = {t: i for i, t in enumerate(HEDGE_TYPES)}

        for htype in HEDGE_TYPES:
            W = self.W_type[htype] if htype in self.W_type else self.W_default
            hedge_idx_cpu = hedge_idx.cpu().tolist()
            edge_mask = [k for k in range(nnz)
                         if hedge_idx_cpu[k] < M and hedge_types[hedge_idx_cpu[k]] == htype]
            if not edge_mask:
                continue
            edge_mask_t = torch.tensor(edge_mask, dtype=torch.long, device=dev)
            nids = node_idx[edge_mask_t]
            out[edge_mask_t] = W(h_nodes[nids])

        hedge_idx_cpu = hedge_idx.cpu().tolist()
        unhandled = [k for k in range(nnz)
                     if hedge_idx_cpu[k] >= M or hedge_types[hedge_idx_cpu[k]] not in type2idx]
        if unhandled:
            ut = torch.tensor(unhandled, dtype=torch.long, device=dev)
            out[ut] = self.W_default(h_nodes[node_idx[ut]])

        return out

    def _scatter_mean_from_edges(self, edge_feats, idx, size):
        D   = edge_feats.size(1)
        dev = edge_feats.device
        out = torch.zeros(size, D, device=dev)
        cnt = torch.zeros(size, device=dev)

        idx_exp = idx.unsqueeze(-1).expand_as(edge_feats)
        out.scatter_add_(0, idx_exp, edge_feats)
        cnt.scatter_add_(0, idx, torch.ones(len(idx), device=dev))
        return out / cnt.unsqueeze(-1).clamp(min=1)

    def _weighted_scatter(self, edge_feats, weights, idx, size):
        D   = edge_feats.size(1)
        dev = edge_feats.device
        weighted  = edge_feats * weights.unsqueeze(-1)
        out       = torch.zeros(size, D, device=dev)
        idx_exp   = idx.unsqueeze(-1).expand_as(weighted)
        out.scatter_add_(0, idx_exp, weighted)
        return out

    def _attn_softmax(self, h_src, h_dst, attn_vec, group_idx, num_groups):
        pair   = torch.cat([h_src, h_dst], dim=-1)
        B      = pair.size(0)
        pair   = pair.view(B, self.num_heads, -1)
        scores = (pair * attn_vec.unsqueeze(0)).sum(-1)
        scores = F.leaky_relu(scores, 0.2).mean(-1)

        scores = scores - scores.max()
        exp_s  = torch.exp(scores)
        denom  = torch.zeros(num_groups, device=scores.device)
        denom.scatter_add_(0, group_idx, exp_s)
        return exp_s / (denom[group_idx] + 1e-8)


# ─────────────────────────────────────────────────────────────
#  4. Full Model
# ─────────────────────────────────────────────────────────────

# Relation names in canonical order — single source of truth.
# All weight/vector lookups use this list to guarantee alignment.
RELATION_NAMES = ["award", "paper", "keyword"]

# Initial weight prior: award > paper > keyword (before softmax).
# Reflects domain knowledge: shared awards = strongest collaboration signal.
# These are starting values only; all three are learned during training.
RELATION_INIT_WEIGHTS = {"award": 1.0, "paper": 0.7, "keyword": 0.3}

class ScholarHGAT(nn.Module):
    """
    Full pipeline:
      ScholarAttrEncoder → HGATConv (×num_layers) → Scholar Embeddings

    Temporal decay is applied inside each HGATConv layer.

    Defaults (updated):
      output_dim = 128  (was 64)
      num_layers = 3    (was 2)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,   # ← changed: 64 → 128
        num_layers: int = 3,     # ← changed: 2 → 3
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = ScholarAttrEncoder(input_dim, hidden_dim, dropout)
        self.temporal_decay = TemporalDecay(init_lambda=0.1)

        self.conv_layers = nn.ModuleList([
            HGATConv(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # ── Multi-relation TransE vectors ──
        # Each relation has its own translation vector (r_vec) and a
        # learnable scalar weight (r_weight).  Both are keyed by the same
        # relation name so they can never get out of sync regardless of
        # iteration order.
        self.r_relations = nn.ParameterDict({
            name: nn.Parameter(torch.randn(output_dim) * 0.01)
            for name in RELATION_NAMES
        })
        # Scalar logit per relation — converted to probabilities via softmax
        # at score time.  Stored as a ParameterDict so weight[name] ↔
        # r_relations[name] is always unambiguous.
        self.r_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(RELATION_INIT_WEIGHTS[name]))
            for name in RELATION_NAMES
        })

        self.activity_lambda = nn.Parameter(torch.tensor(0.1))
        self.activity_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hyperedge_index: torch.Tensor,
        node_types: list[str],
        hedge_types: list[str],
        hedge_years: torch.Tensor,
    ) -> torch.Tensor:
        """Returns node embeddings (N, output_dim)."""
        h = self.encoder(x, node_types)
        decay_weights = self.temporal_decay(hedge_years)
        for conv in self.conv_layers:
            h = conv(h, hyperedge_index, hedge_types, decay_weights)
        out = self.output_proj(h)
        return out

    def get_scholar_embeddings(self, out, scholar_idxs):
        idx = torch.tensor(scholar_idxs, dtype=torch.long, device=out.device)
        return F.normalize(out[idx], p=2, dim=-1)

    def transe_score(self, emb, i_idx, j_idx, activity_scores=None):
        h_s  = emb[i_idx]
        h_sp = emb[j_idx]

        # Softmax over raw logits, indexed by name — no positional assumptions
        raw_w = torch.stack([self.r_weights[n] for n in RELATION_NAMES])
        w     = F.softmax(raw_w, dim=0)          # (3,) sums to 1

        base_score = torch.zeros(h_s.size(0), device=emb.device)
        for wi, name in zip(w, RELATION_NAMES):
            r_vec      = self.r_relations[name]  # explicitly keyed
            diff       = h_s + r_vec - h_sp      # TransE: head + rel ≈ tail
            base_score = base_score + wi * (-torch.norm(diff, p=2, dim=-1))

        if activity_scores is not None:
            base_score = base_score + self.activity_lambda * activity_scores[j_idx]
        return base_score

    def score_all(self, emb, query_idx, activity_scores=None):
        h_s = emb[query_idx].unsqueeze(0)        # (1, d)

        raw_w = torch.stack([self.r_weights[n] for n in RELATION_NAMES])
        w     = F.softmax(raw_w, dim=0)          # (3,)

        scores = torch.zeros(emb.size(0), device=emb.device)
        for wi, name in zip(w, RELATION_NAMES):
            r_vec  = self.r_relations[name]      # explicitly keyed
            diff   = h_s + r_vec - emb           # (n_scholars, d)
            scores = scores + wi * (-torch.norm(diff, p=2, dim=-1))

        if activity_scores is not None:
            scores = scores + self.activity_lambda * activity_scores
        return scores

    def compute_activity_scores(self, X_scholar):
        last_pub  = X_scholar[:, 10].unsqueeze(-1)
        log_award = X_scholar[:, 5].unsqueeze(-1)
        experience= X_scholar[:, 3].unsqueeze(-1)
        activity_feats = torch.cat([last_pub, log_award, experience], dim=-1)
        return self.activity_mlp(activity_feats).squeeze(-1)


# ─────────────────────────────────────────────────────────────
#  5. Margin Ranking Loss (Eq. 6)
# ─────────────────────────────────────────────────────────────

def margin_loss(model, emb, pos_pairs, neg_pairs, activity_scores=None, margin=1.0):
    s_idx  = pos_pairs[:, 0]
    sp_idx = pos_pairs[:, 1]
    sn_idx = neg_pairs[:, 1]
    score_pos = model.transe_score(emb, s_idx, sp_idx, activity_scores)
    score_neg = model.transe_score(emb, s_idx, sn_idx, activity_scores)
    return F.relu(margin + score_neg - score_pos).mean()


# ─────────────────────────────────────────────────────────────
#  5b. Dynamic Hard Negative Mining
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def mine_hard_negatives(
    model, emb, train_pos, activity_scores,
    n_scholars, scholar_idx_map, batch_size=512, top_k_pool=50,
):
    pos_set = set()
    for pair in train_pos.cpu().numpy().tolist():
        a = scholar_idx_map.get(int(pair[0]), None)
        b = scholar_idx_map.get(int(pair[1]), None)
        if a is not None and b is not None:
            pos_set.add((a, b))
            pos_set.add((b, a))

    valid_queries = list({scholar_idx_map[int(p[0].item())]
                          for p in train_pos
                          if int(p[0].item()) in scholar_idx_map})

    hard_negs = []
    query_sample = np.random.choice(valid_queries,
                                    size=min(batch_size, len(valid_queries)),
                                    replace=False)

    for s_local in query_sample:
        scores = model.score_all(emb, s_local, activity_scores).cpu().numpy()
        scores[s_local] = -1e9
        for (a, b) in pos_set:
            if a == s_local and b < len(scores):
                scores[b] = -1e9
        top_pool = np.argsort(-scores)[:top_k_pool]
        chosen   = int(np.random.choice(top_pool))
        hard_negs.append([s_local, chosen])

    if not hard_negs:
        hard_negs = [[np.random.randint(0, n_scholars),
                      np.random.randint(0, n_scholars)]
                     for _ in range(batch_size)]

    return torch.tensor(hard_negs, dtype=torch.long, device=emb.device)


# ─────────────────────────────────────────────────────────────
#  6. Data Loading Helpers
# ─────────────────────────────────────────────────────────────

def load_training_data(data_dir, device):
    H = sp.load_npz(os.path.join(data_dir, "H.npz")).tocoo()
    hyperedge_index = torch.tensor(
        np.vstack([H.row, H.col]), dtype=torch.long, device=device
    )
    X = torch.tensor(
        np.load(os.path.join(data_dir, "X.npy")),
        dtype=torch.float, device=device
    )
    with open(os.path.join(data_dir, "node_id_map.json")) as f:
        id_map = json.load(f)
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)

    node_types   = _extract_node_types(id_map)
    hedge_types, hedge_years = _extract_hedge_info(id_map, meta)
    hedge_years_t = torch.tensor(hedge_years, dtype=torch.float, device=device)

    splits_dir = os.path.join(data_dir, "splits")

    def load_pairs(name):
        path = os.path.join(splits_dir, f"{name}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            if len(arr) > 0:
                return torch.tensor(arr, dtype=torch.long, device=device)
        return None

    return {
        "X"                : X,
        "hyperedge_index"  : hyperedge_index,
        "node_types"       : node_types,
        "hedge_types"      : hedge_types,
        "hedge_years"      : hedge_years_t,
        "scholar_idxs"     : id_map["scholar_idxs"],
        "train_pos"        : load_pairs("train_pairs"),
        "val_pos"          : load_pairs("val_pairs"),
        "test_pos"         : load_pairs("test_pairs"),
        "neg_pairs"        : load_pairs("neg_pairs"),
        "num_nodes"        : meta["num_nodes"],
        "num_hedges"       : meta["num_hyperedges"],
        "input_dim"        : meta["feature_dim"],
        "id_map"           : id_map,
    }


def _extract_node_types(id_map):
    idx2node = id_map["idx2node"]
    return [idx2node[str(i)].split("::")[0] for i in range(len(idx2node))]


def _extract_hedge_info(id_map, meta):
    hedge_ids = id_map["idx2hedge"]
    m = len(hedge_ids)
    type_map = {
        "award": "award_team", "pub": "publication_coauthorship",
        "kw": "keyword_cluster", "venue": "venue_cluster",
        "inst": "institution_affiliation", "prog": "nsf_program_community",
        "div": "nsf_division_community", "tier": "citation_tier",
        "temporal": "temporal_activity", "award_overlap": "award_overlap_period",
    }
    types, years = [], []
    for j in range(m):
        hid   = hedge_ids[str(j)]
        parts = hid.split("::")
        prefix = parts[0].replace("h_", "")
        types.append(type_map.get(prefix, "unknown"))
        year = 0.0
        if prefix in ("temporal", "award_overlap") and len(parts) > 1:
            try:
                year = float(parts[1].split("_")[0])
            except:
                pass
        years.append(year)
    return types, years


# ─────────────────────────────────────────────────────────────
#  7. Evaluation: Recall@K, NDCG@K  (with inference timing)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, data, pos_pairs, train_pos=None, k_list=[5, 10, 20]):
    model.eval()
    device = data["X"].device

    # ── Inference timing ──────────────────────────────────────
    # Warm-up pass (not counted)
    _ = model(data["X"], data["hyperedge_index"],
              data["node_types"], data["hedge_types"], data["hedge_years"])
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    out = model(
        data["X"], data["hyperedge_index"],
        data["node_types"], data["hedge_types"], data["hedge_years"]
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - t0) * 1000
    # ─────────────────────────────────────────────────────────

    emb = model.get_scholar_embeddings(out, data["scholar_idxs"])
    n_scholars = emb.size(0)

    X_scholar = data["X"][data["scholar_idxs"]]
    activity_scores = model.compute_activity_scores(X_scholar)
    scholar_idx_map = {gid: i for i, gid in enumerate(data["scholar_idxs"])}

    gt = {}
    for pair in pos_pairs.cpu().numpy():
        a, b = int(pair[0]), int(pair[1])
        if a in scholar_idx_map and b in scholar_idx_map:
            sa, sb = scholar_idx_map[a], scholar_idx_map[b]
            gt.setdefault(sa, set()).add(sb)
            gt.setdefault(sb, set()).add(sa)

    if not gt:
        return {f"Recall@{k}": 0.0 for k in k_list}, inference_ms

    known = {}
    if train_pos is not None:
        for pair in train_pos.cpu().numpy():
            a, b = int(pair[0]), int(pair[1])
            if a in scholar_idx_map and b in scholar_idx_map:
                sa, sb = scholar_idx_map[a], scholar_idx_map[b]
                known.setdefault(sa, set()).add(sb)
                known.setdefault(sb, set()).add(sa)

    metrics = {f"Recall@{k}": [] for k in k_list}
    metrics.update({f"NDCG@{k}": [] for k in k_list})

    for s in gt.keys():
        scores = model.score_all(emb, s, activity_scores).cpu().numpy()
        scores[s] = -1e9
        for excl in known.get(s, set()):
            scores[excl] = -1e9
        ranked    = np.argsort(-scores)
        positives = gt[s]
        for k in k_list:
            topk = set(ranked[:k].tolist())
            hits = len(topk & positives)
            metrics[f"Recall@{k}"].append(hits / len(positives))
            dcg = sum(1.0 / math.log2(rank + 2)
                      for rank, idx in enumerate(ranked[:k]) if idx in positives)
            ideal = sum(1.0 / math.log2(r2 + 2)
                        for r2 in range(min(k, len(positives))))
            metrics[f"NDCG@{k}"].append(dcg / ideal if ideal > 0 else 0.0)

    return {k: float(np.mean(v)) for k, v in metrics.items()}, inference_ms


def mmr_rerank(scores, emb_np, excluded, k=10, lambda_mmr=0.5):
    """
    Maximal Marginal Relevance reranking.
    Balances relevance (λ) and diversity (1-λ).
    """
    candidates = [i for i in range(len(scores)) if i not in excluded]
    if not candidates:
        return []
    selected, remaining = [], set(candidates)
    for _ in range(k):
        if not remaining:
            break
        if not selected:
            best = max(remaining, key=lambda i: scores[i])
        else:
            sel_emb = emb_np[selected]
            best, best_score = None, -1e9
            for i in remaining:
                sims = sel_emb @ emb_np[i]
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                mmr_score = lambda_mmr * scores[i] - (1 - lambda_mmr) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i
        selected.append(best)
        remaining.remove(best)
    return selected


# ─────────────────────────────────────────────────────────────
#  8. Training Loop  (with epoch + total timing)
# ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading data...")
    data = load_training_data(args.data_dir, device)
    print(f"  Nodes: {data['num_nodes']}, Hedges: {data['num_hedges']}")
    print(f"  Scholars: {len(data['scholar_idxs'])}")
    print(f"  Train pairs: {len(data['train_pos']) if data['train_pos'] is not None else 0}")

    if data["train_pos"] is None or len(data["train_pos"]) == 0:
        print("No training pairs found. Run to_training_data.py first.")
        return

    # ── Model (output_dim=128, num_layers=3) ──────────────────
    model = ScholarHGAT(
        input_dim  = data["input_dim"],
        hidden_dim = args.hidden_dim,
        output_dim = args.output_dim,   # default 128
        num_layers = args.num_layers,   # default 3
        num_heads  = args.num_heads,
        dropout    = args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    print(f"  output_dim={args.output_dim}, num_layers={args.num_layers}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_val_recall = 0.0
    best_epoch      = 0
    train_pos       = data["train_pos"]
    n_scholars      = len(data["scholar_idxs"])
    scholar_idx_map = {gid: i for i, gid in enumerate(data["scholar_idxs"])}
    cached_hard_negs  = None
    hard_neg_interval = 5

    # ── Accumulators for timing ────────────────────────────────
    epoch_times   = []          # wall-clock seconds per training epoch
    total_train_start = time.perf_counter()
    # ──────────────────────────────────────────────────────────

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # ── Per-epoch timer start ──────────────────────────────
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        # ──────────────────────────────────────────────────────

        out = model(
            data["X"], data["hyperedge_index"],
            data["node_types"], data["hedge_types"], data["hedge_years"]
        )
        emb = model.get_scholar_embeddings(out, data["scholar_idxs"])

        X_scholar       = data["X"][data["scholar_idxs"]]
        activity_scores = model.compute_activity_scores(X_scholar)

        if epoch % hard_neg_interval == 1 or cached_hard_negs is None:
            with torch.no_grad():
                cached_hard_negs = mine_hard_negatives(
                    model, emb.detach(), train_pos,
                    activity_scores.detach(), n_scholars,
                    scholar_idx_map=scholar_idx_map,
                    batch_size=args.batch_size,
                )

        batch_idx = torch.randperm(len(train_pos))[:args.batch_size]
        pos_batch = train_pos[batch_idx]
        pos_local = torch.tensor(
            [[scholar_idx_map.get(p[0].item(), 0),
              scholar_idx_map.get(p[1].item(), 0)]
             for p in pos_batch],
            dtype=torch.long, device=device
        )

        n_hard = int(args.batch_size * 0.7)
        n_rand = args.batch_size - n_hard
        hard_idx       = torch.randperm(len(cached_hard_negs))[:n_hard]
        hard_neg_batch = cached_hard_negs[hard_idx]
        rand_neg = torch.stack([
            pos_local[:n_rand, 0],
            torch.randint(0, n_scholars, (n_rand,), device=device)
        ], dim=1)
        neg_local = torch.cat([hard_neg_batch, rand_neg], dim=0)[:args.batch_size]

        loss = margin_loss(model, emb, pos_local, neg_local,
                           activity_scores=activity_scores, margin=args.margin)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── Per-epoch timer end ────────────────────────────────
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_times.append(epoch_elapsed)
        # ──────────────────────────────────────────────────────

        if epoch % args.eval_every == 0:
            val_metrics, infer_ms = evaluate(
                model, data, data["val_pos"], train_pos=data["train_pos"]
            )
            val_recall = val_metrics.get("Recall@10", 0.0)

            avg_epoch_ms = np.mean(epoch_times[-args.eval_every:]) * 1000

            print(
                f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                f"Val Recall@10: {val_recall:.4f} | "
                f"Val NDCG@10: {val_metrics.get('NDCG@10', 0):.4f} | "
                f"Epoch time: {avg_epoch_ms:.1f} ms | "
                f"Inference time: {infer_ms:.1f} ms"
            )

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_epoch = epoch
                torch.save(model.state_dict(),
                           os.path.join(args.save_dir, "best_model.pt"))

    # ── Total training time ────────────────────────────────────
    total_train_elapsed = time.perf_counter() - total_train_start
    avg_epoch_ms        = np.mean(epoch_times) * 1000
    print(f"\n{'─'*60}")
    print(f"Training complete")
    print(f"  Total training time : {total_train_elapsed:.1f} s  "
          f"({total_train_elapsed / 60:.2f} min)")
    print(f"  Avg time / epoch    : {avg_epoch_ms:.1f} ms")
    print(f"  Best epoch          : {best_epoch}  "
          f"(Val Recall@10 = {best_val_recall:.4f})")
    print(f"{'─'*60}\n")
    # ──────────────────────────────────────────────────────────

    model.load_state_dict(
        torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=device)
    )

    if data["test_pos"] is not None:
        test_metrics, test_infer_ms = evaluate(
            model, data, data["test_pos"], train_pos=data["train_pos"]
        )
        print("Test metrics:")
        for k, v in sorted(test_metrics.items()):
            print(f"  {k}: {v:.4f}")
        print(f"  Inference time (full graph): {test_infer_ms:.1f} ms")

    # Save final embeddings
    model.eval()
    with torch.no_grad():
        out = model(
            data["X"], data["hyperedge_index"],
            data["node_types"], data["hedge_types"], data["hedge_years"]
        )
        scholar_emb = model.get_scholar_embeddings(out, data["scholar_idxs"])
        np.save(
            os.path.join(args.save_dir, "scholar_embeddings.npy"),
            scholar_emb.cpu().numpy()
        )
    print(f"Scholar embeddings saved → {args.save_dir}/scholar_embeddings.npy")
    print(f"Embedding shape: {scholar_emb.shape}  (n_scholars × {args.output_dim})")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="training_data/")
    parser.add_argument("--save_dir",   default="checkpoints/")
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--output_dim", type=int,   default=128)   # ← changed: 64 → 128
    parser.add_argument("--num_layers", type=int,   default=3)     # ← changed: 2 → 3
    parser.add_argument("--num_heads",  type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=512)
    parser.add_argument("--margin",     type=float, default=1.0)
    parser.add_argument("--eval_every", type=int,   default=10)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)