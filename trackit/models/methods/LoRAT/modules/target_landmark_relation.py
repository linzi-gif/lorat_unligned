from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head.mlp import Mlp


@dataclass
class RelationState:
    landmark_idx_v: torch.Tensor
    landmark_idx_i: torch.Tensor
    pair_index: torch.Tensor
    pair_conf: torch.Tensor
    target_score_i: torch.Tensor
    target_score_v: torch.Tensor
    thermal_relation_code: torch.Tensor
    geo_prior: torch.Tensor
    geo_feature: torch.Tensor
    relation_tokens: torch.Tensor
    semantic_prompt: torch.Tensor
    route_mode: torch.Tensor
    route_score: torch.Tensor
    global_gate: torch.Tensor
    layer_gate: torch.Tensor
    id_consistency: torch.Tensor
    id_refine_map: torch.Tensor
    diagnostics: dict


def _min_max_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    min_v = x.amin(dim=1, keepdim=True)
    max_v = x.amax(dim=1, keepdim=True)
    return (x - min_v) / (max_v - min_v + eps)


def _batched_gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, gather_index)


def _batched_gather_coords(coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    coords = coords.unsqueeze(0).expand(indices.shape[0], -1, -1)
    gather_index = indices.unsqueeze(-1).expand(-1, -1, coords.shape[-1])
    return torch.gather(coords, 1, gather_index)


def _safe_top2_gap(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x.new_ones((x.shape[0], 1))
    top2 = torch.topk(x, k=2, dim=1).values
    return (top2[:, :1] - top2[:, 1:2]).clamp_min(0.0)


def _normalized_entropy(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = prob / (prob.sum(dim=1, keepdim=True) + eps)
    entropy = -(prob * (prob + eps).log()).sum(dim=1, keepdim=True)
    return entropy / torch.log(torch.tensor(prob.shape[1], dtype=prob.dtype, device=prob.device))


class HistoricalTargetTopologyMemory(nn.Module):
    def __init__(self, embed_dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("history_signature", torch.zeros(1, embed_dim), persistent=False)
        self.register_buffer("history_position", torch.zeros(1, 2), persistent=False)
        self.register_buffer("history_valid", torch.zeros(1, dtype=torch.bool), persistent=False)

    def reset_state(self):
        self.history_signature.zero_()
        self.history_position.zero_()
        self.history_valid.zero_()

    def refine(self, x_v: torch.Tensor, geo_prior: torch.Tensor, coords: torch.Tensor,
               route_quality: torch.Tensor, training: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(geo_prior, dim=1)
        signature = torch.einsum("bn,bnc->bc", weights, x_v)
        position = torch.einsum("bn,nc->bc", weights, coords)

        if training or x_v.shape[0] != 1:
            return signature.new_full((x_v.shape[0], 1), 0.5), torch.zeros_like(geo_prior)

        if not self.history_valid.item():
            if route_quality.mean().item() > 0.55:
                self.history_signature.copy_(signature.detach())
                self.history_position.copy_(position.detach())
                self.history_valid.fill_(True)
            return signature.new_full((1, 1), 0.5), torch.zeros_like(geo_prior)

        history_signature = self.history_signature
        id_consistency = ((F.normalize(signature, dim=-1) * F.normalize(history_signature, dim=-1)).sum(dim=-1, keepdim=True) + 1.0) * 0.5
        refine_map = torch.einsum("bnc,bc->bn", F.normalize(x_v, dim=-1), F.normalize(history_signature, dim=-1))
        pos_bias = 1.0 - (coords.unsqueeze(0) - self.history_position.unsqueeze(1)).pow(2).sum(dim=-1).sqrt().clamp(max=1.0)
        refine_map = 0.7 * _min_max_normalize(refine_map) + 0.3 * _min_max_normalize(pos_bias)

        if route_quality.mean().item() > 0.55:
            self.history_signature.mul_(self.momentum).add_((1.0 - self.momentum) * signature.detach())
            self.history_position.mul_(self.momentum).add_((1.0 - self.momentum) * position.detach())
            self.history_valid.fill_(True)

        return id_consistency, refine_map


class TargetLandmarkRelation(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 search_region_feat_size: Tuple[int, int],
                 num_landmarks: int = 8,
                 match_candidate_topk: int = 4,
                 min_landmark_distance: float = 2.0,
                 route_full_threshold: float = 0.62,
                 route_weak_threshold: float = 0.38,
                 identity_refine_scale: float = 0.10,
                 pair_refresh_conf_threshold: float = 0.35,
                 pair_refresh_min_improvement: float = 0.02,
                 prompt_heads: int = 8,
                 enable_identity_memory: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.search_size = search_region_feat_size
        self.num_landmarks = num_landmarks
        self.match_candidate_topk = match_candidate_topk
        self.min_landmark_distance = min_landmark_distance
        self.route_full_threshold = route_full_threshold
        self.route_weak_threshold = route_weak_threshold
        self.identity_refine_scale = identity_refine_scale
        self.pair_refresh_conf_threshold = pair_refresh_conf_threshold
        self.pair_refresh_min_improvement = pair_refresh_min_improvement
        self.enable_identity_memory = enable_identity_memory

        coord_grid = self._build_coord_grid(search_region_feat_size)
        self.register_buffer("coord_grid", coord_grid, persistent=False)

        relation_dim = 13
        coarse_relation_dim = 7
        topology_dim = 11
        context_dim = embed_dim
        prompt_heads = max(1, min(prompt_heads, embed_dim))
        while embed_dim % prompt_heads != 0 and prompt_heads > 1:
            prompt_heads -= 1

        self.match_proj = nn.Linear(embed_dim, embed_dim)
        self.topology_proj = nn.Linear(topology_dim, embed_dim // 2)
        self.context_proj = nn.Linear(context_dim, embed_dim)
        self.relation_code_proj = nn.Linear(relation_dim, embed_dim)
        self.coarse_relation_code_proj = nn.Linear(coarse_relation_dim, embed_dim)
        self.geo_project = Mlp(embed_dim + 2, hidden_features=embed_dim, out_features=embed_dim, num_layers=2)
        self.relation_token_project = Mlp(embed_dim * 2 + topology_dim + relation_dim + 1,
                                          hidden_features=embed_dim, out_features=embed_dim, num_layers=3)
        self.semantic_attn = nn.MultiheadAttention(embed_dim, prompt_heads, batch_first=True)
        self.semantic_project = Mlp(embed_dim, hidden_features=embed_dim, out_features=embed_dim, num_layers=2)
        self.target_score_scale = nn.Parameter(torch.tensor(6.0))
        self.geo_scale = nn.Parameter(torch.tensor(4.0))
        self.identity_memory = HistoricalTargetTopologyMemory(embed_dim)

    @staticmethod
    def _build_coord_grid(search_region_feat_size: Tuple[int, int]) -> torch.Tensor:
        width, height = search_region_feat_size
        ys, xs = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=height),
            torch.linspace(0.0, 1.0, steps=width),
            indexing="ij",
        )
        return torch.stack((xs, ys), dim=-1).view(height * width, 2)

    def reset_state(self):
        self.identity_memory.reset_state()

    def build_state(self,
                    z_v: torch.Tensor,
                    z_i: torch.Tensor,
                    x_v: torch.Tensor,
                    x_i: torch.Tensor,
                    d_v: torch.Tensor,
                    d_i: torch.Tensor) -> RelationState:
        landmark_idx_v, target_score_v = self._select_landmarks(x_v, z_v, d_v)
        landmark_idx_i, target_score_i = self._select_landmarks(x_i, z_i, d_i)

        landmark_feat_v = _batched_gather_tokens(x_v, landmark_idx_v)
        landmark_feat_i = _batched_gather_tokens(x_i, landmark_idx_i)
        landmark_coord_v = _batched_gather_coords(self.coord_grid, landmark_idx_v)
        landmark_coord_i = _batched_gather_coords(self.coord_grid, landmark_idx_i)

        topo_v = self._encode_landmark_topology(landmark_coord_v)
        topo_i = self._encode_landmark_topology(landmark_coord_i)
        ctx_v = landmark_feat_v - x_v.mean(dim=1, keepdim=True)
        ctx_i = landmark_feat_i - x_i.mean(dim=1, keepdim=True)

        pair_index, pair_conf = self._match_landmarks(landmark_feat_v, landmark_feat_i, topo_v, topo_i, ctx_v, ctx_i)

        return self._assemble_state(
            x_v=x_v,
            x_i=x_i,
            target_score_v=target_score_v,
            target_score_i=target_score_i,
            landmark_idx_v=landmark_idx_v,
            landmark_idx_i=landmark_idx_i,
            pair_index=pair_index,
            pair_conf=pair_conf,
        )

    def refine_state(self,
                     z_v: torch.Tensor,
                     z_i: torch.Tensor,
                     x_v: torch.Tensor,
                     x_i: torch.Tensor,
                     d_v: torch.Tensor,
                     d_i: torch.Tensor,
                     prev_state: RelationState,
                     refresh_pairs: bool = False) -> RelationState:
        target_score_v = self._compute_target_score(x_v, z_v, d_v)
        target_score_i = self._compute_target_score(x_i, z_i, d_i)
        landmark_idx_v = prev_state.landmark_idx_v
        landmark_idx_i = prev_state.landmark_idx_i
        pair_index = prev_state.pair_index
        pair_conf = prev_state.pair_conf

        if refresh_pairs:
            refreshed_pair_index, refreshed_pair_conf = self._refresh_pair_assignment(
                x_v,
                x_i,
                landmark_idx_v,
                landmark_idx_i,
            )
            replace_mask = self._should_replace_pairs(prev_state.pair_conf, refreshed_pair_conf)
            pair_index = torch.where(replace_mask.unsqueeze(-1), refreshed_pair_index, prev_state.pair_index)
            pair_conf = torch.where(replace_mask, refreshed_pair_conf, prev_state.pair_conf)

        return self._assemble_state(
            x_v=x_v,
            x_i=x_i,
            target_score_v=target_score_v,
            target_score_i=target_score_i,
            landmark_idx_v=landmark_idx_v,
            landmark_idx_i=landmark_idx_i,
            pair_index=pair_index,
            pair_conf=pair_conf,
            prev_state=prev_state,
        )

    def inject_visible(self, x_v: torch.Tensor, relation_state: RelationState, inject_scale: float = 1.0) -> torch.Tensor:
        identity_scale = self.identity_refine_scale if (not self.training and self.enable_identity_memory) else 0.0
        final_prior = (relation_state.geo_prior + identity_scale * relation_state.id_refine_map).clamp(0.0, 1.0)
        gate = relation_state.layer_gate * inject_scale
        enhanced = x_v + gate.unsqueeze(-1) * relation_state.semantic_prompt
        enhanced = enhanced + gate.unsqueeze(-1) * final_prior.unsqueeze(-1) * relation_state.geo_feature
        return enhanced

    def export_state(self, relation_state: RelationState) -> dict:
        return {
            "route_mode": relation_state.route_mode,
            "route_score": relation_state.route_score,
            "global_gate": relation_state.global_gate,
            "layer_gate": relation_state.layer_gate,
            "pair_conf": relation_state.pair_conf,
            "geo_prior": relation_state.geo_prior,
            "semantic_energy": relation_state.semantic_prompt.norm(dim=-1),
            "id_consistency": relation_state.id_consistency,
            "diagnostics": relation_state.diagnostics,
        }

    def _assemble_state(self,
                        x_v: torch.Tensor,
                        x_i: torch.Tensor,
                        target_score_v: torch.Tensor,
                        target_score_i: torch.Tensor,
                        landmark_idx_v: torch.Tensor,
                        landmark_idx_i: torch.Tensor,
                        pair_index: torch.Tensor,
                        pair_conf: torch.Tensor,
                        prev_state: Optional[RelationState] = None) -> RelationState:
        landmark_feat_v = _batched_gather_tokens(x_v, landmark_idx_v)
        landmark_feat_i = _batched_gather_tokens(x_i, landmark_idx_i)
        landmark_coord_v = _batched_gather_coords(self.coord_grid, landmark_idx_v)
        landmark_coord_i = _batched_gather_coords(self.coord_grid, landmark_idx_i)

        pair_idx_v = pair_index[..., 0]
        pair_idx_i = pair_index[..., 1]
        matched_feat_v = _batched_gather_tokens(landmark_feat_v, pair_idx_v)
        matched_feat_i = _batched_gather_tokens(landmark_feat_i, pair_idx_i)
        matched_coord_v = _batched_gather_tokens(landmark_coord_v, pair_idx_v)
        matched_coord_i = _batched_gather_tokens(landmark_coord_i, pair_idx_i)
        topo_i = self._encode_landmark_topology(landmark_coord_i)
        matched_topo_i = _batched_gather_tokens(topo_i, pair_idx_i)

        target_coord_i = self._soft_argmax_coords(target_score_i)
        thermal_relation_code = self._build_relation_code(target_coord_i.unsqueeze(1), matched_coord_i).squeeze(1)
        rgb_relation_code = self._build_relation_code(self.coord_grid.unsqueeze(0).expand(x_v.shape[0], -1, -1), matched_coord_v)

        coarse_similarity, full_similarity = self._relation_compatibility(rgb_relation_code, thermal_relation_code, pair_conf)
        route_quality_seed, route_mode_seed = self._route_seed(target_score_i, pair_conf)
        geo_logits = torch.where(route_mode_seed == 2, full_similarity, coarse_similarity)
        geo_prior = torch.sigmoid(self.geo_scale * geo_logits)
        geo_feature = self.geo_project(torch.cat((x_v, geo_prior.unsqueeze(-1), coarse_similarity.unsqueeze(-1)), dim=-1))

        relation_tokens = self._build_relation_tokens(matched_feat_i, matched_feat_v, matched_topo_i, thermal_relation_code, pair_conf)
        semantic_prompt, _ = self.semantic_attn(x_v, relation_tokens, relation_tokens)
        semantic_prompt = self.semantic_project(semantic_prompt) * geo_prior.unsqueeze(-1)

        geo_entropy = _normalized_entropy(F.softmax(geo_logits, dim=1))
        pair_conf_mean = pair_conf.mean(dim=1, keepdim=True)
        pair_coverage = (pair_conf > 0.25).float().mean(dim=1, keepdim=True)
        structure_conf = (1.0 - geo_entropy).clamp(0.0, 1.0)
        route_quality = (0.35 * route_quality_seed + 0.25 * pair_conf_mean + 0.2 * pair_coverage + 0.2 * structure_conf).clamp(0.0, 1.0)

        id_consistency, id_refine_map = self._identity_refine(x_v, geo_prior, route_quality)
        route_mode = self._route_mode_from_quality(route_quality)
        route_score = torch.cat(
            (
                (route_quality < self.route_weak_threshold).float(),
                ((route_quality >= self.route_weak_threshold) & (route_quality < self.route_full_threshold)).float(),
                (route_quality >= self.route_full_threshold).float(),
            ),
            dim=1,
        )

        if prev_state is None:
            global_gate = route_quality
        else:
            global_gate = 0.5 * prev_state.global_gate + 0.5 * route_quality
        raw_gate = global_gate.clamp(0.0, 1.0)
        layer_gate = torch.zeros_like(raw_gate)
        weak_gate = 0.5 * raw_gate
        full_gate = raw_gate
        layer_gate = torch.where(route_mode == 1, weak_gate, layer_gate)
        layer_gate = torch.where(route_mode == 2, full_gate, layer_gate)

        diagnostics = {
            "thermal_peak_conf": _safe_top2_gap(target_score_i),
            "pair_conf_mean": pair_conf_mean,
            "pair_coverage": pair_coverage,
            "geo_entropy": geo_entropy,
            "route_quality": route_quality,
        }

        return RelationState(
            landmark_idx_v=landmark_idx_v,
            landmark_idx_i=landmark_idx_i,
            pair_index=pair_index,
            pair_conf=pair_conf,
            target_score_i=target_score_i,
            target_score_v=target_score_v,
            thermal_relation_code=thermal_relation_code,
            geo_prior=geo_prior,
            geo_feature=geo_feature,
            relation_tokens=relation_tokens,
            semantic_prompt=semantic_prompt,
            route_mode=route_mode,
            route_score=route_score,
            global_gate=global_gate,
            layer_gate=layer_gate,
            id_consistency=id_consistency,
            id_refine_map=id_refine_map,
            diagnostics=diagnostics,
        )

    def _compute_target_score(self, x: torch.Tensor, template: torch.Tensor, online_template: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=-1)
        template_ctx = F.normalize(template.mean(dim=1), dim=-1)
        online_template_ctx = F.normalize(online_template.mean(dim=1), dim=-1)
        fused_template_ctx = F.normalize(template_ctx + online_template_ctx, dim=-1)

        static_sim = torch.einsum("bnc,bc->bn", x_norm, template_ctx)
        online_sim = torch.einsum("bnc,bc->bn", x_norm, online_template_ctx)
        fused_sim = torch.einsum("bnc,bc->bn", x_norm, fused_template_ctx)

        # Let the static and online templates jointly define the target,
        # while still favoring their fused consensus representation.
        target_logit = 0.25 * static_sim + 0.25 * online_sim + 0.5 * fused_sim
        return torch.sigmoid(self.target_score_scale * target_logit)

    def _select_landmarks(self, x: torch.Tensor, template: torch.Tensor, online_template: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_score = self._compute_target_score(x, template, online_template)
        landmark_idx = self._select_landmarks_from_target_score(x, target_score)
        return landmark_idx, target_score

    def _select_landmarks_from_target_score(self, x: torch.Tensor, target_score: torch.Tensor) -> torch.Tensor:
        global_center = self._soft_argmax_coords(target_score)
        dist_to_target = torch.cdist(global_center.unsqueeze(1), self.coord_grid.unsqueeze(0).expand(x.shape[0], -1, -1)).squeeze(1)
        ring_score = (dist_to_target - 0.18).abs()
        ring_score = 1.0 - _min_max_normalize(ring_score)
        structure_score = _min_max_normalize((x - x.mean(dim=1, keepdim=True)).pow(2).mean(dim=-1))
        candidate_score = (1.0 - target_score) * (0.55 + 0.45 * structure_score) * (0.6 + 0.4 * ring_score)
        return self._distance_aware_topk(candidate_score)

    def _distance_aware_topk(self, score: torch.Tensor) -> torch.Tensor:
        num_landmarks = min(self.num_landmarks, score.shape[1])
        coords = self.coord_grid
        selected = []
        order = score.argsort(dim=1, descending=True)
        for b in range(score.shape[0]):
            chosen = []
            for idx in order[b].tolist():
                if len(chosen) >= num_landmarks:
                    break
                if not chosen:
                    chosen.append(idx)
                    continue
                prev = coords[torch.tensor(chosen, device=score.device)]
                cur = coords[idx]
                if torch.cdist(cur.view(1, 2), prev).amin().item() >= self.min_landmark_distance / max(self.search_size):
                    chosen.append(idx)
            if len(chosen) < num_landmarks:
                fill = [idx for idx in order[b].tolist() if idx not in chosen][: num_landmarks - len(chosen)]
                chosen.extend(fill)
            selected.append(torch.tensor(chosen, device=score.device, dtype=torch.long))
        return torch.stack(selected, dim=0)

    def _encode_landmark_topology(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[1] == 1:
            return coords.new_zeros((coords.shape[0], 1, 11))
        delta = coords.unsqueeze(2) - coords.unsqueeze(1)
        dist = delta.pow(2).sum(dim=-1).sqrt()
        eye = torch.eye(coords.shape[1], device=coords.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
        nearest = dist.topk(k=min(4, coords.shape[1] - 1), dim=-1, largest=False).indices
        gather_index = nearest.unsqueeze(-1).expand(-1, -1, -1, 2)
        neighbor_delta = torch.gather(delta, 2, gather_index)
        mean_delta = neighbor_delta.mean(dim=2)
        std_delta = neighbor_delta.std(dim=2, unbiased=False)
        sector = torch.stack(
            (
                (neighbor_delta[..., 0] >= 0).float().mean(dim=2),
                (neighbor_delta[..., 0] < 0).float().mean(dim=2),
                (neighbor_delta[..., 1] >= 0).float().mean(dim=2),
                (neighbor_delta[..., 1] < 0).float().mean(dim=2),
            ),
            dim=-1,
        )
        mean_dist = torch.gather(dist, 2, nearest).mean(dim=2, keepdim=True)
        return torch.cat((coords, mean_delta, std_delta, sector, mean_dist), dim=-1)

    def _match_landmarks(self,
                         feat_v: torch.Tensor,
                         feat_i: torch.Tensor,
                         topo_v: torch.Tensor,
                         topo_i: torch.Tensor,
                         ctx_v: torch.Tensor,
                         ctx_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        desc_v = F.normalize(self.match_proj(feat_v), dim=-1)
        desc_i = F.normalize(self.match_proj(feat_i), dim=-1)
        topo_v = F.normalize(self.topology_proj(topo_v), dim=-1)
        topo_i = F.normalize(self.topology_proj(topo_i), dim=-1)
        ctx_v = F.normalize(self.context_proj(ctx_v), dim=-1)
        ctx_i = F.normalize(self.context_proj(ctx_i), dim=-1)

        feat_cost = torch.cdist(desc_v, desc_i)
        topo_cost = torch.cdist(topo_v, topo_i)
        ctx_cost = torch.cdist(ctx_v, ctx_i)
        total_cost = 0.5 * feat_cost + 0.3 * topo_cost + 0.2 * ctx_cost

        prune_k = min(self.match_candidate_topk, total_cost.shape[-1])
        top_i = total_cost.topk(k=prune_k, largest=False, dim=-1).indices
        candidate_mask = torch.zeros_like(total_cost, dtype=torch.bool)
        candidate_mask.scatter_(2, top_i, True)
        top_v = total_cost.transpose(1, 2).topk(k=prune_k, largest=False, dim=-1).indices
        reverse_mask = torch.zeros_like(total_cost.transpose(1, 2), dtype=torch.bool)
        reverse_mask.scatter_(2, top_v, True)
        candidate_mask = candidate_mask | reverse_mask.transpose(1, 2)
        total_cost = total_cost.masked_fill(~candidate_mask, 1e4)

        pair_index = []
        pair_conf = []
        for b in range(total_cost.shape[0]):
            assignment = self._solve_assignment(total_cost[b])
            pair_index.append(assignment)
            costs = total_cost[b, assignment[:, 0], assignment[:, 1]]
            pair_conf.append(torch.exp(-costs).clamp(0.0, 1.0))
        return torch.stack(pair_index, dim=0), torch.stack(pair_conf, dim=0)

    def _refresh_pair_assignment(self,
                                 x_v: torch.Tensor,
                                 x_i: torch.Tensor,
                                 landmark_idx_v: torch.Tensor,
                                 landmark_idx_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        landmark_feat_v = _batched_gather_tokens(x_v, landmark_idx_v)
        landmark_feat_i = _batched_gather_tokens(x_i, landmark_idx_i)
        landmark_coord_v = _batched_gather_coords(self.coord_grid, landmark_idx_v)
        landmark_coord_i = _batched_gather_coords(self.coord_grid, landmark_idx_i)

        topo_v = self._encode_landmark_topology(landmark_coord_v)
        topo_i = self._encode_landmark_topology(landmark_coord_i)
        ctx_v = landmark_feat_v - x_v.mean(dim=1, keepdim=True)
        ctx_i = landmark_feat_i - x_i.mean(dim=1, keepdim=True)
        return self._match_landmarks(landmark_feat_v, landmark_feat_i, topo_v, topo_i, ctx_v, ctx_i)

    def _should_replace_pairs(self,
                              prev_pair_conf: torch.Tensor,
                              refreshed_pair_conf: torch.Tensor) -> torch.Tensor:
        prev_mean = prev_pair_conf.mean(dim=1, keepdim=True)
        refreshed_mean = refreshed_pair_conf.mean(dim=1, keepdim=True)
        low_conf_ratio = (prev_pair_conf < self.pair_refresh_conf_threshold).float().mean(dim=1, keepdim=True)
        improved = refreshed_mean > (prev_mean + self.pair_refresh_min_improvement)
        return low_conf_ratio.gt(0.25) & improved

    @staticmethod
    def _solve_assignment(cost: torch.Tensor) -> torch.Tensor:
        try:
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
            assignment = torch.stack(
                (
                    torch.as_tensor(row_ind, device=cost.device, dtype=torch.long),
                    torch.as_tensor(col_ind, device=cost.device, dtype=torch.long),
                ),
                dim=-1,
            )
            return assignment
        except Exception:
            remaining_rows = set(range(cost.shape[0]))
            remaining_cols = set(range(cost.shape[1]))
            assignment = []
            flat = torch.argsort(cost.view(-1))
            for flat_idx in flat.tolist():
                row = flat_idx // cost.shape[1]
                col = flat_idx % cost.shape[1]
                if row in remaining_rows and col in remaining_cols:
                    assignment.append((row, col))
                    remaining_rows.remove(row)
                    remaining_cols.remove(col)
                if len(assignment) == min(cost.shape[0], cost.shape[1]):
                    break
            if not assignment:
                assignment = [(i, i) for i in range(min(cost.shape[0], cost.shape[1]))]
            return torch.as_tensor(assignment, device=cost.device, dtype=torch.long)

    def _soft_argmax_coords(self, score: torch.Tensor) -> torch.Tensor:
        weight = F.softmax(score, dim=1)
        return torch.einsum("bn,nc->bc", weight, self.coord_grid)

    def _build_relation_code(self, query_coords: torch.Tensor, ref_coords: torch.Tensor) -> torch.Tensor:
        delta = query_coords.unsqueeze(-2) - ref_coords.unsqueeze(-3)
        dx = delta[..., 0]
        dy = delta[..., 1]
        dist = delta.pow(2).sum(dim=-1).sqrt()

        sign = torch.stack((dx >= 0, dy >= 0), dim=-1).to(query_coords.dtype)
        abs_offset = torch.stack((dx.abs(), dy.abs()), dim=-1)
        sector = torch.stack(
            (
                (dx >= 0) & (dy >= 0),
                (dx < 0) & (dy >= 0),
                (dx < 0) & (dy < 0),
                (dx >= 0) & (dy < 0),
            ),
            dim=-1,
        ).to(query_coords.dtype)
        dist_bucket = torch.stack(
            (
                dist < 0.15,
                (dist >= 0.15) & (dist < 0.3),
                (dist >= 0.3) & (dist < 0.5),
                dist >= 0.5,
            ),
            dim=-1,
        ).to(query_coords.dtype)
        rank = dist.argsort(dim=-1).argsort(dim=-1).to(query_coords.dtype)
        rank = rank / max(1, ref_coords.shape[1] - 1)
        return torch.cat((sign, abs_offset, sector, dist_bucket, rank.unsqueeze(-1)), dim=-1)

    def _relation_compatibility(self, rgb_relation_code: torch.Tensor, thermal_relation_code: torch.Tensor,
                                pair_conf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        thermal_proj = F.normalize(self.relation_code_proj(thermal_relation_code), dim=-1)
        rgb_proj = F.normalize(self.relation_code_proj(rgb_relation_code), dim=-1)
        full_similarity = (rgb_proj * thermal_proj.unsqueeze(1)).sum(dim=-1)

        coarse_idx = torch.tensor([0, 1, 4, 5, 6, 7, 12], device=rgb_relation_code.device)
        thermal_coarse = F.normalize(self.coarse_relation_code_proj(thermal_relation_code.index_select(-1, coarse_idx)), dim=-1)
        rgb_coarse = F.normalize(self.coarse_relation_code_proj(rgb_relation_code.index_select(-1, coarse_idx)), dim=-1)
        coarse_similarity = (rgb_coarse * thermal_coarse.unsqueeze(1)).sum(dim=-1)

        pair_weight = pair_conf.unsqueeze(1)
        coarse_similarity = (coarse_similarity * pair_weight).sum(dim=-1) / (pair_weight.sum(dim=-1) + 1e-6)
        full_similarity = (full_similarity * pair_weight).sum(dim=-1) / (pair_weight.sum(dim=-1) + 1e-6)
        return coarse_similarity, full_similarity

    def _build_relation_tokens(self,
                               matched_feat_i: torch.Tensor,
                               matched_feat_v: torch.Tensor,
                               matched_topo_i: torch.Tensor,
                               thermal_relation_code: torch.Tensor,
                               pair_conf: torch.Tensor) -> torch.Tensor:
        token_input = torch.cat(
            (
                matched_feat_i,
                matched_feat_v,
                matched_topo_i,
                thermal_relation_code,
                pair_conf.unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.relation_token_project(token_input)

    def _route_seed(self, target_score_i: torch.Tensor, pair_conf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        peak_conf = _safe_top2_gap(target_score_i)
        pair_conf_mean = pair_conf.mean(dim=1, keepdim=True)
        pair_coverage = (pair_conf > 0.25).float().mean(dim=1, keepdim=True)
        quality = (0.5 * peak_conf + 0.3 * pair_conf_mean + 0.2 * pair_coverage).clamp(0.0, 1.0)
        return quality, self._route_mode_from_quality(quality)

    def _route_mode_from_quality(self, quality: torch.Tensor) -> torch.Tensor:
        route_mode = torch.zeros_like(quality, dtype=torch.long)
        route_mode = torch.where(quality >= self.route_weak_threshold, torch.ones_like(route_mode), route_mode)
        route_mode = torch.where(quality >= self.route_full_threshold, torch.full_like(route_mode, 2), route_mode)
        return route_mode

    def _identity_refine(self, x_v: torch.Tensor, geo_prior: torch.Tensor, route_quality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_identity_memory:
            return x_v.new_full((x_v.shape[0], 1), 0.5), torch.zeros_like(geo_prior)
        return self.identity_memory.refine(x_v, geo_prior, self.coord_grid, route_quality, self.training)
