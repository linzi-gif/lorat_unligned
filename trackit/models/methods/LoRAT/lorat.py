from typing import Tuple, Union, Iterable, Mapping, Optional
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from trackit.models.backbone.dinov2 import DinoVisionTransformer, interpolate_pos_encoding
from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.head.mlp import MlpAnchorFreeHead, Mlp
from .modules.target_landmark_relation import TargetLandmarkRelation, RelationState
from .funcs.sample_data import generate_LoRAT_sample_data

class LoRAT_DINOv2(nn.Module):
    def __init__(self, vit: DinoVisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 target_landmark_relation_config: Optional[Mapping[str, object]] = None):
        super().__init__()
        assert template_feat_size[0] <= search_region_feat_size[0] and template_feat_size[1] <= search_region_feat_size[1]
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size

        assert isinstance(vit, DinoVisionTransformer)
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        self.blocks = vit.blocks
        self.norm = vit.norm
        self.embed_dim = vit.embed_dim

        self.pos_embed = nn.Parameter(torch.empty(1, self.x_size[0] * self.x_size[1], self.embed_dim))
        self.pos_embed.data.copy_(interpolate_pos_encoding(vit.pos_embed.data[:, 1:, :],
                                                           self.x_size,
                                                           vit.patch_embed.patches_resolution,
                                                           num_prefix_tokens=0, interpolate_offset=0))
        self.token_type_embed = nn.Parameter(torch.empty(3, self.embed_dim))
        self.token_type_embed_new = nn.Parameter(torch.empty(5, self.embed_dim))
        trunc_normal_(self.token_type_embed_new, std=.02)
        self.fuse_search = Mlp(self.embed_dim * 2, out_features=self.embed_dim, num_layers=3)
        self.head = MlpAnchorFreeHead(self.embed_dim, self.x_size)

        relation_cfg = dict(target_landmark_relation_config or {})
        self.relation_build_layer = relation_cfg.get('build_layer', 5)
        self.relation_refresh_layer = relation_cfg.get('refresh_layer', 8)
        self.relation_refine_layers = tuple(relation_cfg.get('refine_layers', (11,)))
        self.inject_layers = tuple(relation_cfg.get('inject_layers', (8,)))
        self.relation_active_layers = tuple(sorted({
            self.relation_build_layer,
            self.relation_refresh_layer,
            *self.relation_refine_layers,
            *self.inject_layers,
        }))
        relation_kwargs = {
            'num_landmarks': relation_cfg.get('num_landmarks', 8),
            'match_candidate_topk': relation_cfg.get('match_candidate_topk', 4),
            'min_landmark_distance': relation_cfg.get('min_landmark_distance', 2.0),
            'route_full_threshold': relation_cfg.get('route_full_threshold', 0.62),
            'route_weak_threshold': relation_cfg.get('route_weak_threshold', 0.38),
            'identity_refine_scale': relation_cfg.get('identity_refine_scale', 0.10),
            'pair_refresh_conf_threshold': relation_cfg.get('pair_refresh_conf_threshold', 0.35),
            'pair_refresh_min_improvement': relation_cfg.get('pair_refresh_min_improvement', 0.02),
            'prompt_heads': relation_cfg.get('prompt_heads', 8),
            'enable_identity_memory': relation_cfg.get('enable_identity_memory', True),
        }
        self.target_landmark_relation = TargetLandmarkRelation(self.embed_dim, self.x_size, **relation_kwargs)

    def forward(self, z: torch.Tensor, x: torch.Tensor, d: torch.Tensor,
                z_feat_mask: torch.Tensor, zi_feat_mask: torch.Tensor,
                  d_feat_mask: torch.Tensor, di_feat_mask: torch.Tensor):
        
        z_feat_v, z_feat_i = self._z_feat(z, z_feat_mask, zi_feat_mask)
        d_feat_v, d_feat_i = self._d_feat(d, d_feat_mask, di_feat_mask)
        x_feat_v, x_feat_i = self._x_feat(x)
        x_feat_v, x_feat_i, relation_state = self._synchronized_fusion(
            z_feat_v, x_feat_v, d_feat_v,
            z_feat_i, x_feat_i, d_feat_i,
        )
        x_feat = self._fuse_search(x_feat_v, x_feat_i)
        output = self.head(x_feat)
        if relation_state is not None:
            output['relation_aux'] = self.target_landmark_relation.export_state(relation_state)
        return output

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor, zi_feat_mask: torch.Tensor):
        z_v = self.patch_embed(z[:, :3])
        z_i = self.patch_embed(z[:, 3:])

        z_W, z_H = self.z_size
        z_v = z_v + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(
            1, z_H * z_W, self.embed_dim)
        z_v = z_v + self.token_type_embed_new[:2][z_feat_mask.flatten(1)]

        z_i = z_i + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(
            1, z_H * z_W, self.embed_dim)
        z_i = z_i + self.token_type_embed_new[:2][zi_feat_mask.flatten(1)]

        return z_v, z_i

    def _d_feat(self, d: torch.Tensor, d_feat_mask: torch.Tensor, di_feat_mask: torch.Tensor):
        d_v = self.patch_embed(d[:, :3])
        d_i = self.patch_embed(d[:, 3:])

        d_W, d_H = self.z_size
        d_v = d_v + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : d_H, : d_W, :].reshape(
            1, d_H * d_W, self.embed_dim)
        d_v = d_v + self.token_type_embed_new[2:4][d_feat_mask.flatten(1)]

        d_i = d_i + self.pos_embed.view(1, self.x_size[1], self.x_size[0], self.embed_dim)[:, : d_H, : d_W, :].reshape(
            1, d_H * d_W, self.embed_dim)
        d_i = d_i + self.token_type_embed_new[2:4][di_feat_mask.flatten(1)]

        return d_v, d_i

    def _x_feat(self, x: torch.Tensor):
        x_v = self.patch_embed(x[:, :3])
        x_i = self.patch_embed(x[:, 3:])

        x_v = x_v + self.pos_embed
        x_v = x_v + self.token_type_embed_new[4].view(1, 1, self.embed_dim)
        x_i = x_i + self.pos_embed
        x_i = x_i + self.token_type_embed_new[4].view(1, 1, self.embed_dim)
        return x_v, x_i

    def _synchronized_fusion(self,
                             z_feat_v: torch.Tensor, x_feat_v: torch.Tensor, d_feat_v: torch.Tensor,
                             z_feat_i: torch.Tensor, x_feat_i: torch.Tensor, d_feat_i: torch.Tensor):
        fusion_v = torch.cat((z_feat_v, x_feat_v, d_feat_v), dim=1)
        fusion_i = torch.cat((z_feat_i, x_feat_i, d_feat_i), dim=1)
        relation_state: Optional[RelationState] = None
        z_len = z_feat_v.shape[1]
        x_len = x_feat_v.shape[1]

        for layer_idx, block in enumerate(self.blocks, start=1):
            fusion_v = block(fusion_v)
            fusion_i = block(fusion_i)

            if layer_idx not in self.relation_active_layers:
                continue

            cur_z_v, cur_x_v, cur_d_v = self._split_triplet(fusion_v, z_len, x_len)
            cur_z_i, cur_x_i, cur_d_i = self._split_triplet(fusion_i, z_len, x_len)

            if layer_idx == self.relation_build_layer:
                relation_state = self.target_landmark_relation.build_state(
                    cur_z_v, cur_z_i, cur_x_v, cur_x_i, cur_d_v, cur_d_i
                )
            elif relation_state is not None:
                relation_state = self.target_landmark_relation.refine_state(
                    cur_z_v,
                    cur_z_i,
                    cur_x_v,
                    cur_x_i,
                    cur_d_v,
                    cur_d_i,
                    relation_state,
                    refresh_pairs=layer_idx == self.relation_refresh_layer,
                )
            else:
                continue

            if layer_idx in self.inject_layers:
                cur_x_v = self.target_landmark_relation.inject_visible(cur_x_v, relation_state)
                fusion_v = self._replace_search(fusion_v, z_len, x_len, cur_x_v)

        fusion_v = self.norm(fusion_v)
        fusion_i = self.norm(fusion_i)
        return self._split_search(fusion_v, z_len, x_len), self._split_search(fusion_i, z_len, x_len), relation_state
    
    def _split_search(self, feat,  z_len, x_len):
        search = feat[:, z_len:z_len + x_len, :]
        return search

    def _split_triplet(self, feat: torch.Tensor, z_len: int, x_len: int):
        x_start = z_len
        d_start = z_len + x_len
        z_feat = feat[:, :z_len, :]
        x_feat = feat[:, x_start:d_start, :]
        d_feat = feat[:, d_start:, :]
        return z_feat, x_feat, d_feat

    def _replace_search(self, feat: torch.Tensor, z_len: int, x_len: int, new_search: torch.Tensor):
        out = feat.clone()
        out[:, z_len:z_len + x_len, :] = new_search
        return out
    
    def _fuse_search(self, search_v, search_i):
        search = torch.cat([search_v, search_i], dim=2)
        return self.fuse_search(search)
        # return search_v

    def reset_tracking_state(self):
        self.target_landmark_relation.reset_state()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.reset_tracking_state()
        return self
    
    def get_sample_data(self, batch_size: int,
                        device: torch.device,
                        dtype: torch.dtype, _) -> Union[torch.Tensor, Iterable, Mapping]:
        return generate_LoRAT_sample_data(self.z_size, self.x_size, self.patch_embed.patch_size,
                                          batch_size, device, dtype)
