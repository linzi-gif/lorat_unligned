import torch
import torch.nn as nn
import torch.nn.functional as F
from trackit.criteria import CriterionOutput
from trackit.criteria.modules.iou_loss import bbox_overlaps
from trackit.core.runtime.context.epoch import get_current_epoch_context
from trackit.miscellanies.torch.distributed.reduce_mean import reduce_mean_


class SimpleCriteria(nn.Module):
    def __init__(self, cls_loss: nn.Module, bbox_reg_loss: nn.Module,
                 iou_aware_classification_score: bool,
                 cls_loss_weight: float, bbox_reg_loss_weight: float,
                 cls_loss_display_name: str, bbox_reg_loss_display_name: str,
                 relation_geo_loss_weight: float = 0.15,
                 relation_gate_loss_weight: float = 0.03,
                 relation_semantic_region_loss_weight: float = 0.02,
                 relation_aux_warmup_epochs: int = 2,
                 relation_semantic_warmup_start_epoch: int = 2,
                 relation_semantic_warmup_end_epoch: int = 5,
                 relation_geo_negative_weight: float = 0.25,
                 relation_semantic_negative_weight: float = 0.25,
                 relation_gate_reduction_factor_on_main_loss_spike: float = 0.5,
                 relation_main_loss_spike_ratio: float = 1.15,
                 relation_main_loss_ema_momentum: float = 0.9):
        super().__init__()
        self.cls_loss = cls_loss
        self.bbox_reg_loss = bbox_reg_loss
        self.iou_aware_classification_score = iou_aware_classification_score
        self.cls_loss_weight = cls_loss_weight
        self.bbox_reg_loss_weight = bbox_reg_loss_weight
        self.cls_loss_display_name = cls_loss_display_name
        self.bbox_reg_loss_display_name = bbox_reg_loss_display_name
        self._across_all_nodes_normalization = True
        self.relation_geo_loss_weight = relation_geo_loss_weight
        self.relation_gate_loss_weight = relation_gate_loss_weight
        self.relation_semantic_region_loss_weight = relation_semantic_region_loss_weight
        self.relation_aux_warmup_epochs = relation_aux_warmup_epochs
        self.relation_semantic_warmup_start_epoch = relation_semantic_warmup_start_epoch
        self.relation_semantic_warmup_end_epoch = relation_semantic_warmup_end_epoch
        self.relation_geo_negative_weight = relation_geo_negative_weight
        self.relation_semantic_negative_weight = relation_semantic_negative_weight
        self.relation_gate_reduction_factor_on_main_loss_spike = relation_gate_reduction_factor_on_main_loss_spike
        self.relation_main_loss_spike_ratio = relation_main_loss_spike_ratio
        self.relation_main_loss_ema_momentum = relation_main_loss_ema_momentum
        self._main_loss_ema = None

    def _get_relation_warmup_factor(self) -> float:
        if self.relation_aux_warmup_epochs <= 0:
            return 1.0
        epoch_context = get_current_epoch_context()
        if epoch_context is None:
            return 1.0
        return float(min(max(epoch_context.epoch, 0), self.relation_aux_warmup_epochs)) / float(self.relation_aux_warmup_epochs)

    def _get_semantic_warmup_factor(self) -> float:
        if self.relation_semantic_region_loss_weight <= 0.0:
            return 0.0
        epoch_context = get_current_epoch_context()
        if epoch_context is None:
            return 1.0
        epoch = float(max(epoch_context.epoch, 0))
        start = float(self.relation_semantic_warmup_start_epoch)
        end = float(self.relation_semantic_warmup_end_epoch)
        if end <= start:
            return 1.0 if epoch >= end else 0.0
        if epoch <= start:
            return 0.0
        if epoch >= end:
            return 1.0
        return (epoch - start) / (end - start)

    def _build_geo_targets(self,
                           num_batches: int,
                           response_map_size: int,
                           device: torch.device,
                           positive_sample_batch_dim_index: torch.Tensor,
                           positive_sample_feature_map_dim_index: torch.Tensor) -> torch.Tensor:
        geo_target_map = torch.zeros((num_batches, response_map_size), dtype=torch.float32, device=device)
        if positive_sample_batch_dim_index is not None:
            geo_target_map[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index] = 1.0
        return geo_target_map

    def _compute_relation_geo_loss(self, geo_prior: torch.Tensor, geo_target_map: torch.Tensor) -> torch.Tensor:
        geo_prior = geo_prior.to(torch.float).clamp(1e-4, 1.0 - 1e-4)
        geo_target_map = geo_target_map.to(geo_prior.dtype)
        geo_logits = torch.logit(geo_prior)
        geo_loss_map = F.binary_cross_entropy_with_logits(geo_logits, geo_target_map, reduction='none')

        positive_mask = geo_target_map > 0.5
        negative_mask = ~positive_mask

        if positive_mask.any():
            positive_loss = geo_loss_map.masked_select(positive_mask).mean()
        else:
            positive_loss = geo_loss_map.new_zeros(())

        if negative_mask.any():
            negative_loss = geo_loss_map.masked_select(negative_mask).mean()
        else:
            negative_loss = geo_loss_map.new_zeros(())

        return positive_loss + self.relation_geo_negative_weight * negative_loss

    @staticmethod
    def _compute_gate_target(geo_prior: torch.Tensor, geo_target_map: torch.Tensor) -> torch.Tensor:
        geo_prior = geo_prior.detach().to(torch.float)
        geo_target_map = geo_target_map.to(geo_prior.dtype)
        positive_norm = geo_target_map.sum(dim=1, keepdim=True).clamp_min(1.0)
        negative_target_map = 1.0 - geo_target_map
        negative_norm = negative_target_map.sum(dim=1, keepdim=True).clamp_min(1.0)

        positive_coverage = (geo_prior * geo_target_map).sum(dim=1, keepdim=True) / positive_norm
        negative_leak = (geo_prior * negative_target_map).sum(dim=1, keepdim=True) / negative_norm
        return (positive_coverage - 0.5 * negative_leak).clamp(0.0, 1.0)

    def _compute_relation_gate_loss(self,
                                    global_gate: torch.Tensor,
                                    geo_prior: torch.Tensor,
                                    geo_target_map: torch.Tensor) -> torch.Tensor:
        gate_target = self._compute_gate_target(geo_prior, geo_target_map)
        global_gate = global_gate.to(torch.float).view(gate_target.shape[0], -1).mean(dim=1, keepdim=True).clamp(1e-4, 1.0 - 1e-4)
        gate_logits = torch.logit(global_gate)
        return F.binary_cross_entropy_with_logits(gate_logits, gate_target, reduction='mean')

    def _compute_relation_semantic_region_loss(self,
                                               semantic_energy: torch.Tensor,
                                               layer_gate: torch.Tensor,
                                               geo_target_map: torch.Tensor) -> torch.Tensor:
        semantic_energy = semantic_energy.to(torch.float)
        geo_target_map = geo_target_map.to(semantic_energy.dtype)
        layer_gate = layer_gate.to(torch.float).view(semantic_energy.shape[0], -1).mean(dim=1, keepdim=True)
        semantic_effect = semantic_energy * layer_gate
        semantic_map = semantic_effect / semantic_effect.amax(dim=1, keepdim=True).clamp_min(1e-6)
        semantic_logits = torch.logit(semantic_map.clamp(1e-6, 1.0 - 1e-6))
        semantic_loss_map = F.binary_cross_entropy_with_logits(semantic_logits, geo_target_map, reduction='none')

        positive_mask = geo_target_map > 0.5
        negative_mask = ~positive_mask

        if positive_mask.any():
            positive_loss = semantic_loss_map.masked_select(positive_mask).mean()
        else:
            positive_loss = semantic_loss_map.new_zeros(())

        if negative_mask.any():
            negative_loss = semantic_loss_map.masked_select(negative_mask).mean()
        else:
            negative_loss = semantic_loss_map.new_zeros(())

        return positive_loss + self.relation_semantic_negative_weight * negative_loss

    def _get_gate_loss_scale(self, main_loss: torch.Tensor) -> float:
        if not self.training or self.relation_gate_loss_weight <= 0.0:
            return 1.0

        reduced_main_loss = main_loss.detach().to(torch.float).view(1)
        reduce_mean_(reduced_main_loss)
        current_main_loss = reduced_main_loss.item()

        gate_loss_scale = 1.0
        if self._main_loss_ema is not None and current_main_loss > self._main_loss_ema * self.relation_main_loss_spike_ratio:
            gate_loss_scale = self.relation_gate_reduction_factor_on_main_loss_spike

        if self._main_loss_ema is None:
            self._main_loss_ema = current_main_loss
        else:
            momentum = self.relation_main_loss_ema_momentum
            self._main_loss_ema = momentum * self._main_loss_ema + (1.0 - momentum) * current_main_loss

        return gate_loss_scale

    def forward(self, outputs: dict, targets: dict):
        num_positive_samples = targets['num_positive_samples']
        assert isinstance(num_positive_samples, torch.Tensor)

        reduce_mean_(num_positive_samples)  # caution: inplace update
        num_positive_samples.clamp_(min=1.)

        predicted_score_map = outputs['score_map'].to(torch.float)
        predicted_bboxes = outputs['boxes'].to(torch.float)
        groundtruth_bboxes = targets['boxes']

        N, H, W = predicted_score_map.shape

        # shape: (num_positive_samples, )
        positive_sample_batch_dim_index = targets['positive_sample_batch_dim_indices']
        # shape: (num_positive_samples, )
        positive_sample_feature_map_dim_index = targets['positive_sample_map_dim_indices']

        has_positive_samples = positive_sample_batch_dim_index is not None

        if has_positive_samples:
            predicted_bboxes = predicted_bboxes.view(N, H * W, 4)
            predicted_bboxes = predicted_bboxes[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index]
            groundtruth_bboxes = groundtruth_bboxes[positive_sample_batch_dim_index]

        with torch.no_grad():
            groundtruth_response_map = torch.zeros((N, H * W),  dtype=torch.float32, device=predicted_score_map.device)
            if has_positive_samples:
                if self.iou_aware_classification_score:
                    groundtruth_response_map.index_put_(
                        (positive_sample_batch_dim_index, positive_sample_feature_map_dim_index),
                        bbox_overlaps(groundtruth_bboxes, predicted_bboxes, is_aligned=True))
                else:
                    groundtruth_response_map[positive_sample_batch_dim_index, positive_sample_feature_map_dim_index] = 1.
            groundtruth_geo_map = self._build_geo_targets(
                N,
                H * W,
                predicted_score_map.device,
                positive_sample_batch_dim_index,
                positive_sample_feature_map_dim_index,
            )

        cls_loss = self.cls_loss(predicted_score_map.view(N, -1), groundtruth_response_map).sum() / num_positive_samples

        if has_positive_samples:
            reg_loss = self.bbox_reg_loss(predicted_bboxes, groundtruth_bboxes).sum() / num_positive_samples
        else:
            reg_loss = predicted_bboxes.mean() * 0

        if self.cls_loss_weight != 1.:
            cls_loss = cls_loss * self.cls_loss_weight
        if self.bbox_reg_loss_weight != 1.:
            reg_loss = reg_loss * self.bbox_reg_loss_weight

        main_loss = cls_loss + reg_loss
        relation_geo_loss = main_loss.new_zeros(())
        relation_gate_loss = main_loss.new_zeros(())
        relation_semantic_loss = main_loss.new_zeros(())
        relation_geo_loss_scaled = main_loss.new_zeros(())
        relation_gate_loss_scaled = main_loss.new_zeros(())
        relation_semantic_loss_scaled = main_loss.new_zeros(())
        relation_warmup_factor = self._get_relation_warmup_factor()
        relation_semantic_warmup_factor = self._get_semantic_warmup_factor()
        relation_gate_loss_scale = self._get_gate_loss_scale(main_loss)

        relation_aux = outputs.get('relation_aux')
        if relation_aux is not None:
            geo_prior = relation_aux.get('geo_prior')
            global_gate = relation_aux.get('global_gate')
            layer_gate = relation_aux.get('layer_gate')
            semantic_energy = relation_aux.get('semantic_energy')
            if geo_prior is not None and self.relation_geo_loss_weight > 0.0:
                relation_geo_loss = self._compute_relation_geo_loss(geo_prior.view(N, -1), groundtruth_geo_map)
                relation_geo_loss_scaled = relation_geo_loss * (relation_warmup_factor * self.relation_geo_loss_weight)
            if geo_prior is not None and global_gate is not None and self.relation_gate_loss_weight > 0.0:
                relation_gate_loss = self._compute_relation_gate_loss(global_gate, geo_prior.view(N, -1), groundtruth_geo_map)
                relation_gate_loss_scaled = relation_gate_loss * (
                    relation_warmup_factor * self.relation_gate_loss_weight * relation_gate_loss_scale
                )
            if semantic_energy is not None and layer_gate is not None and self.relation_semantic_region_loss_weight > 0.0:
                relation_semantic_loss = self._compute_relation_semantic_region_loss(
                    semantic_energy.view(N, -1),
                    layer_gate,
                    groundtruth_geo_map,
                )
                relation_semantic_loss_scaled = relation_semantic_loss * (
                    relation_semantic_warmup_factor * self.relation_semantic_region_loss_weight
                )

        total_loss = main_loss + relation_geo_loss_scaled + relation_gate_loss_scaled + relation_semantic_loss_scaled

        cls_loss_cpu = cls_loss.detach().cpu().item()
        reg_loss_cpu = reg_loss.detach().cpu().item()
        relation_geo_loss_cpu = relation_geo_loss_scaled.detach().cpu().item()
        relation_gate_loss_cpu = relation_gate_loss_scaled.detach().cpu().item()
        relation_semantic_loss_cpu = relation_semantic_loss_scaled.detach().cpu().item()
        main_loss_cpu = main_loss.detach().cpu().item()

        metrics = {
            f'Loss/{self.cls_loss_display_name}': cls_loss_cpu,
            f'Loss/{self.bbox_reg_loss_display_name}': reg_loss_cpu,
            'Loss/relation_geo': relation_geo_loss_cpu,
            'Loss/relation_gate': relation_gate_loss_cpu,
            'Loss/relation_semantic_region': relation_semantic_loss_cpu,
        }
        extra_metrics = {
            f'Loss/{self.cls_loss_display_name}_unscale': cls_loss_cpu / self.cls_loss_weight,
            f'Loss/{self.bbox_reg_loss_display_name}_unscale': reg_loss_cpu / self.bbox_reg_loss_weight,
            'Loss/main_total': main_loss_cpu,
            'Loss/relation_geo_unscale': relation_geo_loss.detach().cpu().item(),
            'Loss/relation_gate_unscale': relation_gate_loss.detach().cpu().item(),
            'Loss/relation_semantic_region_unscale': relation_semantic_loss.detach().cpu().item(),
            'Loss/relation_warmup_factor': relation_warmup_factor,
            'Loss/relation_semantic_warmup_factor': relation_semantic_warmup_factor,
            'Loss/relation_gate_scale': relation_gate_loss_scale,
        }

        return CriterionOutput(total_loss, metrics, extra_metrics)
