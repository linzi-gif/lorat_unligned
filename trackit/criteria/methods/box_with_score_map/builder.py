from torch import nn

from trackit.miscellanies.pretty_format import pretty_format


def build_box_with_score_map_criteria(criteria_config: dict):
    print('criteria config:\n' + pretty_format(criteria_config))
    classification_config = criteria_config['classification']
    if classification_config['type'] == 'binary_cross_entropy':
        cls_loss = nn.BCEWithLogitsLoss(reduction='none')
        cls_loss_name = 'bce'
    elif classification_config['type'] == 'varifocal':
        from ...modules.varifocal_loss import VarifocalLoss
        cls_loss = VarifocalLoss(alpha=classification_config['alpha'],
                                 gamma=classification_config['gamma'],
                                 iou_weighted=classification_config['iou_weighted'])
        cls_loss_name = 'varifocal'
    else:
        raise NotImplementedError(f"Classification type {classification_config['type']} is not implemented")
    iou_aware_classification_score = classification_config['iou_aware_classification_score']
    cls_loss_weight = classification_config['weight']

    bbox_regression_config = criteria_config['bbox_regression']
    if bbox_regression_config['type'] == 'iou':
        from ...modules.iou_loss import IoULoss
        bbox_reg_loss = IoULoss()
        bbox_reg_loss_name = 'iou'
    elif bbox_regression_config['type'] == 'GIoU':
        from ...modules.iou_loss import GIoULoss
        bbox_reg_loss = GIoULoss()
        bbox_reg_loss_name = 'iou'
    else:
        raise NotImplementedError(f"BBox regression type {bbox_regression_config['type']} is not implemented")
    bbox_reg_loss_weight = bbox_regression_config['weight']
    relation_auxiliary_config = criteria_config.get('relation_auxiliary', {})

    from . import SimpleCriteria
    return SimpleCriteria(cls_loss, bbox_reg_loss, iou_aware_classification_score,
                          cls_loss_weight, bbox_reg_loss_weight, cls_loss_name, bbox_reg_loss_name,
                          relation_geo_loss_weight=relation_auxiliary_config.get('geo_weight', 0.15),
                          relation_gate_loss_weight=relation_auxiliary_config.get('gate_weight', 0.03),
                          relation_semantic_region_loss_weight=relation_auxiliary_config.get('semantic_region_weight', 0.02),
                          relation_aux_warmup_epochs=relation_auxiliary_config.get('warmup_epochs', 2),
                          relation_semantic_warmup_start_epoch=relation_auxiliary_config.get('semantic_region_warmup_start_epoch', 2),
                          relation_semantic_warmup_end_epoch=relation_auxiliary_config.get('semantic_region_warmup_end_epoch', 5),
                          relation_geo_negative_weight=relation_auxiliary_config.get('geo_negative_weight', 0.25),
                          relation_semantic_negative_weight=relation_auxiliary_config.get('semantic_region_negative_weight', 0.25),
                          relation_gate_reduction_factor_on_main_loss_spike=relation_auxiliary_config.get('gate_reduction_factor_on_main_loss_spike', 0.5),
                          relation_main_loss_spike_ratio=relation_auxiliary_config.get('main_loss_spike_ratio', 1.15),
                          relation_main_loss_ema_momentum=relation_auxiliary_config.get('main_loss_ema_momentum', 0.9))
