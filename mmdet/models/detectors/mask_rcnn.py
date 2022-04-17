from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class MaskRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 track_head,
                 prop_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 neck_attn=None,
                 shared_head=None,
                 pretrained=None):
        print("YOLO")
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            neck_attn=neck_attn,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            track_head=track_head,
            prop_head=prop_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
