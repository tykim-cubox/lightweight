_base_ = [
    '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/mmdetection/configs/_base_/schedules/schedule_1x.py', 
    '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/mmdetection/configs/_base_/default_runtime.py'
]

# please install mmcls>=0.20.0
# import mmcls.models to trigger register_module in mmcls
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmcls.TIMMBackbone', # Using timm from mmcls
        model_name='efficientnet_b1',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3, 4)), # Modify out_indices
    neck=dict(in_channels=[24, 40, 112, 320])) # Modify in_channels

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)