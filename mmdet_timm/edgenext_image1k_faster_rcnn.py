_base_ = [
    '/home/aiteam/tykim/scratch/lightweight/mmcv_phone/tutorial_exps/faster_rcnn_res50_phone.py'
]

custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
model = dict(
    backbone=dict(
        _delete_=True, # Delete the backbone field in _base_
        type='mmcls.TIMMBackbone', # Using timm from mmcls
        model_name='edgenext_small',
        features_only=True,
        pretrained=True,
        out_indices=(0, 1, 2, 3)), # Modify out_indices
    neck=dict(in_channels=[48, 96, 160, 304])) # Modify in_channels