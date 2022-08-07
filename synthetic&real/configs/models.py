'''
Date: 2021-04-11 11:05:25
LastEditors: Please set LastEditors
LastEditTime: 2021-11-02 17:59:08
FilePath: /new_OverlapPredator_kl/configs/models.py
'''
architectures = dict()
architectures['indoor'] = [
    'simple',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'last_unary'
]

architectures['kitti'] = [
    'simple',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'last_unary'
]

architectures['modelnet'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]

architectures['multi'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
architectures['multi_multi'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
architectures['box'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
architectures['semantic'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
architectures['ipa'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
# architectures['box'] = [
#     'simple',
#     'resnetb',
#     'resnetb_strided',
#     'resnetb',
#     'resnetb',
#     'resnetb_strided',
#     'resnetb',
#     'resnetb',
#     'resnetb_strided',
#     'resnetb',
#     'resnetb',
#     'nearest_upsample',
#     'unary',
#     'nearest_upsample',
#     'unary',
#     'nearest_upsample',
#     'last_unary'
# ]
architectures['bigger_box'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
architectures['real'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
architectures['split'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
architectures['rgbd'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]
# architectures['multi'] = [
#     'simple',
#     'resnetb',
#     'resnetb',
#     'resnetb_strided',
#     'resnetb',
#     'resnetb',
#     #'resnetb_strided',
#     #'resnetb',
#     'resnetb',
#     'nearest_upsample',
#     'unary',
#     #'unary',
#     #'nearest_upsample',
#     'unary',
#     'last_unary'
# ]