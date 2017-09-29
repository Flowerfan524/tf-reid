
FEATURE_MAP = {
                'vgg_16': 'fc7_end_points',
                'inception_v3':'InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0',
                'inception_v4': 'PreLogitsFlatten',
                'inception_resnet_v2': 'PreLogitsFlatten',
                'resnet_v1_50': 'resnet_v1_50/pool5:0',
                'resnet_v2_50': 'pool5_end_points',
                'mobilenet_v1': 'AvgPool_1a',
               }


def get_last_feature_name(model_name):
    if model_name not in FEATURE_MAP:
        raise ValueError('wrong model_name input')
    return FEATURE_MAP[model_name]
