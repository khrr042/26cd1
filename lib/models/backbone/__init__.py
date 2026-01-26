from .resnet import resnet50, resnet152
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a, se_resnet101_ibn_a
from .resnext_ibn_a import resnext50_ibn_a, resnext101_ibn_a
from .mixstyle import MixStyle, MixStyle2
from .STNModule import SpatialTransformer
# from .nfnet import dm_nfnet_f0

factory = {
    'resnet50': resnet50,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnext101_ibn_a': resnext101_ibn_a,
    'resnet152': resnet152,
}
def build_backbone(name, *args, **kwargs):
    if name not in factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return factory[name](*args, **kwargs)