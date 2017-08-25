from .enet_mixed import ENetMixed, ENetPatch
from .unet import UNet2D, UResNet2D
from .deepmedic_bottle import DeepMedicBottle
from .deepmedic_highway import DeepMedicHigh
from .deepmedic_orig import DeepMedicOrig
from .deepmedic_res import DeepMedicRes
from .enet import ENet2D

name2model_core = {
    'deepmedic_orig': DeepMedicOrig,
    'deepmedic_highway': DeepMedicHigh,
    'deepmedic_res': DeepMedicRes,
    'deepmedic_bottle': DeepMedicBottle,
    'enet2d': ENet2D,
    'enet_mixed': ENetMixed,
    'enet_patch': ENetPatch,
    'unet2d': UNet2D,
    'uresnet2d': UResNet2D,
}
