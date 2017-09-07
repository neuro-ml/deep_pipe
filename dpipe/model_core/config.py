from .enet_mixed import ENetMixed, ENetPatch
from .unet import UNet2D, UResNet2D
from .deepmedic_orig import DeepMedicOrig
from .deepmedic_els import DeepMedicEls
from .enet import ENet2D

name2model_core = {
    'deepmedic_orig': DeepMedicOrig,
    'deepmedic_els': DeepMedicEls,
    'enet2d': ENet2D,
    'enet_mixed': ENetMixed,
    'enet_patch': ENetPatch,
    'unet2d': UNet2D,
    'uresnet2d': UResNet2D,
}
