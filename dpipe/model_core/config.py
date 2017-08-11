from .unet import UNet2D, UResNet2D
from .deepmedic_bottle import DeepMedicBottle
from .deepmedic_highway import DeepMedicHigh
from .deepmedic_orig import DeepMedicOrig
from .deepmedic_res import DeepMedicRes
from .enet import ENet2D

module_builders = {
    'deepmedic_orig': DeepMedicOrig,
    'deepmedic_highway': DeepMedicHigh,
    'deepmedic_res': DeepMedicRes,
    'deepmedic_bottle': DeepMedicBottle,
    'enet2d': ENet2D,
    'unet2d': UNet2D,
    'uresnet2d': UResNet2D,
}
