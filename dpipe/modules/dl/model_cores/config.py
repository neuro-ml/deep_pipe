from .enet import ENet2D
from .deepmedic_orig import DeepMedic

model_core_name2model_core = {
    'deepmedic': DeepMedic,
    'enet2d': ENet2D,
}
