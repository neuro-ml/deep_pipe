from .enet import ENet2D
from .deepmedic_orig import DeepMedic

model_name2model = {
    'deepmedic': DeepMedic,
    'enet2d': ENet2D,
}
