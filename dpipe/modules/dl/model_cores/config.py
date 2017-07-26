from .enet import ENet2D
from .deepmedic_orig import DeepMedicOrig
from .deepmedic_highway import DeepMedicHigh
from .deepmedic_res import DeepMedicRes
from .deepmedic_bottle import DeepMedicBottle

model_core_name2model_core = {
    'deepmedic_orig': DeepMedicOrig,
    'deepmedic_highway': DeepMedicHigh,
    'deepmedic_res': DeepMedicRes,
    'deepmedic_bottle': DeepMedicBottle,
    'enet2d': ENet2D,
}
