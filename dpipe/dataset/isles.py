import functools

from .segmentation import FromCSVMultiple

Isles2017 = functools.partial(
    FromCSVMultiple,
    modalities=['ADC', 'MTT', 'TTP', 'Tmax', 'rCBF', 'rCBV'],
    targets=['OT'],
    metadata_rpath='meta2017.csv'
)
