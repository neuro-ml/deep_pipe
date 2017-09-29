import numpy as np

# FIXME is this code dead? Remove it all?
# decorator for decorators. yeah, baby
def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


@parametrized
def apply(cls, func, **kwargs):
    class Wrapped(cls):
        def load_mscan(self, patient_id):
            image = super().load_mscan(patient_id)
            return func(image, **kwargs)

        def load_msegm(self, patient):
            image = super().load_msegm(patient)
            image = func(image, **kwargs)

            return image >= .5

    return Wrapped


@parametrized
def mscan(cls, func, **kwargs):
    class Wrapped(cls):
        def load_mscan(self, patient_id):
            image = super().load_mscan(patient_id)
            return func(image, **kwargs)

    return Wrapped


@parametrized
def msegm(cls, func, **kwargs):
    class Wrapped(cls):
        def load_msegm(self, patient):
            image = super().load_msegm(patient)
            image = func(image, **kwargs)

            return image >= .5

    return Wrapped


def append_channels(cls):
    class Wrapped(cls):
        def __init__(self, *args, append_paths, **kwargs):
            super(Wrapped, self).__init__(*args, **kwargs)
            self.append_paths = append_paths

        def load_mscan(self, patient_id):
            image = super().load_mscan(patient_id)

            additional = [i % patient_id for i in self.append_paths]
            second = self.load_by_paths(additional)
            if second.ndim != image.ndim:
                second = second[0]
            image = np.vstack((image, second))
            image = image.astype('float32')

            return image

        @property
        def n_chans_mscan(self):
            return len(self.append_paths) + super().n_chans_mscan

    return Wrapped
