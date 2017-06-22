class Model():
    def __init__(self, n_chans_in: int, n_chans_out: int):
        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out

        self.x_phs = self.training_ph = self.y_ph = None
        self.loss = self.y_pred = None
