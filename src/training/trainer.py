class Trainer:
    """
    Placeholder trainer class.
    Actual training will be implemented in Colab.
    """

    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, fcgr_orig, fcgr_mimic, weights=None):
        p = self.model(fcgr_orig)
        q = self.model(fcgr_mimic)

        loss = self.loss_fn(p, q, weights)

        return loss
