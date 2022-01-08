import math
import numpy as np

from tensorflow.keras.models import load_model
from ..callbacks import CartesianMetrics
from tensorflow.keras.models import Model
from .rl_lin_int import IntMOL


class Env(object):

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def seed(self, seed=None):
        return

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

    def __enter__(self):
        """Support with-statement for the environment. """
        return self

class ChemEnv(Env):

    def __init__(self):
        self.state = None
        self.ts_threshold = 0.1
        path = self.exp_config["run_config"]["model_path"]
        print(f"Loading pre-trained model from file {path}")
        self.model = load_model(path)

    @staticmethod
    def _structure_loss(z, y_pred, y_true):
        d = MaskedDistanceMatrix()
        one_hot = OneHot(np.max(z) + 1)(z)
        dist_matrix = np.abs(d([one_hot, y_pred]) - d([one_hot, y_true]))
        dist_matrix = np.triu(dist_matrix)
        return (
            float(np.mean(dist_matrix[dist_matrix != 0])),
            float(np.mean(np.sum(np.sum(dist_matrix, axis=-1), axis=-1), axis=0)),
        )

    def _compute_error(self, data):
        z = data[0][0]
        y_true = data[1][0]
        y_pred = self.model.predict(data[0])
        return self._structure_loss(z, y_pred, y_true)


    def _new_geometries(self, data, coef):

        reactent = IntMOL(data[0], coef)
        product  = IntMOL(data[1], coef)

        return reactent, product


    def step(self, action):

        self.state = self._new_geometries(self, data, coef)

        reward = self._compute_error(self, data, coef)

        done =  reward < self.ts_threshold
        done = bool(done)

        if done:
            reward = 1.0

        return np.array(self.state), reward, done, {}

    def reset(self, data):

        return data.next()
