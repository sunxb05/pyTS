import math
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from ...layers import MaskedDistanceMatrix, OneHot


class ChemEnv():

    def __init__(self, data):
        self.state = None
        self.ts_threshold = 0.1
        self.data = data
        # path = self.exp_config["run_config"]["model_path"]
        # print(f"Loading pre-trained model from file {path}")
        path = "/Users/xiaobo/pyTS/others/run/0104/sacred_storage/1/model.hdf5"
        self.model = load_model(path)

    def tfn_mae(self, y_pred, y_true):
        loss = tf.abs(y_pred - y_true)
        return tf.reduce_mean(loss[loss != 0])

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

    def _int_mol(self, mol_A, mol_B, coef):

        nint = 500
        trj = []
        diff = []
        for atom_A, atom_B in zip(mol_A, mol_B):
            diff.append([
                (atom_A[0] - atom_B[0]) / (nint + 1),
                (atom_A[1] - atom_B[1]) / (nint + 1),
                (atom_A[2] - atom_B[2]) / (nint + 1)
            ])

        for frame in range(1, nint + 1):
            mol_frame = []
            for atom, ato_diff in zip(mol_A, diff):
                mol_frame.append([
                    atom[0] - ato_diff[0] * frame,
                    atom[1] - ato_diff[1] * frame,
                    atom[2] - ato_diff[2] * frame
                ])
            trj.append(mol_frame)

        closest_frame = nint * coef

        return trj[closest_frame]


    def step(self, action):

        z = self.state[0][0]
        r = self.state[0][1]
        p = self.state[0][2]
        t = self.state[0][3]
        y_true = self.state[1][0]

        action_pair = [[0.1, 0.1], [0.1, 0.2], [0.1, 0.2], [0.1, 0.1], [0.1, 0.2], [0.1, 0.2], [0.1, 0.1], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]]
        reactent = self._int_mol(r, t, action_pair[action][0])
        product  = self._int_mol(t, p, action_pair[action][1])
        ts_predicted = self.model.predict([[z, reactent, product], [y_true]])
        reward = self._structure_loss(z, ts_predicted, y_true)

        done =  reward < self.ts_threshold
        done = bool(done)
        if done:
            reward = 1.0

        self.state = [[z, reactent, product, ts_predicted], [y_true]]

        return np.array(self.state), reward, done, {}

    def reset(self, ml_number):

        (x, y) =  self.data[0]
        z = x[0]
        r = x[1]
        p = x[2]
        y_true = y[0]
        ts_predicted = self.model.predict(self.data[0])
        self.state = [[z[ml_number], r[ml_number], p[ml_number], ts_predicted[ml_number]]], [y_true[ml_number]]

        return self.state
