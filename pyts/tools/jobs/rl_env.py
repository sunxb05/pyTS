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
        path = "/Users/xiaobo/pyTS/others/run/0104/model.hdf5"
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

        return trj[int(closest_frame)]


    def step(self, action):

        [x, y] =  self.state
        z = x[0]
        r = x[1]
        p = x[2]
        t = x[3]
        y_true = y[0]

        action_pair = [[0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4], [0.1, 0.5], [0.1, 0.6], [0.1, 0.7], [0.1, 0.8], [0.1, 0.9], [0.1, 1.0], [0.2, 0.1], [0.2, 0.2], [0.2, 0.3], [0.2, 0.4], [0.2, 0.5], [0.2, 0.6], [0.2, 0.7], [0.2, 0.8], [0.2, 0.9], [0.2, 1.0], [0.3, 0.1], [0.3, 0.2], [0.3, 0.3], [0.3, 0.4], [0.3, 0.5], [0.3, 0.6], [0.3, 0.7], [0.3, 0.8], [0.3, 0.9], [0.3, 1.0], [0.4, 0.1], [0.4, 0.2], [0.4, 0.3], [0.4, 0.4], [0.4, 0.5], [0.4, 0.6], [0.4, 0.7], [0.4, 0.8], [0.4, 0.9], [0.4, 1.0], [0.5, 0.1], [0.5, 0.2], [0.5, 0.3], [0.5, 0.4], [0.5, 0.5], [0.5, 0.6], [0.5, 0.7], [0.5, 0.8], [0.5, 0.9], [0.5, 1.0], [0.6, 0.1], [0.6, 0.2], [0.6, 0.3], [0.6, 0.4], [0.6, 0.5], [0.6, 0.6], [0.6, 0.7], [0.6, 0.8], [0.6, 0.9], [0.6, 1.0], [0.7, 0.1], [0.7, 0.2], [0.7, 0.3], [0.7, 0.4], [0.7, 0.5], [0.7, 0.6], [0.7, 0.7], [0.7, 0.8], [0.7, 0.9], [0.7, 1.0], [0.8, 0.1], [0.8, 0.2], [0.8, 0.3], [0.8, 0.4], [0.8, 0.5], [0.8, 0.6], [0.8, 0.7], [0.8, 0.8], [0.8, 0.9], [0.8, 1.0], [0.9, 0.1], [0.9, 0.2], [0.9, 0.3], [0.9, 0.4], [0.9, 0.5], [0.9, 0.6], [0.9, 0.7], [0.9, 0.8], [0.9, 0.9], [0.9, 1.0], [1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4], [1.0, 0.5], [1.0, 0.6], [1.0, 0.7], [1.0, 0.8], [1.0, 0.9], [1.0, 1.0]]

        reactent = self._int_mol(r, t, action_pair[action][0])
        product  = self._int_mol(t, p, action_pair[action][1])
        print ([[z, reactent, product], [y_true]])
        print ('===========================================')
        ts_predicted = self.model.predict([[z, np.array(reactent), np.array(product)], [y_true]])
        print ("ts_predicted")
        print (ts_predicted)
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
        self.state = [[z[ml_number], r[ml_number], p[ml_number], ts_predicted[ml_number]], [y_true[ml_number]]]
        return self.state
