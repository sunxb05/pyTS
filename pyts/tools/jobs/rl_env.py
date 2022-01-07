import math
# from gym.utils import seeding
import numpy as np

from tensorflow.keras.models import load_model
from ..callbacks import CartesianMetrics
from tensorflow.keras.models import Model

class Env(object):

    # Set this in SOME subclasses
    metadata = {'render.modes': []}
    reward_range = (-float('inf'), float('inf'))
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        pass

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

    def __exit__(self, *args):
        """Support with-statement for the environment. """
        self.close()
        # propagate exception
        return False

class ChemEnv(Env):

    def __init__(self):
        self.seed()
        self.state = None
        self.steps_beyond_done  = None

        path = self.exp_config["run_config"]["model_path"]
        print(f"Loading pre-trained model from file {path}")
        self.model = load_model(path)


    def _compute_metrics(self, data):
        z = data[0][0]
        y_true = data[1][0]
        y_pred = self.model.predict(data[0])
        return self.structure_loss(z, y_pred, y_true)


    def _get_prediction(self, x):
        if self._output_type == "distance_matrix":
            model = Model(
                inputs=self.model.input,
                outputs=self.model.get_layer("cartesians").output,
            )
        else:
            model = self.model
        return model.predict(x)

    def _unwrap_data_lazily(self, data: list):
        """
        :param data: data in format (x, y) where x is [atomic_nums, reactants, products] and y is
            [ts_cartesians].
        :return: i, z, r, p, ts_true, ts_pred
        """
        predicted_transition_states = self._get_prediction(data[0])
        ((atomic_nums, reactants, products), (true_transition_states,),) = data
        for i, structures in enumerate(
            zip(
                atomic_nums[: self.max_structures],
                reactants[: self.max_structures],
                products[: self.max_structures],
                true_transition_states[: self.max_structures],
                predicted_transition_states[: self.max_structures],
            )
        ):
            output = [np.expand_dims(a, 0) for a in structures]
            output.insert(0, i)
            yield output


    @staticmethod
    def structure_loss(z, y_pred, y_true):
        d = MaskedDistanceMatrix()
        one_hot = OneHot(np.max(z) + 1)(z)
        dist_matrix = np.abs(d([one_hot, y_pred]) - d([one_hot, y_true]))
        dist_matrix = np.triu(dist_matrix)
        return (
            float(np.mean(dist_matrix[dist_matrix != 0])),
            float(np.mean(np.sum(np.sum(dist_matrix, axis=-1), axis=-1), axis=0)),
        )

    def write_cartesians(
        self, data: list, path: Path, write_static_structures: bool = False
    ):
        for i, z, r, p, true, pred in self._unwrap_data_lazily(data):
            # Make .xyz message lines
            arrays = (
                {"reactant": r, "product": p, "true": true, "predicted": pred}
                if write_static_structures
                else {"predicted": pred}
            )
            for name, array in arrays.items():
                loss = self.loss(array, true)
                error = self.structure_loss(z, array, true)[0] if name != "true" else 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, step_name):

        bondChange = self.bondChangerate if action==1 else -self.bondChangerate
        self.bondlength +=  bondChange
        self.state = self.model.predict(self.bondlength)
        bondlength_r1, bondlength_r2, energy_r1, energy_r2 = self.state

        done =  bondlength_r1 < self.bondlength_threshold

        done = bool(done)
        self.previous_energy_r1 = energy_r1
        if not done:
            reward = 1.0

        elif self.steps_beyond_done is None:
            # Pole just fell!
            steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                print ("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.bondlength  =  bondlengthR1
        self.previous_energy_r1 = 0
        self.state = self.np_random.uniform(low=0.00, high=0.10, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)
