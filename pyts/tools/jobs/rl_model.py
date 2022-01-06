import numpy as np
import os
import random
import shutil
from statistics import mean
import datetime
from .rl_logger import Logger

from ..builder.rl_cartesian_builder import CartesianBuilder
from .rl_keras_job import KerasJob


class BaseModel(KerasJob):

    def __init__(self, game_name, mode_name, logger_path, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.logger = Logger(game_name + " " + mode_name, logger_path)

    def save_run(self, score, step, run):
        self.logger.add_score(score)
        self.logger.add_step(step)
        self.logger.add_run(run)

    def get_move(self, state):
        pass

    def act(self, state):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def step_update(self, total_step):
        pass

    def _get_date(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))


class RlModel(BaseModel):

    def __init__(self, game_name, mode_name, observation_space, action_space, logger_path, model_path):
        BaseModel.__init__(self, game_name,
                               mode_name,
                               logger_path,
                               observation_space,
                               action_space)
        self.model_path = model_path
        self.rl = self._load_fitable(loader, fitable_config)
        if os.path.isfile(self.model_path):
            self.rl.load_weights(self.model_path)

    def _save_model(self):
        self.rl.save_weights(self.model_path)


class rlSolver(RlModel):

    def __init__(self, game_name, observation_space, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/testing/model.h5"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        RlModel.__init__(self,
                               game_name,
                               "rl testing",
                               observation_space,
                               action_space,
                               "./output/logs/" + game_name + "/testing/" + self._get_date() + "/",
                               testing_model_path)

    def act(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        q_values = self.rl.predict(state)
        return np.argmax(q_values[0])


class rlTrainer(RlModel):

    def __init__(self, game_name, observation_space, action_space):
        RlModel.__init__(self,
                               game_name,
                               "rl training",
                               observation_space,
                               action_space,
                               "./output/logs/" + game_name + "/training/" + self._get_date() + "/",
                               "./output/neural_nets/" + game_name + "/" + self._get_date() + "/model.h5")

        if os.path.exists(os.path.dirname(self.model_path)):
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

        self.rl_target = self._load_fitable(loader, fitable_config)
        self._reset_target_network()
        self.exploration_rate = EXPLORATION_MAX
        self.memory = []

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
        #or len(self.memory) < REPLAY_START_SIZE
            return random.randrange(self.action_space)
        q_values = self.rl.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def step_update(self, total_step):

        if total_step % TRAINING_FREQUENCY == 0:
            # loss, accuracy, average_max_q = self.experience_replay()
            loss, average_max_q = self.experience_replay()
            self.logger.add_loss(loss)
            # self.logger.add_accuracy(accuracy)
            self.logger.add_q(average_max_q)

        if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            self._save_model()

        if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self._reset_target_network()
            print('{{"metric": "total_step", "value": {}}}'.format(total_step))

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            # reward = -1
            q_update = reward
            # reward = 1
            # np.amax  The maximum value along a given axis.
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.rl_target.predict(state_next)[0]))
            q_values = self.rl.predict(state)
            q_values[0][action] = q_update

            # Preload weights if necessary
            if fitable is not None:
                fitable.save_weights("./temp_weights.hdf5")
                model.load_weights("./temp_weights.hdf5")

            model = self._fit(
                run,
                self.rl,
                data,
                callbacks=[
                    CartesianMetrics(
                        self.exp_config["run_config"]["root_dir"] / "cartesians",
                        *data,
                        **self.exp_config["cm_config"],
                    )
                ],
            )

            fit = self.rl.fit(state, q_values, batch_size=BATCH_SIZE, verbose=0)
            loss = fit.history["loss"][0]
            # accuracy = fit.history["accuracy"][0]
            # return loss, accuracy, q_update
            return loss,  q_update
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def _reset_target_network(self):
        self.rl_target.set_weights(self.rl.get_weights())
