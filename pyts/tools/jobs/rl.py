import argparse
import numpy as np
import random
from ..rl_env import ChemEnv
from ..rl_builder import ConvolutionalNeuralNetwork

from copy import copy

from sacred.run import Run
from tensorflow.keras.models import Model

from . import KerasJob, config_defaults
from ..callbacks import CartesianMetrics


class rl(KerasJob):
    @property
    def config_defaults(self):
        base = super().config_defaults
        base["loader_config"][
            "map_points"
        ] = False  # Ensure reconstruction works properly
        base["cm_config"] = copy(config_defaults.cm_config)
        return base

    def _main(
        self,
        env: ChemEnv, 
        rl_config: dict = None,
        run: Run,
        fitable: Model = None,
        fitable_config: dict = None,
        loader_config: dict = None,
    ):
        run = 0
        total_step = 0
        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print ("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1
            state = env.reset()
            state = np.reshape(state, [1, self.observation_space])
            step = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print ("Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1
                step += 1
                action = game_model.act(state)
                step_name = str(run)+"_"+str(step)
                state_next, reward, terminal, info = env.step(action, step_name)
                score += reward
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, self.observation_space])
                game_model.remember(state, action, reward, state_next, terminal)
                state = state_next
                game_model.step_update(total_step)
                if terminal:
                    game_model.save_run(score, step, run)
                    break

        return model