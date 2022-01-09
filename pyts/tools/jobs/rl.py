import argparse
import numpy as np
import random
from copy import copy
from sacred.run import Run
from tensorflow.keras.models import Model
from .rl_env import ChemEnv
from . import  config_defaults
from .rl_keras_job import KerasJob

import numpy as np
import os
import random
import shutil
from statistics import mean
import datetime
from .rl_logger import Logger


class rl(KerasJob):

    def _main(
        self,
        env: ChemEnv,
        logger: Logger
        run: Run,
        seed: int,
        fitable: Model = None,
        fitable_config: dict = None,
        loader_config: dict = None,
    ):

        run_step = 0
        total_step = 0
        ml_number = 0
        memory = []

        rl_config = self.exp_config["rl_config"]
        total_run_limit = rl_config["total_run_limit"]
        # total_step_limit = rl_config["total_step_limit"]
        total_step_limit = 2

        loader, data = self._load_data(loader_config)
        fitable = fitable or self._load_fitable(loader, fitable_config)
        env = ChemEnv(data)
        logger = Logger("rl", logger_path)

        while True:
            if total_run_limit is not None and run_step >= total_run_limit:
                print ("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            state = env.reset(ml_number)
            ml_number+= 1
            run_step += 1
            step  = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print ("Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1
                step += 1
                action = self.act(fitable, state)
                state_next, reward, terminal, info = env.step(action)
                score += reward
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, self.observation_space])
                self.remember(memory, state, action, reward, state_next, terminal)
                state = state_next
                self.step_update(logger, total_step)
                if terminal:
                    self._save_fitable(score, step, run_step)
                    break
