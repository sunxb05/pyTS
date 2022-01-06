import argparse
import numpy as np
import random
from copy import copy
from sacred.run import Run
from .rl_job import Job
from .rl_model import RlModel, rlSolver, rlTrainer
from .rl_env import ChemEnv




class rl(Job):
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
        run: Run,
        rl_model: RlModel
    ):
        run_step = 0
        total_step = 0
        rl_config = self.exp_config["rl_config"]
        total_run_limit = rl_config["total_run_limit"]
        total_step_limit = rl_config["total_step_limit"]
        if rl_config["total_run_limit"] == "training":
            rl_model = rlTrainer
        else:
            rl_model = rlSolver
        while True:
            if total_run_limit is not None and run_step >= total_run_limit:
                print ("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run_step += 1
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
                action = rl_model.act(state)
                step_name = str(run_step)+"_"+str(step)
                state_next, reward, terminal, info = env.step(action, step_name)
                score += reward
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, self.observation_space])
                rl_model.remember(state, action, reward, state_next, terminal)
                state = state_next
                rl_model.step_update(total_step)
                if terminal:
                    rl_model.save_run(score, step, run_step)
                    break

        return model
