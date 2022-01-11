from pathlib import Path
from typing import Tuple

from sacred.run import Run
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

from .job import Job
from ..ingredients import (
    get_data_loader,
    get_builder,
)
from ..loaders import DataLoader


class KerasJob(Job):

    def _main(
        self,
        run: Run,
        seed: int,
        fitable: Model = None,
        fitable_config: dict = None,
        loader_config: dict = None,
    ):

        loader, data = self._load_data(loader_config)
        fitable = fitable or self._load_fitable(loader, fitable_config)
        fitable = self._fit(run, fitable, data)
        if self.exp_config["run_config"]["test"]:
            self._test_fitable(run, fitable, data[-1])
        if self.exp_config["run_config"]["save_model"]:
            self._save_fitable(run, fitable)
        return fitable

    def _load_data(self, config: dict = None) -> Tuple[DataLoader, Tuple]:

        config = config or self.exp_config["loader_config"]
        loader = get_data_loader(**config)
        data = loader.load_data(**config["load_kwargs"])
        return loader, data

    def _load_fitable(self, loader: DataLoader, fitable_config: dict = None) -> Model:

        fitable_config = fitable_config or self.exp_config["builder_config"]
        conf = dict(
            **fitable_config,
            max_z=loader.max_z,
            num_points=loader.num_points,
            mu=loader.mu,
            sigma=loader.sigma,
        )
        builder = get_builder(**conf)
        run_config = self.exp_config["run_config"]
        compile_kwargs = dict(
            loss=run_config["loss"],
            loss_weights=run_config["loss_weights"],
            optimizer=run_config["optimizer"],
            metrics=run_config["metrics"],
            run_eagerly=run_config["run_eagerly"],
        )

        model = builder.get_model()
        model.compile(**compile_kwargs)
        # model.summary()
        # plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
        return model

    def _fit(
        self, run: Run, fitable: Model, data: tuple,
    ) -> Model:

        tensorboard_directory = self.exp_config["run_config"]["root_dir"] / "logs"
        x_train, y_train = data[0]

        kwargs = dict(
            x=x_train[0],
            y=y_train,
            epochs=self.exp_config["run_config"]["epochs"],
            batch_size=self.exp_config["run_config"]["batch_size"],
            class_weight=self.exp_config["run_config"]["class_weight"],
            verbose=self.exp_config["run_config"]["fit_verbosity"],
        )
        fitable.fit(**kwargs)

        return fitable

    def _test_fitable(self, run: Run, fitable: Model, test_data: list) -> float:

        if test_data is None:
            return 0.0
        x_test, y_test = test_data
        loss = fitable.evaluate(x=x_test, y=y_test, verbose=0)
        print(f"Test split results: {loss}")
        return loss

    def _save_fitable(self, run: Run, fitable: Model):

        path = self.exp_config["run_config"]["model_path"]
        if self.exp_config["run_config"]["save_verbosity"] > 0:
            fitable.summary()
        fitable.save(self.exp_config["run_config"]["model_path"])
        run.add_artifact(path)


    def _save_model(self, fitable: Model):
        path = self.exp_config["run_config"]["model_path"]
        fitable.save_weights(path)


    def _new_model_path(self, name: str):
        model_path = Path(self.exp_config["run_config"]["model_path"]).parent / name
        self.exp_config["run_config"]["model_path"] = model_path
        return model_path

    def act(self, fitable, state):
        q_values = fitable.predict(state)
        return np.argmax(q_values[0])

    def remember(self, memory, state, action, reward, next_state, terminal):
        memory.append((state, action, reward, next_state, terminal))
        if len(memory) > MEMORY_SIZE:
            memory.pop(0)

    def step_update(self, logger, total_step):
        if total_step % TRAINING_FREQUENCY == 0:
            loss, average_max_q = self._experience_replay()
            logger.add_loss(loss)
            logger.add_q(average_max_q)

        if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            self._save_fitable(run, fitable)

    def _experience_replay(self, run, fitable):
        if len(memory) < BATCH_SIZE:
            return
        batch = random.sample(memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(fitable.predict(state_next)[0]))
            q_values = fitable.predict(state)
            q_values[0][action] = q_update

            # Preload weights if necessary
            if fitable is not None:
                fitable.save_weights("./temp_weights.hdf5")
                fitable.load_weights("./temp_weights.hdf5")

            model = self._fit(
                run,
                fitable,
                [state, q_values]
            )
            loss = model.history["loss"][0]

            return loss,  q_update
