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

    def _load_fitable(self, loader: DataLoader, fitable_config: dict = None) -> Model:
        """
        Defines and compiles a fitable (keras.model or keras_tuner.tuner) which implements
        a 'fit' method. This method calls either get_builder, or get_hyper_factory, depending on
        which type of fitable is beind loaded.

        :return: Model or Tuner object.
        """
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
        if run_config["use_strategy"]:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model = builder.get_model()
                model.compile(**compile_kwargs)
                model.summary()
                plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
        else:
            model = builder.get_model()
            model.compile(**compile_kwargs)
            model.summary()
            plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
        return model

    def _fit(
        self, run: Run, fitable: Model, data: tuple, callbacks: list = None,
    ) -> Model:
        """

        :param run: sacred.Run object. See sacred documentation for details on utility.
        :param fitable: tensorflow.keras.Model object.
        :param data: tuple. train, validation, and test data in the form (train, val, test),
        where train is
            the tuple (x_train, y_train).
        :param callbacks: Optional list. List of tensorflow.keras.Callback objects to pass to
            fitable.fit method.
        :return: tensorflow.keras.Model object.
        """
        tensorboard_directory = self.exp_config["run_config"]["root_dir"] / "logs"
        (x_train, y_train), val, _ = data
        callbacks = callbacks or []
        if self.exp_config["run_config"]["use_default_callbacks"]:
            callbacks.extend(
                [
                    TensorBoard(
                        **dict(
                            **self.exp_config["tb_config"],
                            log_dir=tensorboard_directory,
                        )
                    ),
                    ReduceLROnPlateau(**self.exp_config["lr_config"]),
                ]
            )
        kwargs = dict(
            x=x_train,
            y=y_train,
            epochs=self.exp_config["run_config"]["epochs"],
            batch_size=self.exp_config["run_config"]["batch_size"],
            validation_data=val,
            class_weight=self.exp_config["run_config"]["class_weight"],
            callbacks=callbacks,
            verbose=self.exp_config["run_config"]["fit_verbosity"],
        )
        fitable.fit(**kwargs)

        return fitable

    def _save_fitable(self, run: Run, fitable: Model):
        """
        :param run: sacred.Run object. see sacred documentation for more details on utility.
        :param fitable: tensorflow.keras.Model object.
        """
        path = self.exp_config["run_config"]["model_path"]
        if self.exp_config["run_config"]["save_verbosity"] > 0:
            fitable.summary()
        fitable.save(self.exp_config["run_config"]["model_path"])
        run.add_artifact(path)