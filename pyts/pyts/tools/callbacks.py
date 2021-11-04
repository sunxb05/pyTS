import os
from pathlib import Path
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import get as get_loss
from tensorflow.keras import backend as K

from .converters import ndarrays_to_xyz
from ..layers import MaskedDistanceMatrix, OneHot


class TestModel(Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test

    def on_train_end(self, logs=None):
        predictions = self.model.evaluate(x=self.x_test, y=self.y_test)
        if not isinstance(predictions, list):
            predictions = [predictions]
        for pred, name in zip(predictions, self.model.metric_names):
            logs["test_{}".format(name)] = pred


class CartesianMetrics(Callback):
    def __init__(
        self,
        path: Union[str, Path],
        train: list = None,
        validation: list = None,
        test: list = None,
        max_structures: int = 64,
        write_rate: int = 50,
        tensorboard_logdir: str = None,
    ):
        """
        :param path: Base path to which .xyz files will be written to. Will create subdirectories
            at this path as needed.
        :param validation: Defaults to None. List of val data to write. Of shape [[z, r, p], [ts]].
            If None, will not write validation .xyz files.
        :param test: list. Defaults to None. Same as validation, but for test data.
        :param max_structures: int. Defaults to 10. Max number of structures to write .xyz files
            for.
        :param write_rate: int. Defaults to 50. Number of epochs to take before writing
            validation Cartesians (if validation data provided).
        """
        super().__init__()
        self.path = Path(path)
        self.train = train
        self.validation = validation
        self.test = test
        self.max_structures = max_structures
        self.write_rate = write_rate
        logdir = tensorboard_logdir or self.path.parent / "logs"
        self.file_writers = [
            tf.summary.create_file_writer(str(logdir / "train")),
            tf.summary.create_file_writer(str(logdir / "train")),
            tf.summary.create_file_writer(str(logdir / "validation")),
            tf.summary.create_file_writer(str(logdir / "validation")),
        ]

        self.total_epochs = -1
        self._prediction_type = "vectors"
        self._output_type = "cartesians"

    def get_vectors(self, inputs):
        model = Model(
            inputs=self.model.input, outputs=self.model.get_layer("vectors").output,
        )
        vectors = model.predict([np.expand_dims(a, axis=0) for a in inputs])
        return np.squeeze(vectors, axis=0)

    @staticmethod
    def write_vectors(vectors, path, i):
        vector_path = path / "vectors"
        os.makedirs(vector_path, exist_ok=True)
        np.savetxt(vector_path / f"{i}_vectors.txt", vectors)  # Write vectors .txt file

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
        predict_point2, predict_point3, predict_point4, predict_point5, predict_point6, predict_point7, predict_point8, predict_point9, predict_point10, predict_point11 = self._get_prediction(data[0])
        ((atomic_nums, point1), (point2, point3, point4, point5, point6, point7, point8, point9, point10, point11),) = data
        for i, structures in enumerate(
            zip(
                atomic_nums[: self.max_structures],
                point1[: self.max_structures],
                point2[: self.max_structures],
                point3[: self.max_structures],
                point4[: self.max_structures],
                point5[: self.max_structures],
                point6[: self.max_structures],
                point7[: self.max_structures],
                point8[: self.max_structures],
                point9[: self.max_structures],
                point10[: self.max_structures],
                point11[: self.max_structures],

                predict_point2[: self.max_structures],
                predict_point3[: self.max_structures],
                predict_point4[: self.max_structures],
                predict_point5[: self.max_structures],
                predict_point6[: self.max_structures],
                predict_point7[: self.max_structures],
                predict_point8[: self.max_structures],
                predict_point9[: self.max_structures],
                predict_point10[: self.max_structures],
                predict_point11[: self.max_structures],
            )
        ):
            output = [np.expand_dims(a, 0) for a in structures]
            output.insert(0, i)
            yield output

    def loss(self, a, b):
        if a.shape != b.shape:
            return 0
        else:
            return K.mean(get_loss(self.model.loss)(a, b))

    @staticmethod
    def structure_loss(z, y_pred, y_true):
        d = MaskedDistanceMatrix()
        one_hot = OneHot(np.max(z) + 1)(z)
        dist_matrix = K.abs(d([one_hot, y_pred]) - d([one_hot, y_true]))
        dist_matrix = K.triu(dist_matrix)
        return (
            float(K.mean(dist_matrix[dist_matrix != 0])),
            float(np.mean(np.sum(np.sum(dist_matrix, axis=-1), axis=-1), axis=0)),
        )

    def write_cartesians(
        self, data: list, path: Path, write_static_structures: bool = False
    ):
        for i, z, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p2_, p3_ , p4_, p5_, p6_, p7_, p8_, p9_, p10_, p11_ in self._unwrap_data_lazily(data):
            # Make .xyz message lines
            arrays = (
                {"point1": p1, "point2": p2, "point3": p3, "point4": p4, "point5": p5, "point6": p6, "point7": p7, "point8": p8, "point9": p9, "point10": p10, "point11": p11,"predicted_p2": p2_, "predicted_p3": p3_, "predicted_p4": p4_, "predicted_p5": p5_,"predicted_p6": p6_, "predicted_p7": p7_, "predicted_p8": p8_, "predicted_p9": p9_, "predicted_p10": p10_,"predicted_p11": p11_}
            )

            loss_p2 = self.loss(arrays["point2"], arrays["predicted_p2"])
            loss_p3 = self.loss(arrays["point3"], arrays["predicted_p3"])
            loss_p4 = self.loss(arrays["point4"], arrays["predicted_p4"])
            loss_p5 = self.loss(arrays["point5"], arrays["predicted_p5"])
            loss_p6 = self.loss(arrays["point6"], arrays["predicted_p6"])
            loss_p7 = self.loss(arrays["point7"], arrays["predicted_p7"])
            loss_p8 = self.loss(arrays["point8"], arrays["predicted_p8"])
            loss_p9 = self.loss(arrays["point9"], arrays["predicted_p9"])
            loss_p10 = self.loss(arrays["point10"], arrays["predicted_p10"])
            loss_p11 = self.loss(arrays["point11"], arrays["predicted_p11"])
            # error_p2 = self.structure_loss(z, array["point2"], array["predicted_p2"])[0]
            # error_p3 = self.structure_loss(z, array["point3"], array["predicted_p3"])[0]
            # message_p2 = f"loss: {loss_p2} " f"-- distance_error: {error_p2} "
            # message_p3 = f"loss: {loss_p3} " f"-- distance_error: {error_p3} "

            ndarrays_to_xyz(arrays["predicted_p2"][0], z[0], path / f"{i}_predicted_p2.xyz")
            ndarrays_to_xyz(arrays["predicted_p3"][0], z[0], path / f"{i}_predicted_p3.xyz")
            ndarrays_to_xyz(arrays["predicted_p4"][0], z[0], path / f"{i}_predicted_p4.xyz")
            ndarrays_to_xyz(arrays["predicted_p5"][0], z[0], path / f"{i}_predicted_p5.xyz")
            ndarrays_to_xyz(arrays["predicted_p6"][0], z[0], path / f"{i}_predicted_p6.xyz")
            ndarrays_to_xyz(arrays["predicted_p7"][0], z[0], path / f"{i}_predicted_p7.xyz")
            ndarrays_to_xyz(arrays["predicted_p8"][0], z[0], path / f"{i}_predicted_p8.xyz")
            ndarrays_to_xyz(arrays["predicted_p9"][0], z[0], path / f"{i}_predicted_p9.xyz")
            ndarrays_to_xyz(arrays["predicted_p10"][0], z[0], path / f"{i}_predicted_p10.xyz")
            ndarrays_to_xyz(arrays["predicted_p11"][0], z[0], path / f"{i}_predicted_p11.xyz")
            # Add vector information if relevant
            # if self._prediction_type == "vectors":
            #     # Write vectors
            #     vectors = self.get_vectors([z, r, p])
            #     self.write_vectors(vectors, path, i)

    def compute_metrics(self, epoch, split: str = "train"):
        if split == "train":
            data = self.train
            file_writers = self.file_writers[:2]
        else:
            data = self.validation
            file_writers = self.file_writers[2:]
        self.write_metrics(
            {"distance_error": self._compute_metrics(data)[0]},
            epoch,
            file_writers,
            split,
        )

    def _compute_metrics(self, data):
        z = data[0][0]
        y_true = data[1][0]
        y_pred = self.model.predict(data[0])
        return self.structure_loss(z, y_pred, y_true)

    def _writing(self, epoch):
        return (
            False
            if self.write_rate <= 0
            else (epoch in range(25) or (epoch + 1) % self.write_rate == 0)
        )

    def write_metrics(
        self,
        metrics: dict,
        epoch: int,
        file_writers: list = None,
        prefix: str = "scalar",
    ):
        file_writers = file_writers or self.file_writers
        print(
            " -- ".join(
                [
                    f"{prefix}_{name}: {round(metric, 8)}"
                    for name, metric in metrics.items()
                ]
            )
        )

        if self.write_rate > 0:
            for writer, (name, metric) in zip(file_writers, metrics.items()):
                with writer.as_default():
                    tf.summary.scalar(name, metric, epoch)

    def on_train_begin(self, logs=None):
        data = self.validation or self.test
        if data is None:
            return
        else:
            if data[1][-1].shape[-1] != 3:
                self._output_type = "distance_matrix"
            if "vectors" not in [layer.name for layer in self.model.layers]:
                self._prediction_type = "cartesians"

            self.write_cartesians(
                self.train,
                self.path / "pre_training" / "train",
                write_static_structures=True,
            )
            self.write_cartesians(
                self.validation,
                self.path / "pre_training" / "validation",
                write_static_structures=True,
            )

    def on_epoch_end(self, epoch, logs=None):
        if self.write_rate <= 0:
            return

        # if self._output_type == "cartesians":
        #     self.compute_metrics(epoch, "train")
        #     self.compute_metrics(epoch, "validation")

        if self._writing(epoch):
            self.total_epochs = epoch
            if self.validation is not None and self.train is not None:

                self.write_cartesians(
                    self.train, self.path / "epochs" / f"epoch_{epoch + 1}" / "train",
                )
                self.write_cartesians(
                    self.validation,
                    self.path / "epochs" / f"epoch_{epoch + 1}" / "validation",
                )

    def on_train_end(self, logs=None):
        for name, data in zip(
            ["train", "val", "test"], [self.train, self.validation, self.test]
        ):
            if data is not None:
                # print(
                #     f"final {name} loss: {round(self.model.evaluate(*data, verbose=0), 8)}"
                # )
                # if self._output_type == "cartesians":
                #     self.compute_metrics(self.total_epochs + 1, name)
                if self.write_rate > 0:
                    self.write_cartesians(data, self.path / "post_training" / name)


class ClassificationMetrics(Callback):
    def __init__(self, validation, log_dir):
        super().__init__()
        self.validation = validation
        self.val_f1s = None
        self.val_recalls = None
        self.val_precisions = None
        self.file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
        self.file_writer.set_as_default()

    def on_epoch_end(self, epoch, logs=None):
        target = self.validation[1]
        prediction = np.asarray(self.model.predict(self.validation[0]))
        f1score = self.f1_score(target, prediction)
        precision = self.precision(target, prediction)
        recall = self.recall(target, prediction)
        accuracy = np.mean(categorical_accuracy(target, prediction))
        tf.summary.scalar("f1score", f1score, epoch)
        tf.summary.scalar("precision", precision, epoch)
        tf.summary.scalar("recall", recall, epoch)
        tf.summary.scalar("accuracy", accuracy, epoch)
        print(
            f"Metrics for epoch {epoch}:"
            f" -- val_f1score: {f1score} -- val_precision: {precision} -- val_recall: {recall} "
            f" -- val_accuracy: {accuracy}"
        )

    def f1_score(self, y_true, y_pred):
        recall = self.recall(y_true, y_pred)
        precision = self.precision(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + 1e-7))

    def recall(self, y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + 1e-7)

    def precision(self, y_true, y_pred):
        true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
        predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + 1e-7)
