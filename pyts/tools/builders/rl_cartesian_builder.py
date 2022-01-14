import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, Add, Dense, Flatten

from .rl_multi_trunk_builder import DualTrunkBuilder
from ...layers.utility_layers import MaskedDistanceMatrix


class RlCartesianBuilder(DualTrunkBuilder):
    def __init__(self, *args, **kwargs):
        self.prediction_type = kwargs.pop(
            "prediction_type", "cartesians"
        )  # or 'vectors'
        self.output_type = kwargs.pop(
            "output_type", "cartesians"
        )  # or 'distance_matrix'
        super().__init__(*args, **kwargs)

    def get_inputs(self):
        return [
            Input([self.num_points,],   name="atomic_nums", dtype="int32"),
            Input([self.num_points, 3], name="reactant_cartesians", dtype="float32"),
            Input([self.num_points, 3], name="product_cartesians", dtype="float32"),
            Input([self.num_points, 3], name="ts_cartesians", dtype="float32"),
        ]

    def get_model(self, use_cache=True):
        if self.model is not None and use_cache:
            return self.model
        inputs = self.get_inputs()
        point_cloud, learned_tensors = self.get_learned_output(inputs)
        output = self.get_model_output(point_cloud, learned_tensors)
        if self.prediction_type == "vectors":
            # mix reactant and product cartesians commutatively
            midpoint = Lambda(lambda x: (x[0] + x[1]) / 2, name="midpoint")(
                [inputs[1], inputs[2]]
            )
            output = Add(name="cartesians")([midpoint, output])  # (batch, points, 3)
        if self.output_type == "distance_matrix":
            output = MaskedDistanceMatrix(name="distance_matrix")(
                [point_cloud[0][0], output]
            )  # (batch, points, points)
            output = Lambda(
                lambda x: tf.linalg.band_part(x, 0, -1), name="upper_triangle"
            )(output)
        self.model = Model(inputs=inputs, outputs=output, name=self.name)
        return self.model

    def get_learned_output(self, inputs: list):
        z, r, p, t = inputs
        point_clouds = [self.point_cloud_layer([z, x]) for x in (r, p, t)]
        inputs = self.get_dual_trunks(point_clouds)
        return point_clouds, inputs

    def get_model_output(self, point_cloud: list, inputs: list):
        one_hot, output = self.mix_dual_trunks(
            point_cloud, inputs, output_order=1, output_type=self.output_type
        )
        output = Lambda(lambda x: tf.squeeze(x, axis=-2), name=self.prediction_type)(
            output[0]
        )
        output = Flatten(name="Flatten")(output)
        output = Dense(100, activation='relu', name="dense")(
            output
        )
        return output  # (batch, points, 3)

    # def get_model_output(self, point_cloud: list, inputs: list):
    #     one_hot, output_r, output_p = self.mix_dual_trunks(
    #         point_cloud, inputs, output_order=1, output_type=self.output_type
    #     )
    #     output_r = Lambda(lambda x: tf.squeeze(x, axis=-2), name=f"{self.prediction_type}_r")(
    #         output_r[0]
    #     )
    #     output_p = Lambda(lambda x: tf.squeeze(x, axis=-2), name=f"{self.prediction_type}_p")(
    #         output_p[0]
    #     )
    #     return [output_r, output_p]  # (batch, points, 3)
