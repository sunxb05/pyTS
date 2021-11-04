import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda, Add
from tensorflow.keras import backend as K
from .multi_trunk_builder import DualTrunkBuilder
from ...layers.utility_layers import MaskedDistanceMatrix
import time



class CartesianBuilder(DualTrunkBuilder):
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
            Input([self.num_points,], name="atomic_nums", dtype="int32"),
            Input([self.num_points, 3], name="reactant_cartesians", dtype="float32"),
            # Input([self.num_points, 3], name="product_cartesians", dtype="float32"),
        ]


    def get_model(self, use_cache=True):
        if self.model is not None and use_cache:
            return self.model
        # loss = 0
        inputs = self.get_inputs()

        # for i in range(1):
        #     if i == 0:
        #         point_cloud, learned_tensors = self.get_learned_output(inputs, i)
        #     else:
        #         point_cloud, learned_tensors = self.get_learned_output([inputs[0], output], i)
        #     output = self.get_model_output(point_cloud, learned_tensors, i)

        point_cloud, learned_tensors = self.get_learned_output(inputs, 1)
        output1 = self.get_model_output(point_cloud, learned_tensors, 1)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output1], 2)
        output2 = self.get_model_output(point_cloud, learned_tensors, 2)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output2], 3)
        output3 = self.get_model_output(point_cloud, learned_tensors, 3)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output3], 4)
        output4 = self.get_model_output(point_cloud, learned_tensors, 4)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output4], 5)
        output5 = self.get_model_output(point_cloud, learned_tensors, 5)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output5], 6)
        output6 = self.get_model_output(point_cloud, learned_tensors, 6)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output6], 7)
        output7 = self.get_model_output(point_cloud, learned_tensors, 7)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output7], 8)
        output8 = self.get_model_output(point_cloud, learned_tensors, 8)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output8], 9)
        output9 = self.get_model_output(point_cloud, learned_tensors, 9)

        point_cloud, learned_tensors = self.get_learned_output([inputs[0], output9], 10)
        output10 = self.get_model_output(point_cloud, learned_tensors, 10)



        self.model = Model(inputs=inputs, outputs=[output1, output2, output3, output4, output5, output6, output7, output8, output9, output10], name=self.name)
        return self.model

        # self.model = Model(inputs=inputs, outputs=[output1, output2], name=self.name)
        # return self.model






    # def get_model(self, use_cache=True):
    #     if self.model is not None and use_cache:
    #         return self.model
    #     inputs = self.get_inputs()
    #     point_cloud, learned_tensors = self.get_learned_output(inputs)
    #     output = self.get_model_output(point_cloud, learned_tensors)
    #     if self.prediction_type == "vectors":
    #         # mix reactant and product cartesians commutatively
    #         midpoint = Lambda(lambda x: (x[0] + x[1]) / 2, name="midpoint")(
    #             [inputs[1], inputs[2]]
    #         )
    #         output = Add(name="cartesians")([midpoint, output])  # (batch, points, 3)
    #     if self.output_type == "distance_matrix":
    #         output = MaskedDistanceMatrix(name="distance_matrix")(
    #             [point_cloud[0][0], output]
    #         )  # (batch, points, points)
    #         output = Lambda(
    #             lambda x: tf.linalg.band_part(x, 0, -1), name="upper_triangle"
    #         )(output)

    #     self.model = Model(inputs=inputs, outputs=output, name=self.name)
    #     return self.model


    def get_learned_output(self, inputs: list, layer_name):
        z, r = inputs
        point_clouds = [self.point_cloud_layer([z, r])]
        inputs = self.get_dual_trunks(point_clouds, layer_name)
        return point_clouds, inputs

    def get_model_output(self, point_cloud: list, inputs: list, layer_name):
        one_hot, output = self.mix_dual_trunks(
            point_cloud, inputs, layer_name, output_order=1, output_type=self.output_type
        )
        output = Lambda(lambda x: tf.squeeze(x, axis=-2), name=f"{self.prediction_type}_{layer_name}")(
            output[0]
        )
        return output  # (batch, points, 3)
