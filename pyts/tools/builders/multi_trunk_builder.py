import tensorflow as tf
from tensorflow.python.keras.layers import Lambda

from pyts.layers import MolecularConvolution, MolecularSelfInteraction, SelfInteraction
from pyts.tools.builders import Builder


class DualTrunkBuilder(Builder):
    def get_dual_trunks(self, point_clouds: list, layer_name):
        embedding_layer = MolecularSelfInteraction(
            self.embedding_units, name=f"embedding_{layer_name}"
        )
        embeddings = [
            self.make_embedding(pc[0], embedding_layer) for pc in point_clouds
        ]
        layers = self.get_layers()
        inputs = [
            self.get_learned_tensors(e, pc, layers)
            for e, pc in zip(embeddings, point_clouds)
        ]
        return inputs

    def mix_dual_trunks(
        self,
        point_cloud: list,
        inputs: list,
        layer_name,
        output_order: int = 0,
        output_type: str = None,

    ):
        # Select smaller molecule
        one_hot = [p[0] for p in point_cloud]
        # one_hots = [p[0] for p in point_cloud]  # [(batch, points, max_z), ...]
        # one_hot = Lambda(
        #     lambda x: tf.where(tf.reduce_sum(x[0]) > tf.reduce_sum(x[1]), x[1], x[0],),
        #     name="one_hot_select",
        # )(one_hots)
        # Truncate to RO0 outputs
        layer = MolecularConvolution(
            name=f"truncate_layer_{layer_name}",
            radial_factory=self.radial_factory,
            si_units=self.final_si_units,
            activation=self.activation,
            output_orders=[output_order],
            dynamic=self.dynamic,
            sum_points=self.sum_points,
        )
        output = [layer(z + x)[0] for x, z in zip(inputs, point_cloud)]
        # outputs = [layer(z + x)[0] for x, z in zip(inputs, point_cloud)]
        # output_type = output_type or "vectors"
        # if output_type == "cartesians":
        #     output = Lambda(lambda x: (x[0] + x[1]), name="learned_midpoint")(outputs)
        # else:
        #     output = Lambda(lambda x: tf.abs(x[1] - x[0]), name="absolute_difference")(
        #         outputs
        #     )
        output = self.get_final_output(one_hot, output, layer_name)
        return one_hot, output

    def get_final_output(self, one_hot: tf.Tensor, inputs: list, layer_name, output_dim: int = 1):
        output = inputs
        for i in range(self.num_final_si_layers):
            output = SelfInteraction(self.final_si_units, name=f"si_{i}_{layer_name}")(output)
        return SelfInteraction(output_dim, name=f"si_{self.num_final_si_layers}_{layer_name}")(
            output
        )

    def get_model_output(self, point_cloud: list, inputs: list):
        raise NotImplementedError