import tensorflow as tf
from tensorflow.keras import layers

# ============================================================
# CUSTOM POSITION EMBEDDING LAYER
# ============================================================

class PositionEmbedding(layers.Layer):
    """
    Required custom layer for loading the trained Transformer model.
    This must match the PositionEmbedding layer used during training.
    """

    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim
        })
        return config

