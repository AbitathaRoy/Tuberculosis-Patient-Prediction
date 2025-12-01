# register_activation.py

from keras.layers import Layer
from keras.saving import register_keras_serializable
import keras.ops as ops
import tensorflow as tf

@register_keras_serializable(package="Custom")

# Modified ReLU Activation Function
class ModifiedReLU(Layer):
    def __init__(self, a=0.01, **kwargs):
        super(ModifiedReLU, self).__init__(**kwargs)
        self.a = a

    def call(self, inputs):
        return tf.where(inputs > 0, self.a * inputs, inputs)

    def get_config(self):
        config = super(ModifiedReLU, self).get_config()
        config.update({"a": self.a})
        return config