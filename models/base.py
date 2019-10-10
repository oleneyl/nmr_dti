import tensorflow as tf


class BaseModel(tf.keras.layers.Layer):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args

    def get_output(self):
        return self.output

    def call(self):
        pass
