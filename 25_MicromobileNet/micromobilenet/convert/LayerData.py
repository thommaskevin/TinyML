import numpy as np


class LayerData:
    """
    Wrap layer to get its data
    """
    def __init__(self, layer):
        """

        :param layer:
        """
        self.layer = layer

    def __repr__(self):
        """
        Proxy
        :return:
        """
        return repr(self.layer)

    def __getattr__(self, item):
        """
        Proxy
        :param item:
        :return:
        """
        return getattr(self.layer, item)

    @property
    def io(self):
        return getattr(self.layer, "_io", None)

    @property
    def input_shape(self):
        return self.layer.input.shape[1:]

    @property
    def output_shape(self):
        return self.layer.output.shape[1:]

    @property
    def weights(self):
        return self.io["weights"] if self.io is not None else np.asarray(self.layer.weights[0])

    @property
    def bias(self):
        return self.io["bias"] if self.io is not None else self.layer.bias.numpy()

