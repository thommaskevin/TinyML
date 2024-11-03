from itertools import groupby
from os.path import join, dirname, realpath
from typing import Dict

import numpy as np

from micromobilenet.convert.Environment import Environment
from micromobilenet.convert.Loader import Loader
from micromobilenet.convert.LayerData import LayerData


class MobileNetConverter:
    """
    Convert BaseMobileNet to C++
    """
    def __init__(self, net: "BaseMobileNet"):
        """

        :param net:
        """
        self.net = net

    def to_cpp(self, classname: str = None) -> str:
        """
        Convert to C++
        :param classname:
        :return:
        """
        root = join(dirname(realpath(__file__)), "templates")
        loader = Loader(root)
        env = Environment(loader=loader)
        template = env.get_template("BaseMobileNet")
        data = self.get_data()

        if classname is not None:
            data.update(classname=classname)

        # render template
        output = template.render(data)

        return output

    def get_data(self) -> Dict:
        """
        Get data for code generation
        :return:
        """
        model = self.net.model
        classname = self.net.__class__.__name__
        layers = [LayerData(l) for l in self.net.layers]
        inputs = layers[0]
        conv_0 = LayerData(model.get_layer("conv2d_0"))
        maxpool = LayerData(model.get_layer("maxpool_last"))
        conv_last = LayerData(model.get_layer("conv2d_last"))
        softmax = LayerData(model.get_layer("softmax"))

        # group hidden layers into chunks
        hidden_layers = [l for l in layers if l.name.startswith("hidden_")]
        hidden_layers = [list(ll) for _, ll in groupby(hidden_layers, key=lambda l: l.name.split("__")[0])]
        hidden_layers = [{l.name.split("__")[1]: l for l in chunk} for chunk in hidden_layers]

        num_inputs = np.product(inputs.shape[1:])
        num_outputs = softmax.output_shape[-1]
        output_sizes = [np.product(l.output_shape) for l in layers[1:]]
        arena_size = max(output_sizes)

        return locals()