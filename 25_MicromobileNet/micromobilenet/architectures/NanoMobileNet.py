from keras.layers import Conv2D, MaxPool2D, Dropout
from micromobilenet.architectures.BaseMobileNet import BaseMobileNet


class NanoMobileNet(BaseMobileNet):
    def make_layers(self):
        yield Conv2D(3, (3, 3), padding="valid", use_bias=False, strides=(2, 2), name="conv2d_0")
        yield self.make_depthwise(filters=6, stride=2)
        yield self.make_depthwise(filters=12, stride=2)
        yield self.make_depthwise(filters=24, stride=2)
        yield self.make_depthwise(filters=24, stride=2)
        yield MaxPool2D((3, 3), name="maxpool_last")
        yield Dropout(0.1, name="dropout")
