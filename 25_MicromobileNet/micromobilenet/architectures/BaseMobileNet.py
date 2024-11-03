import logging
import os.path
from typing import List, Generator, Iterable

import numpy as np
from cached_property import cached_property
from keras import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, Reshape, Softmax, ZeroPadding2D, ReLU, DepthwiseConv2D
from micromobilenet.convert.MobileNetConverter import MobileNetConverter
from micromobilenet.architectures.Config import Config


class BaseMobileNet:
    """
    Base class for BaseMobileNet architectures
    """
    def __init__(self, num_classes: int):
        """

        """
        self.num_classes = num_classes
        self.history = None
        self.layers = []
        self.config = Config()
        self.model = None
        self.i = 1

    def __repr__(self):
        """

        :return:
        """
        self.model.summary()
        return str(self.model)

    @property
    def weights_file(self) -> str:
        """
        Get path to weights file
        :return:
        """
        return f"{self.config.checkpoint_path}.weights.h5" if self.config.checkpoint_path != "" else ""

    @cached_property
    def convert(self) -> MobileNetConverter:
        """
        Get instance of C++ converter
        :return:
        """
        return MobileNetConverter(self)

    def build(self):
        """
        Generate model
        :return:
        """
        self.i = 1
        self.model = Sequential()
        self.add(Input(shape=(96, 96, 1), name="input"))

        # add middle layers
        for layers in self.make_layers():
            if isinstance(layers, Iterable):
                layers = list(layers)

            if not isinstance(layers, List):
                layers = [layers]

            for l in layers:
                self.add(l)

        # head
        self.add(Conv2D(self.num_classes, (1, 1), padding="same", name="conv2d_last"))
        self.add(Reshape((self.num_classes,), name="reshape"))
        self.add(Softmax(name="softmax"))

    def add(self, layer):
        """
        Add layer
        :param layer:
        :return:
        """
        self.model.add(layer)
        self.layers.append(layer)

    def load_weights(self, abort_on_fail: bool = True):
        """
        Load checkpoint
        :return:
        """
        assert self.config.checkpoint_path != "", "you must set net.config.checkpoint_path!"

        if os.path.isfile(self.weights_file):
            self.model.load_weights(self.weights_file)
        else:
            logging.warning(f"Cannot load weight file {self.weights_file}")

            if abort_on_fail:
                raise FileNotFoundError(self.weights_file)

        return self

    def compile(self):
        """
        Compile model
        :return:
        """
        if self.model is None:
            self.build()

        self.model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss=self.config.loss,
            metrics=self.config.metrics,
        )

        return self

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, epochs: int = 100):
        """
        Fit model
        :param train_x:
        :param train_y:
        :param val_x:
        :param val_y:
        :return:
        """
        callbacks = []

        if self.weights_file != "":
            callbacks.append(ModelCheckpoint(
                self.weights_file,
                monitor=f"val_{self.config.metrics[0]}",
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                initial_value_threshold=self.config.checkpoint_min_accuracy
            ))

        self.history = self.model.fit(
            train_x,
            train_y,
            validation_data=(val_x, val_y),
            batch_size=self.config.batch_size,
            epochs=epochs,
            verbose=self.config.verbosity,
            callbacks=callbacks
        )

        return self

    def predict(self, xs: np.ndarray) -> np.ndarray:
        """
        Predict
        :param xs:
        :return:
        """
        return self.model.predict(xs)

    def make_depthwise(self, filters: int, stride: int = 1, padding: str = "same") -> Generator:
        """
        Generate depthwise + pointwise layers
        :param padding:
        :param filters:
        :param stride:
        :return:
        """
        i = self.i
        self.i += 1

        if padding == "same":
            yield ZeroPadding2D(name=f"hidden_{i}__padding")

        yield DepthwiseConv2D((3, 3), padding="valid", strides=(stride, stride), use_bias=False, name=f"hidden_{i}__dw")
        yield ReLU(6., name=f"hidden_{i}__relu_1")
        yield Conv2D(filters, (1, 1), padding="same", strides=(1, 1), use_bias=False, name=f"hidden_{i}__pw")
        yield ReLU(6., name=f"hidden_{i}__relu_2")
