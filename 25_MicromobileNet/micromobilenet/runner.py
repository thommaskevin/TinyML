import logging
import os.path
import warnings
from tempfile import gettempdir
from subprocess import check_output, check_call

import numpy as np


class Runner:
    """
    Run C++ MobileNet
    """
    def __init__(self, net):
        """

        :param net:
        """
        self.net = net

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict samples
        :param X:
        :return:
        """
        root = os.path.abspath(gettempdir())
        logging.warning(f"setting CWD={root}")

        # save input to binary file
        with open(os.path.join(root, "X.bin"), "wb") as file:
            file.write(X.flatten().astype(np.uint8).tobytes("C"))

        # save net to file
        with open(os.path.join(root, "MobileNet.h"), "w") as file:
            file.write(self.net.convert.to_cpp(classname="MobileNet"))

        # create C++ main file
        src = os.path.join(os.path.dirname(__file__), "convert", "templates", "predict_file.jinja")
        dest = os.path.join(root, "mobilenet_test.cpp")

        with open(src) as fin, open(dest, "w") as fout:
            fout.write(fin.read())

        # compile (disable compilation warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if check_call(["g++", "mobilenet_test.cpp", "-o", "mobilenet_test"], cwd=root) == 0:
                output = check_output(["./mobilenet_test"], cwd=root).decode()
                return np.asarray([int(x) for x in output.split("\n") if x.strip()])
