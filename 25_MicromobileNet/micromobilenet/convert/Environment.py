from typing import List

from jinja2 import Environment as Base
from os.path import normpath, join, dirname
from math import ceil, floor
import numpy as np


class Environment(Base):
    """
    Override default Environment
    """
    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """
        filters = kwargs.pop("filters", {})
        globals = kwargs.pop("globals", {})
        kwargs.setdefault("extensions", [])

        super().__init__(*args, **kwargs)
        self._add_filters()
        self._add_globals()
        self.filters.update(filters)
        self.globals.update(globals)

    def join_path(self, template: str, parent: str) -> str:
        """

        :param template:
        :param parent:
        :return:
        """
        return normpath(join(dirname(parent), template))

    def _add_filters(self):
        """
        Add language-agnostic filters
        :return:
        """
        def to_array(arr) -> str:
            values = ", ".join("%.11f" % x for x in arr.flatten())
            return f"{{{values}}}"

        def to_weights_shape(weights: np.ndarray) -> str:
            h, w, c, d = weights.shape

            if d == 1:
                # depthwise kernel
                return f"[{c}][{h * w}]"

            return f"[{d}][{h * w * c}]"

        def to_weights_array(weights: np.ndarray) -> str:
            h, w, c, d = weights.shape

            if d == 1:
                # depthwise kernel
                values = ",\n".join(to_array(weights[:, :, i]) for i in range(c))
            else:
                values = ",\n".join(to_array(weights[:, :, :, i]) for i in range(d))

            return f"{{{values}}}"

        self.filters.update({
            "ceil": ceil,
            "floor": floor,
            "to_array": to_array,
            "to_weights_shape": to_weights_shape,
            "to_weights_array": to_weights_array
        })

    def _add_globals(self):
        """
        Add language-agnostic globals
        :return:
        """
        self.globals.update({
            "np": np,
            "len": len,
            "zip": zip,
            "int": int,
            "ceil": ceil,
            "eps": 0.0001,
            "floor": floor,
            "range": range,
            "sorted": sorted,
            "enumerate": enumerate,
            "isinstance": isinstance
        })
