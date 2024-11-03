import re
import json
import numpy as np

global_vars = {}


def update_globals(**kwargs):
    global global_vars

    global_vars.update(**kwargs)


def get_globals():
    global global_vars

    return global_vars


def parse_npy(x: str):
    """
    Parse Numpy output as array
    :param x:
    :return:
    """
    x = re.sub(r"(\d)\s+([-0-9])", lambda m: f"{m.group(1)}, {m.group(2)}", x)
    x = re.sub(r"\]\s+\[", "],\n[", x)
    x = json.loads(x)

    return np.asarray(x)
