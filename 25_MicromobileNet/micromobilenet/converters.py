import numpy as np


def convert_xs(xs: np.ndarray, ys: np.ndarray) -> str:
    """
    Convert one sample for each y
    :param ys:
    :param xs:
    :return:
    """
    samples = []
    ys = ys.argmax(axis=1)

    for y in range(ys.max()):
        sample = xs[ys == y][-1].flatten()
        data = ", ".join("%.4f" % xi for xi in sample)
        samples.append(f"float x{y}[{len(sample)}] = {{ {data} }};")

    return "\n".join(samples)