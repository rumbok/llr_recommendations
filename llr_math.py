import logging
from math import sqrt, log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def llr_sqrt(k11, k12, k21, k22):
    v_llr = llr(k11, k12, k21, k22)
    root = sqrt(v_llr)
    if float(k11) / (k11 + k12) < float(k21) / (k21 + k22):
        root = -root
    return root


def llr(k11, k12, k21, k22):
    row_entropy = entropy(k11 + k12, k21 + k22)
    column_entropy = entropy(k11 + k21, k12 + k22)
    matrix_entropy = entropy(k11, k12, k21, k22)
    if row_entropy + column_entropy < matrix_entropy:
        return 0.0

    return 2.0 * (row_entropy + column_entropy - matrix_entropy)


def entropy(a, b, c=0, d=0):
    """Вычисление перекрестной энтропии Шеннона"""
    return x_log_x(a + b + c + d) - x_log_x(a) - x_log_x(b) - x_log_x(c) - x_log_x(d)


def x_log_x(x):
    if x and x > 0:
        return x * log(x)
    else:
        return 0.0
