# quantile_regression/__init__.py
"""
TinyML — Quantile Regression
=============================
A modular framework for training, evaluating, and deploying quantile
regression neural networks on microcontrollers (Arduino / ESP32).

Public API
----------
from model   import QRModel
from layers  import PinballLoss, get_activation, ACTIVATIONS
from losses  import compute_loss, LOSS_NAMES
from utils   import train_model, export_to_json, plot_quantile_bands, ...
from vi      import quantile_elbo_loss, qr_reg_loss
from cpp_generator import generate_ino
"""
