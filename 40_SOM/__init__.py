# som/__init__.py
"""
TinyML -- Self-Organizing Maps (SOM)
======================================
A modular framework for training, evaluating, and deploying
Self-Organizing Maps on microcontrollers (Arduino / ESP32).

Public API
----------
from model         import SOMModel
from layers        import NeighborhoodKernel, LearningRateSchedule, KERNELS
from losses        import compute_loss, LOSS_NAMES
from utils         import train_model, export_to_json, plot_umatrix, ...
from cpp_generator import generate_ino
"""
