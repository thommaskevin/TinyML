"""
TinyML - Radial Basis Function Networks
Author: Thommas Kevin Sales Flores
Institution: Federal University of Rio Grande do Norte
Email: thommas.flores.101@ufrn.edu.br
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import base64
import zlib
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Union, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import warnings

# ============================
# 1. Expanded Enums and Types
# ============================

class RBFType(Enum):
    GAUSSIAN = "gaussian"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    THIN_PLATE_SPLINE = "thin_plate_spline"

class CenterInitType(Enum):
    KMEANS = "kmeans"
    RANDOM = "random"
    UNIFORM = "uniform"
    PCA_BASED = "pca_based"

class WidthCalcType(Enum):
    P_NEAREST = "p_nearest"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    COVERAGE = "coverage"

class TrainingMethod(Enum):
    DIRECT = "direct"
    GRADIENT = "gradient"
    RLS = "rls"
    EKF = "ekf"

class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    MAHALANOBIS = "mahalanobis"
    COSINE = "cosine"

class LayerType(Enum):
    RBF = "rbf"
    DENSE = "dense"

# ============================
# 2. Expanded Data Classes
# ============================

@dataclass
class PerformanceMetrics:
    """Network performance metrics"""
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    test_loss: Optional[float] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None
    train_rmse: Optional[float] = None
    val_rmse: Optional[float] = None
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    epochs_completed: Optional[int] = None
    final_learning_rate: Optional[float] = None

@dataclass
class RBFNetworkMetadata:
    """RBF network metadata"""
    creation_date: str
    framework_version: str = "3.0.0"
    description: str = "Advanced RBF Network Model"
    author: str = "RBF Framework Pro"
    tags: List[str] = field(default_factory=lambda: ["rbf", "neural_network", "machine_learning", "advanced"])

@dataclass
class RBFNetworkParameters:
    """Configurable RBF network parameters for serialization"""
    hidden_layer_sizes: List[int]
    hidden_layer_types: List[str]
    output_activation: str
    use_bias: bool
    rbf_types: List[str]
    distance_metric: str
    center_init_types: List[str]
    width_calc_types: List[str]
    training_method: str
    regularization: float
    learning_rate: float
    max_iter: int
    tolerance: float
    p_neighbors: int
    width_value: float
    sigma_values: List[float]
    c_values: List[float]
    optimizer: str
    beta1: float
    beta2: float
    epsilon: float

# ============================
# 3. Distance Functions
# ============================

class DistanceMetricCalculator:
    """Distance metric calculator"""
    
    @staticmethod
    def calculate(x: np.ndarray, y: np.ndarray, metric: str = "euclidean", **kwargs) -> float:
        metric = metric.lower()
        if metric == "euclidean":
            return np.sqrt(np.sum((x - y) ** 2))
        elif metric == "manhattan":
            return np.sum(np.abs(x - y))
        elif metric == "cosine":
            dot_product = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            return 1 - (dot_product / (norm_x * norm_y + 1e-8))
        else:
            # Default fallback
            return np.sqrt(np.sum((x - y) ** 2))

# ============================
# 4. Activation Functions
# ============================

class ActivationFunction:
    """Activation functions for dense layers"""
    
    @staticmethod
    def apply(x: np.ndarray, activation: str, **kwargs) -> np.ndarray:
        activation = activation.lower()
        if activation == "linear":
            return x
        elif activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -709, 709)))
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "relu":
            return np.maximum(0, x)
        elif activation == "softmax":
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        elif activation == "softplus":
            return np.log(1 + np.exp(np.clip(x, -709, 709)))
        else:
            return x

# ============================
# 5. Radial Basis Functions
# ============================

class RBF(ABC):
    """Abstract class for radial basis functions"""
    def __init__(self, sigma=1.0, c=1.0, epsilon=1e-8):
        self.sigma = sigma
        self.c = c
        self.epsilon = epsilon
    
    @abstractmethod
    def __call__(self, r):
        pass

class GaussianRBF(RBF):
    def __call__(self, r):
        return np.exp(-(r**2) / (2 * self.sigma**2 + self.epsilon))

class MultiquadricRBF(RBF):
    def __call__(self, r):
        return np.sqrt(r**2 + self.c**2)

class InverseMultiquadricRBF(RBF):
    def __call__(self, r):
        return 1.0 / (np.sqrt(r**2 + self.c**2) + self.epsilon)

class ThinPlateSplineRBF(RBF):
    def __call__(self, r):
        r_safe = np.where(r == 0, self.epsilon, r)
        return r_safe**2 * np.log(r_safe)

class ExponentialRBF(RBF):
    def __call__(self, r):
        return np.exp(-r / (2 * self.sigma**2 + self.epsilon))

class BumpRBF(RBF):
    def __call__(self, r):
        mask = r <= 1.0
        result = np.zeros_like(r)
        result[mask] = np.exp(-1 / (1 - r[mask]**2 + self.epsilon))
        return result

class QuadraticRBF(RBF):
    def __call__(self, r):
        return 1 + r**2

class CubicRBF(RBF):
    def __call__(self, r):
        return r**3

class WaveletRBF(RBF):
    def __call__(self, r):
        return np.cos(5 * r) * np.exp(-(r**2) / (2 * self.sigma**2 + self.epsilon))

# ============================
# 6. Center Initialization
# ============================

class CenterInitializer(ABC):
    @abstractmethod
    def initialize(self, X, n_centers):
        pass

class KMeansInitializer(CenterInitializer):
    def __init__(self, random_state=42):
        self.random_state = random_state
        
    def initialize(self, X, n_centers):
        n_samples = X.shape[0]
        if n_centers > n_samples:
            n_centers = n_samples
        kmeans = KMeans(n_clusters=n_centers, random_state=self.random_state, n_init='auto')
        kmeans.fit(X)
        return kmeans.cluster_centers_

class RandomInitializer(CenterInitializer):
    def initialize(self, X, n_centers):
        indices = np.random.choice(len(X), min(n_centers, len(X)), replace=False)
        return X[indices]

class UniformInitializer(CenterInitializer):
    def initialize(self, X, n_centers):
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        return np.random.uniform(min_vals, max_vals, (n_centers, X.shape[1]))

class PCABasedInitializer(CenterInitializer):
    def initialize(self, X, n_centers):
        # Simplification to not depend on external PCA if not necessary
        # Robust placeholder implementation
        return X[np.random.choice(len(X), n_centers, replace=False)]

class SOMBasedInitializer(CenterInitializer):
    def initialize(self, X, n_centers):
        return X[np.random.choice(len(X), n_centers, replace=False)]

class MaxMinInitializer(CenterInitializer):
    def initialize(self, X, n_centers):
        centers = [X[np.random.randint(len(X))]]
        while len(centers) < n_centers:
            dists = np.array([min([np.linalg.norm(x - c) for c in centers]) for x in X])
            centers.append(X[np.argmax(dists)])
        return np.array(centers)

# ============================
# 7. Width Calculation
# ============================

class WidthCalculator(ABC):
    @abstractmethod
    def calculate(self, centers, X=None):
        pass

class AdaptiveWidth(WidthCalculator):
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def calculate(self, centers, X=None):
        if len(centers) <= 1:
            return np.ones(len(centers)) * self.multiplier
        dists = euclidean_distances(centers)
        np.fill_diagonal(dists, np.inf)
        nearest = np.min(dists, axis=1)
        return nearest * self.multiplier

class FixedWidth(WidthCalculator):
    def __init__(self, width=1.0):
        self.width = width

    def calculate(self, centers, X=None):
        return np.ones(centers.shape[0]) * self.width

class PNearestNeighborWidth(WidthCalculator):
    def __init__(self, p=2):
        self.p = p

    def calculate(self, centers, X=None):
        if len(centers) <= 1:
            return np.ones(len(centers))
        dists = euclidean_distances(centers)
        np.fill_diagonal(dists, np.inf)
        dists.sort(axis=1)
        return np.mean(dists[:, :self.p], axis=1)

class CoverageWidth(WidthCalculator):
    def calculate(self, centers, X=None):
        return np.ones(len(centers)) # Placeholder

class SilvermanWidth(WidthCalculator):
    def calculate(self, centers, X=None):
        return np.ones(len(centers)) # Placeholder

class MedianWidth(WidthCalculator):
    def calculate(self, centers, X=None):
        return np.ones(len(centers)) # Placeholder

# ============================
# 8. Main Advanced RBF Network Class
# ============================

class AdvancedRBFNetwork:
    """Advanced Radial Basis Function Neural Network"""
    
    def __init__(self, 
                 hidden_layers: List[Dict] = None,
                 output_activation: str = 'linear',
                 rbf_types: Union[str, List[str]] = 'gaussian',
                 distance_metric: str = 'euclidean',
                 center_init_types: Union[str, List[str]] = 'kmeans',
                 width_calc_types: Union[str, List[str]] = 'adaptive',
                 training_method: str = 'direct',
                 regularization: float = 0.0,
                 learning_rate: float = 0.01,
                 max_iter: int = 1000,
                 tolerance: float = 1e-4,
                 batch_size: int = 32,
                 p_neighbors: int = 2,
                 width_value: float = 1.0,
                 sigma_values: Union[float, List[float]] = 1.0,
                 c_values: Union[float, List[float]] = 1.0,
                 optimizer: str = 'adam',
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 use_bias: bool = True,
                 normalize_input: bool = True,
                 normalize_output: bool = True,
                 verbose: bool = False):
        
        # ===== ARCHITECTURE =====
        if hidden_layers is None:
            hidden_layers = [{'type': 'rbf', 'units': 10}]
        
        self.hidden_layers_config = hidden_layers
        self.output_activation = output_activation
        self.use_bias = use_bias
        
        # ===== RBF HYPERPARAMETERS =====
        num_layers = len(hidden_layers)
        self.rbf_types = self._to_list(rbf_types, num_layers)
        self.distance_metric = distance_metric
        self.center_init_types = self._to_list(center_init_types, num_layers)
        self.width_calc_types = self._to_list(width_calc_types, num_layers)
        
        # ===== TRAINING =====
        self.training_method = training_method
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tolerance
        self.batch_size = batch_size
        
        # ===== SPECIFIC PARAMETERS =====
        self.p_neighbors = p_neighbors
        self.width_value = width_value
        self.sigma_values = self._to_list(sigma_values, num_layers)
        self.c_values = self._to_list(c_values, num_layers)
        
        # ===== OPTIMIZATION =====
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # ===== CONFIGURATIONS =====
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.verbose = verbose
        
        # ===== NETWORK COMPONENTS =====
        self.layers = []
        self.scaler_X = StandardScaler() if normalize_input else None
        self.scaler_y = StandardScaler() if normalize_output else None
        
        self.performance_metrics = PerformanceMetrics()
    
    def _to_list(self, value, length):
        if isinstance(value, list):
            if len(value) != length:
                # Simple padding if sizes don't match, to avoid crash on load
                if len(value) < length:
                    return value + [value[-1]] * (length - len(value))
                return value[:length]
            return value
        else:
            return [value] * length
    
    def _init_rbf_function(self, rbf_type, sigma, c):
        rbf_classes = {
            'gaussian': GaussianRBF,
            'multiquadric': MultiquadricRBF,
            'inverse_multiquadric': InverseMultiquadricRBF,
            'thin_plate_spline': ThinPlateSplineRBF,
            'exponential': ExponentialRBF,
            'bump': BumpRBF,
            'quadratic': QuadraticRBF,
            'cubic': CubicRBF,
            'wavelet': WaveletRBF
        }
        
        if rbf_type not in rbf_classes:
            # Fallback to gaussian if unknown
            return GaussianRBF(sigma=sigma, c=c)
        
        return rbf_classes[rbf_type](sigma=sigma, c=c)
    
    def _init_center_strategy(self, center_init_type):
        init_classes = {
            'kmeans': KMeansInitializer,
            'random': RandomInitializer,
            'uniform': UniformInitializer,
            'pca_based': PCABasedInitializer,
            'som_based': SOMBasedInitializer,
            'max_min': MaxMinInitializer
        }
        return init_classes.get(center_init_type, KMeansInitializer)()

    def _init_width_strategy(self, width_calc_type, p_neighbors, width_value):
        calc_classes = {
            'p_nearest': PNearestNeighborWidth,
            'fixed': FixedWidth,
            'adaptive': AdaptiveWidth,
            'coverage': CoverageWidth,
            'silverman': SilvermanWidth,
            'median': MedianWidth
        }
        
        calc_class = calc_classes.get(width_calc_type, AdaptiveWidth)
        
        if width_calc_type == 'p_nearest':
            return calc_class(p=p_neighbors)
        elif width_calc_type == 'fixed':
            return calc_class(width=width_value)
        elif width_calc_type == 'adaptive':
            return calc_class(multiplier=width_value)
        else:
            return calc_class()

    def _calculate_distance(self, x, center):
        return DistanceMetricCalculator.calculate(x, center, self.distance_metric)
    
    def _build_network(self, X_shape, y_shape):
        input_dim = X_shape[1]
        
        for i, layer_config in enumerate(self.hidden_layers_config):
            layer_type = layer_config.get('type', 'rbf')
            units = layer_config.get('units', 10)
            
            if layer_type == 'rbf':
                rbf_type = self.rbf_types[i]
                center_init = self.center_init_types[i]
                width_calc = self.width_calc_types[i]
                sigma = self.sigma_values[i]
                c = self.c_values[i]
                
                layer = {
                    'type': 'rbf',
                    'units': units,
                    'rbf_function': self._init_rbf_function(rbf_type, sigma, c),
                    'center_initializer': self._init_center_strategy(center_init),
                    'width_calculator': self._init_width_strategy(width_calc, self.p_neighbors, self.width_value),
                    'centers': None,
                    'widths': None
                }
            else:
                activation = layer_config.get('activation', 'relu')
                layer = {
                    'type': 'dense',
                    'units': units,
                    'activation': activation,
                    'weights': None,
                    'bias': None if not self.use_bias else np.zeros((1, units))
                }
            
            self.layers.append(layer)
        
        # Output layer
        output_units = 1 if len(y_shape) == 1 else y_shape[1]
        output_layer = {
            'type': 'dense',
            'units': output_units,
            'activation': self.output_activation,
            'weights': None,
            'bias': None if not self.use_bias else np.zeros((1, output_units))
        }
        self.layers.append(output_layer)

    def fit(self, X, y, validation_data=None, **kwargs):
        X = np.asarray(X)
        y = np.asarray(y)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if self.normalize_input:
            X = self.scaler_X.fit_transform(X)
        
        if self.normalize_output:
            y = self.scaler_y.fit_transform(y)
        
        if not self.layers:
            self._build_network(X.shape, y.shape)
        
        if self.training_method == 'direct':
            self._fit_direct(X, y)
        elif self.training_method == 'gradient':
            self._fit_gradient(X, y, validation_data)
        else:
            self._fit_direct(X, y) # Fallback
            
        self._calculate_metrics(X, y, validation_data)
        return self

    def _fit_direct(self, X, y):
            """Training by direct solution (least squares)"""
            # For each RBF layer, initialize centers and widths
            current_X = X.copy()
            for i, layer in enumerate(self.layers):
                if layer['type'] == 'rbf':
                    # Initialize centers
                    layer['centers'] = layer['center_initializer'].initialize(current_X, layer['units'])
                    
                    # Calculate widths
                    layer['widths'] = layer['width_calculator'].calculate(layer['centers'], current_X)
                    
                    # Calculate RBF activations
                    activations = self._rbf_activation(current_X, layer['centers'], layer['widths'], 
                                                    layer['rbf_function'], self.distance_metric)
                    current_X = activations  # Output becomes input for next layer
            
            # Last layer (output) - training by least squares
            output_layer = self.layers[-1]
            
            # Add bias if necessary
            if self.use_bias:
                Phi = np.hstack([current_X, np.ones((current_X.shape[0], 1))])
            else:
                Phi = current_X
            
            # Regularization
            n_features = Phi.shape[1]
            I = np.eye(n_features)
            if self.use_bias:
                I[-1, -1] = 0  # Do not regularize bias
            
            # Regularized least squares solution
            A = Phi.T @ Phi + self.regularization * I
            b = Phi.T @ y
            
            try:
                theta = np.linalg.lstsq(A, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Add small identity to ensure inversion
                A += np.eye(A.shape[0]) * 1e-8
                theta = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # Separate weights and bias
            if self.use_bias:
                output_layer['weights'] = theta[:-1, :]
                # CORRECTION HERE: Removed .T
                # Before: output_layer['bias'] = theta[-1:, :].T
                # Now: Keep shape (1, n_outputs) for correct broadcasting
                output_layer['bias'] = theta[-1:, :] 
            else:
                output_layer['weights'] = theta

    def _rbf_activation(self, X, centers, widths, rbf_function, distance_metric='euclidean'):
        n_samples = X.shape[0]
        n_centers = centers.shape[0]
        
        # Calculate distances from all points to all centers
        # Using sklearn's euclidean_distances which is optimized
        if distance_metric == 'euclidean':
            distances = euclidean_distances(X, centers)
        else:
            distances = np.zeros((n_samples, n_centers))
            for i in range(n_samples):
                for j in range(n_centers):
                    distances[i, j] = self._calculate_distance(X[i], centers[j])
        
        # Broadcasting widths for division
        widths_broadcast = widths.reshape(1, -1)
        normalized_distances = distances / (widths_broadcast + 1e-8)
        
        # Apply RBF function
        return rbf_function(normalized_distances)

    def _fit_gradient(self, X, y, validation_data=None):
        # Basic placeholder implementation to avoid excessive complexity in the fix
        self._fit_direct(X, y)

    def _calculate_metrics(self, X, y, validation_data=None):
        y_pred = self.predict(X, training_mode=True)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
        if y.ndim == 1: y = y.reshape(-1, 1)

        if self.normalize_output:
            y_pred_orig = self.scaler_y.inverse_transform(y_pred)
            y_orig = self.scaler_y.inverse_transform(y)
        else:
            y_pred_orig = y_pred
            y_orig = y

        self.performance_metrics.train_rmse = np.sqrt(np.mean((y_pred_orig - y_orig) ** 2))

        if validation_data:
            X_val, y_val = validation_data
            y_val_pred = self.predict(X_val)
            if y_val_pred.ndim == 1: y_val_pred = y_val_pred.reshape(-1, 1)
            if y_val.ndim == 1: y_val = y_val.reshape(-1, 1)
            
            if self.normalize_output:
                y_val_pred_orig = self.scaler_y.inverse_transform(y_val_pred)
                y_val_orig = self.scaler_y.inverse_transform(y_val)
            else:
                y_val_pred_orig = y_val_pred
                y_val_orig = y_val
                
            self.performance_metrics.val_rmse = np.sqrt(np.mean((y_val_pred_orig - y_val_orig) ** 2))

    def predict(self, X, training_mode=False):
        if not self.layers:
            raise ValueError("Model not trained. Call fit() first.")
        
        X = np.asarray(X)
        if self.normalize_input and hasattr(self.scaler_X, 'mean_'):
            X = self.scaler_X.transform(X)
        
        current_X = X
        for layer in self.layers:
            if layer['type'] == 'rbf':
                if layer['centers'] is None:
                    raise ValueError("RBF centers not initialized")
                current_X = self._rbf_activation(current_X, layer['centers'], layer['widths'], 
                                               layer['rbf_function'], self.distance_metric)
            else:
                z = current_X @ layer['weights']
                if self.use_bias and layer['bias'] is not None:
                    z += layer['bias']
                current_X = ActivationFunction.apply(z, layer['activation'])
        
        if self.normalize_output and hasattr(self.scaler_y, 'mean_') and not training_mode:
            current_X = self.scaler_y.inverse_transform(current_X)
        
        return current_X.flatten() if current_X.shape[1] == 1 else current_X

    # ============================
    # SERIALIZATION METHODS
    # ============================
    
    def to_dict(self) -> Dict[str, Any]:
        is_trained = any(layer.get('centers') is not None or layer.get('weights') is not None 
                        for layer in self.layers)
        
        metadata = asdict(RBFNetworkMetadata(creation_date=datetime.now().isoformat()))
        
        architecture = {
            "hidden_layers": self.hidden_layers_config,
            "output_activation": self.output_activation,
            "use_bias": self.use_bias,
            "normalize_input": self.normalize_input,
            "normalize_output": self.normalize_output
        }
        
        parameters = asdict(RBFNetworkParameters(
            hidden_layer_sizes=[layer.get('units', 0) for layer in self.hidden_layers_config],
            hidden_layer_types=[layer.get('type', 'rbf') for layer in self.hidden_layers_config],
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            rbf_types=self.rbf_types,
            distance_metric=self.distance_metric,
            center_init_types=self.center_init_types,
            width_calc_types=self.width_calc_types,
            training_method=self.training_method,
            regularization=self.regularization,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            tolerance=self.tol,
            p_neighbors=self.p_neighbors,
            width_value=self.width_value,
            sigma_values=self.sigma_values,
            c_values=self.c_values,
            optimizer=self.optimizer,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon
        ))
        
        learned_parameters = {}
        if is_trained:
            for i, layer in enumerate(self.layers):
                layer_key = f"layer_{i}_{layer['type']}"
                learned_parameters[layer_key] = {}
                
                if layer['type'] == 'rbf':
                    if layer['centers'] is not None:
                        learned_parameters[layer_key]['centers'] = self._array_to_dict(layer['centers'])
                    if layer['widths'] is not None:
                        learned_parameters[layer_key]['widths'] = self._array_to_dict(layer['widths'])
                else:
                    if layer['weights'] is not None:
                        learned_parameters[layer_key]['weights'] = self._array_to_dict(layer['weights'])
                    if self.use_bias and layer['bias'] is not None:
                        learned_parameters[layer_key]['bias'] = self._array_to_dict(layer['bias'])
        
        components = {
            "normalizers": {
                "input_scaler": self._array_to_dict(self.scaler_X.mean_) if hasattr(self.scaler_X, 'mean_') else None,
                "input_var": self._array_to_dict(self.scaler_X.var_) if hasattr(self.scaler_X, 'var_') else None,
                "output_scaler": self._array_to_dict(self.scaler_y.mean_) if hasattr(self.scaler_y, 'mean_') else None,
                "output_var": self._array_to_dict(self.scaler_y.var_) if hasattr(self.scaler_y, 'var_') else None
            }
        }
        
        return {
            "metadata": metadata,
            "architecture": architecture,
            "parameters": parameters,
            "learned_parameters": learned_parameters,
            "components": components
        }
    
    def _array_to_dict(self, array):
        if array is None: return None
        return {"data": array.tolist(), "shape": list(array.shape), "dtype": str(array.dtype)}
    
    def save_json(self, filename: str, compress: bool = False):
        model_dict = self.to_dict()
        if compress: model_dict = self._compress_model_dict(model_dict)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(model_dict, f, indent=2, ensure_ascii=False)
        print(f"Model saved at: {filename}")
    
    def _compress_model_dict(self, model_dict):
        # Simplified for the example
        return model_dict

    @classmethod
    def load_json(cls, filename: str):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        params = data["parameters"]
        
        # 1. Reconstruct layer configuration
        hidden_layers = []
        for size, layer_type in zip(params["hidden_layer_sizes"], params["hidden_layer_types"]):
            hidden_layers.append({"type": layer_type, "units": size})

        # 2. Instantiate network
        network = cls(
            hidden_layers=hidden_layers,
            output_activation=params["output_activation"],
            rbf_types=params["rbf_types"],
            distance_metric=params["distance_metric"],
            center_init_types=params["center_init_types"],
            width_calc_types=params["width_calc_types"],
            training_method=params["training_method"],
            regularization=params["regularization"],
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            tolerance=params["tolerance"],
            p_neighbors=params["p_neighbors"],
            width_value=params["width_value"],
            sigma_values=params["sigma_values"],
            c_values=params["c_values"],
            optimizer=params["optimizer"],
            beta1=params["beta1"],
            beta2=params["beta2"],
            epsilon=params["epsilon"],
            use_bias=params["use_bias"],
            verbose=False
        )

        # 3. Recreate layer structure (empty)
        network.layers = []
        rbf_layer_counter = 0

        for layer_cfg in hidden_layers:
            if layer_cfg["type"] == "rbf":
                # === CRITICAL CORRECTION ===
                # Instead of using lambda, we use the instance of the original RBF class
                # that stores sigma/c internally.
                rbf_type = network.rbf_types[rbf_layer_counter]
                sigma = network.sigma_values[rbf_layer_counter]
                c = network.c_values[rbf_layer_counter]
                
                # Uses the class's internal method to create the RBF function correctly
                rbf_function_instance = network._init_rbf_function(rbf_type, sigma, c)
                
                layer = {
                    "type": "rbf",
                    "units": layer_cfg["units"],
                    "centers": None,
                    "widths": None,
                    "rbf_function": rbf_function_instance # Instance with __call__(r)
                }
                rbf_layer_counter += 1
            else:
                layer = {
                    "type": "dense",
                    "units": layer_cfg["units"],
                    "weights": None,
                    "bias": None,
                    "activation": "relu" # Default fallback
                }
            network.layers.append(layer)

        # Add output layer
        # We need to infer output units from saved weights if possible, or use architecture
        # Here we simplify assuming it will be loaded below
        network.layers.append({
            "type": "dense",
            "units": 1, # Placeholder, will be adjusted during weight loading
            "activation": network.output_activation,
            "weights": None,
            "bias": None
        })

        # 4. Load learned parameters (weights/centers)
        if "learned_parameters" in data:
            for layer_key, layer_data in data["learned_parameters"].items():
                parts = layer_key.split('_')
                if len(parts) < 2: continue
                
                try: idx = int(parts[1])
                except ValueError: continue
                
                if idx >= len(network.layers): continue
                
                layer = network.layers[idx]
                
                for param_key, param_dict in layer_data.items():
                    if not param_dict or "data" not in param_dict: continue
                    array = np.array(param_dict["data"])
                    
                    if param_key in ["weights", "bias", "centers", "widths"]:
                        layer[param_key] = array
                        # Adjust output layer units if necessary
                        if idx == len(network.layers) - 1 and param_key == 'weights':
                            layer['units'] = array.shape[1]

        # 5. Load components (Scalers)
        comps = data.get("components", {}).get("normalizers", {})
        if network.normalize_input and comps.get("input_scaler"):
             network.scaler_X.mean_ = np.array(comps["input_scaler"]["data"])
             if comps.get("input_var"):
                 network.scaler_X.var_ = np.array(comps["input_var"]["data"])
                 network.scaler_X.scale_ = np.sqrt(network.scaler_X.var_)
        
        if network.normalize_output and comps.get("output_scaler"):
             network.scaler_y.mean_ = np.array(comps["output_scaler"]["data"])
             if comps.get("output_var"):
                 network.scaler_y.var_ = np.array(comps["output_var"]["data"])
                 network.scaler_y.scale_ = np.sqrt(network.scaler_y.var_)

        print(f"Model loaded from: {filename}")
        return network

    def visualize_architecture(self):
        print("Simplified architecture visualization (text mode):")
        for i, layer in enumerate(self.layers):
            t = layer['type']
            u = layer.get('units', '?')
            print(f"Layer {i}: Type={t}, Units={u}")

# ============================
# 9. Usage Example
# ============================

def simple_example():
    print("=" * 60)
    print("RBF NETWORK TEST")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0]**2 + np.sin(X[:, 1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = AdvancedRBFNetwork(
        hidden_layers=[{'type': 'rbf', 'units': 15}],
        rbf_types='gaussian',
        training_method='direct',
        max_iter=50,
        verbose=True
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    
    print("Saving...")
    model.save_json('model_fixed.json')
    
    print("Loading...")
    loaded_model = AdvancedRBFNetwork.load_json('model_fixed.json')
    
    print("Testing loaded model prediction...")
    try:
        y_pred = loaded_model.predict(X_test)
        print("Success! Prediction performed.")
        print(f"Prediction shape: {y_pred.shape}")
    except TypeError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"Other error: {e}")

if __name__ == "__main__":
    simple_example()