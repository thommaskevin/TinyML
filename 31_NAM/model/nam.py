import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Activation, Layer
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import json



# ================== BiasLayer and NAM class ==================
class BiasLayer(Layer):
    """Trainable bias layer that can output a scalar or a vector."""
    def __init__(self, units=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.bias = self.add_weight(name='bias',
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.bias


class NAM:
    """
    Neural Additive Model.

    Parameters
    ----------
    feature_names : list of str
        Names of the input features.
    hidden_units : list of int, optional (default=[64, 32])
        Number of units in each hidden layer of every feature network.
    activation : str, optional (default='relu')
        Activation function for hidden layers.
    output_activation : str or None, optional (default=None)
        Activation for the final output. Use 'sigmoid' for binary classification,
        'softmax' for multiclass classification, None for regression.
    num_classes : int, optional (default=None)
        Number of classes for multiclass classification. Required if
        output_activation='softmax'. Ignored otherwise.
    """
    def __init__(self, feature_names, hidden_units=[64, 32], activation='relu',
                 output_activation=None, num_classes=None):
        self.feature_names = feature_names
        self.hidden_units = hidden_units
        self.activation = activation
        self.output_activation = output_activation
        self.num_classes = num_classes

        if output_activation == 'softmax' and num_classes is None:
            raise ValueError("For multiclass classification (softmax), you must specify num_classes.")
        if output_activation == 'softmax' and num_classes < 2:
            raise ValueError("num_classes must be >= 2 for softmax.")

        self.model = None
        self.scaler = StandardScaler()
        self.fitted_scaler = False

    def _build_model(self):
        """Build the Keras model (multiple inputs, one output)."""
        inputs = []
        feature_outputs = []

        # Determine output dimension of each feature network
        out_units = self.num_classes if self.output_activation == 'softmax' else 1

        for name in self.feature_names:
            inp = Input(shape=(1,), name=name)
            x = Dense(self.hidden_units[0], activation=self.activation, name=f'{name}_dense_0')(inp)
            for i, units in enumerate(self.hidden_units[1:], start=1):
                x = Dense(units, activation=self.activation, name=f'{name}_dense_{i}')(x)
            out = Dense(out_units, activation='linear', name=f'{name}_out')(x)   # logits
            inputs.append(inp)
            feature_outputs.append(out)

        if len(feature_outputs) > 1:
            add = Add()(feature_outputs)   # element-wise sum across features
        else:
            add = feature_outputs[0]

        bias = BiasLayer(units=out_units, name='bias_layer')(add)

        if self.output_activation == 'sigmoid':
            output = Activation('sigmoid')(bias)
        elif self.output_activation == 'softmax':
            output = Activation('softmax')(bias)
        else:
            output = bias

        self.model = Model(inputs=inputs, outputs=output)

    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=1, **kwargs):
        """
        Train the NAM.

        Parameters
        ----------
        X : pandas DataFrame or numpy array of shape (n_samples, n_features)
            Training data. Must have columns in the same order as `feature_names`.
        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target values. For multiclass with softmax, can be integer labels or one-hot.
        epochs, batch_size, validation_split, verbose : passed to model.fit()
        kwargs : additional arguments for model.compile() (e.g., optimizer, loss, metrics)
        """
        # Convert to numpy and normalize
        if hasattr(X, 'values'):
            X = X.values
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.fitted_scaler = True

        # Build model if not yet built
        if self.model is None:
            self._build_model()

        # Prepare inputs: list of arrays, each shape (n_samples, 1)
        inputs = [X_scaled[:, i:i+1] for i in range(X_scaled.shape[1])]

        # Default compilation arguments based on output_activation
        if self.output_activation == 'softmax':
            default_loss = 'categorical_crossentropy'
            default_metrics = ['accuracy']
        elif self.output_activation == 'sigmoid':
            default_loss = 'binary_crossentropy'
            default_metrics = ['accuracy']
        else:
            default_loss = 'mse'
            default_metrics = ['mae']

        compile_kwargs = {
            'optimizer': 'adam',
            'loss': default_loss,
            'metrics': default_metrics
        }
        compile_kwargs.update(kwargs)
        self.model.compile(**compile_kwargs)

        history = self.model.fit(
            inputs, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return history

    def predict(self, X):
        """
        Predict using the trained model.

        Returns
        -------
        - For regression: (n_samples,) array of predictions.
        - For binary classification: (n_samples,) array of probabilities.
        - For multiclass: (n_samples, n_classes) array of probabilities.
        """
        if hasattr(X, 'values'):
            X = X.values
        if self.fitted_scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        inputs = [X_scaled[:, i:i+1] for i in range(X_scaled.shape[1])]
        preds = self.model.predict(inputs)
        # Squeeze if regression/binary to keep 1D
        if self.output_activation not in ['softmax'] and preds.shape[-1] == 1:
            preds = preds.flatten()
        return preds

    def save_parameters(self, filepath):
        if self.model is None:
            raise RuntimeError("Model not built. Call fit() first.")

        params = {
            'feature_names': self.feature_names,
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'num_classes': self.num_classes,
            'bias': None,
            'scaler': {
                'mean': self.scaler.mean_.tolist() if self.fitted_scaler else None,
                'scale': self.scaler.scale_.tolist() if self.fitted_scaler else None
            },
            'networks': {}
        }

        bias_layer = self.model.get_layer('bias_layer')
        params['bias'] = bias_layer.get_weights()[0].tolist()   # list of floats (vector)

        for name in self.feature_names:
            layers = []
            i = 0
            while True:
                try:
                    layer = self.model.get_layer(f'{name}_dense_{i}')
                except ValueError:
                    break
                w, b = layer.get_weights()
                layers.append({'weights': w.tolist(), 'biases': b.tolist()})
                i += 1
            out_layer = self.model.get_layer(f'{name}_out')
            w_out, b_out = out_layer.get_weights()
            layers.append({'weights': w_out.tolist(), 'biases': b_out.tolist()})
            params['networks'][name] = {'layers': layers}

        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)

    def load_parameters(self, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)

        self.feature_names = params['feature_names']
        self.hidden_units = params['hidden_units']
        self.activation = params['activation']
        self.output_activation = params['output_activation']
        self.num_classes = params.get('num_classes', None)   # for backward compatibility

        self._build_model()

        for name, net_params in params['networks'].items():
            layers_data = net_params['layers']
            for i, layer_data in enumerate(layers_data):
                layer_name = f'{name}_dense_{i}' if i < len(layers_data)-1 else f'{name}_out'
                layer = self.model.get_layer(layer_name)
                layer.set_weights([np.array(layer_data['weights']), np.array(layer_data['biases'])])

        bias_layer = self.model.get_layer('bias_layer')
        bias_layer.set_weights([np.array(params['bias'])])

        if params['scaler']['mean'] is not None:
            self.scaler.mean_ = np.array(params['scaler']['mean'])
            self.scaler.scale_ = np.array(params['scaler']['scale'])
            self.fitted_scaler = True

    def export_to_c(self, filepath):
        """
        Generate C code for inference on a microcontroller.

        For multiclass classification, the generated code provides:
        - A function for each feature that writes its logit contributions into an array.
        - A predict function that sums these logits, adds bias, and applies softmax.
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call fit() first.")

        out_units = self.num_classes if self.output_activation == 'softmax' else 1

        lines = [
            "// Neural Additive Model generated by NAM framework",
            "#ifndef NAM_MODEL_H",
            "#define NAM_MODEL_H",
            "",
            "#include <math.h>",
            ""
        ]

        if self.fitted_scaler:
            lines.append(f"static const float SCALE_MEAN[{len(self.feature_names)}] = {{" +
                        ", ".join(f"{v:.8f}f" for v in self.scaler.mean_) + "};")
            lines.append(f"static const float SCALE_STD[{len(self.feature_names)}] = {{" +
                        ", ".join(f"{v:.8f}f" for v in self.scaler.scale_) + "};")
            lines.append("")

        # For each feature, generate weight arrays and a function that fills a logits array
        for name in self.feature_names:
            # Collect layers
            layers = []
            i = 0
            while True:
                try:
                    layer = self.model.get_layer(f'{name}_dense_{i}')
                except ValueError:
                    break
                layers.append(layer)
                i += 1
            layers.append(self.model.get_layer(f'{name}_out'))

            # Generate weight arrays
            for idx, layer in enumerate(layers):
                w, b = layer.get_weights()
                in_dim = w.shape[0]
                out_dim = w.shape[1]
                lines.append(f"static const float {name}_w_{idx}[{in_dim * out_dim}] = {{")
                row_strings = []
                for r in range(in_dim):
                    row_strings.append(", ".join(f"{v:.8f}f" for v in w[r, :]))
                lines.append("    " + ",\n    ".join(row_strings) + "};")
                lines.append(f"static const float {name}_b_{idx}[{out_dim}] = {{" +
                            ", ".join(f"{v:.8f}f" for v in b) + "};")
                lines.append("")

            # Generate feature function (writes logits to an output array)
            lines.append(f"static void feature_{name}(float x, float* out) {{")

            # Apply scaling if available
            if self.fitted_scaler:
                idx = self.feature_names.index(name)
                lines.append(f"    x = (x - SCALE_MEAN[{idx}]) / SCALE_STD[{idx}];")

            lines.append("    float in[1];")
            lines.append("    in[0] = x;")

            # Process each hidden layer
            for idx, layer in enumerate(layers[:-1]):
                w, b = layer.get_weights()
                out_dim = w.shape[1]
                in_name = "in" if idx == 0 else f"out_{idx-1}"
                w_name = f"{name}_w_{idx}"
                b_name = f"{name}_b_{idx}"
                lines.append(f"    float out_{idx}[{out_dim}];")
                lines.append(f"    for (int j = 0; j < {out_dim}; j++) {{")
                lines.append(f"        out_{idx}[j] = {b_name}[j];")
                lines.append(f"        for (int i = 0; i < {w.shape[0]}; i++) {{")
                lines.append(f"            out_{idx}[j] += {in_name}[i] * {w_name}[i * {out_dim} + j];")
                lines.append(f"        }}")
                lines.append(f"        if (out_{idx}[j] < 0) out_{idx}[j] = 0; // ReLU")
                lines.append(f"    }}")

            # Last layer (output layer) – writes to 'out' parameter
            last_layer = layers[-1]
            w_last, b_last = last_layer.get_weights()
            out_dim_last = w_last.shape[1]
            if len(layers) > 1:
                in_last = f"out_{len(layers)-2}"
            else:
                in_last = "in"
            w_last_name = f"{name}_w_{len(layers)-1}"
            b_last_name = f"{name}_b_{len(layers)-1}"
            lines.append(f"    for (int j = 0; j < {out_dim_last}; j++) {{")
            lines.append(f"        out[j] = {b_last_name}[j];")
            lines.append(f"        for (int i = 0; i < {w_last.shape[0]}; i++) {{")
            lines.append(f"            out[j] += {in_last}[i] * {w_last_name}[i * {out_dim_last} + j];")
            lines.append(f"        }}")
            lines.append(f"    }}")
            lines.append("}")
            lines.append("")

        # Global predict function
        if self.output_activation == 'softmax':
            lines.append(f"void predict(float features[], float probs[{self.num_classes}]) {{")
            lines.append(f"    float logits[{self.num_classes}] = {{0}};")
        else:
            lines.append("float predict(float features[]) {")
            if out_units == 1:
                lines.append("    float sum = 0.0f;")
                lines.append("    float out;")
            else:
                lines.append(f"    float logits[{out_units}] = {{0}};")

        # Accumulate contributions from each feature
        for name in self.feature_names:
            if self.output_activation == 'softmax':
                lines.append(f"    float temp_{name}[{self.num_classes}];")
                lines.append(f"    feature_{name}(features[{self.feature_names.index(name)}], temp_{name});")
                lines.append(f"    for (int i = 0; i < {self.num_classes}; i++) logits[i] += temp_{name}[i];")
            elif out_units == 1:
                # Regression with single output: use temporary variable
                lines.append(f"    feature_{name}(features[{self.feature_names.index(name)}], &out);")
                lines.append(f"    sum += out;")
            else:
                # Multi-output regression (not typical) – would need array handling
                pass

        # Add bias
        bias_vals = self.model.get_layer('bias_layer').get_weights()[0]
        if self.output_activation == 'softmax':
            for i, val in enumerate(bias_vals):
                lines.append(f"    logits[{i}] += {val:.8f}f;")
        elif out_units == 1:
            lines.append(f"    sum += {bias_vals.item():.8f}f;  // bias")
        else:
            # multi-output regression not handled
            pass

        # Final activation
        if self.output_activation == 'sigmoid':
            lines.append("    return 1.0f / (1.0f + expf(-sum));")
        elif self.output_activation == 'softmax':
            lines.append("    // Softmax")
            lines.append("    float max_logit = logits[0];")
            lines.append(f"    for (int i = 1; i < {self.num_classes}; i++) if (logits[i] > max_logit) max_logit = logits[i];")
            lines.append("    float sum_exp = 0.0f;")
            lines.append(f"    for (int i = 0; i < {self.num_classes}; i++) {{")
            lines.append("        probs[i] = expf(logits[i] - max_logit);")
            lines.append("        sum_exp += probs[i];")
            lines.append("    }")
            lines.append(f"    for (int i = 0; i < {self.num_classes}; i++) probs[i] /= sum_exp;")
            # No closing brace here – will be added after the chain
        elif out_units == 1:
            lines.append("    return sum;")

        # Always close the predict function
        lines.append("}")

        lines.append("")
        lines.append("#endif // NAM_MODEL_H")

        with open(filepath, 'w') as f:
            f.write("\n".join(lines))