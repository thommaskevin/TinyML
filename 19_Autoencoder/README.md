# TinyML - AutoEncoder

*From mathematical foundations to edge implementation*

**Social media:**
👨🏽‍💻 Github: [thommaskevin/TinyML](https://github.com/thommaskevin/TinyML)

👷🏾 Linkedin: [Thommas Kevin](https://www.linkedin.com/in/thommas-kevin-ab9810166/)

📽 Youtube: [Thommas Kevin](https://www.youtube.com/channel/UC7uazGXaMIE6MNkHg4ll9oA)

:pencil2:CV Lattes CNPq: [Thommas Kevin Sales Flores](http://lattes.cnpq.br/0630479458408181)

👨🏻‍🏫 Research group: [Conecta.ai](https://conect2ai.dca.ufrn.br/)



![Figure 1](/19_Autoencoder/figures/fig0.png)




## SUMMARY

1 - Introduction to AutoEncoder

 1.1 - Mathematical Foundations of Autoencoders
 
 1.2 - Loss Function and Optimization
 
 1.3 -The Training Process
 
 1.4 - Understanding the Latent Space
 
 1.5 - Visualizing the Autoencoder Process
 
 1.6  - The Role of Activation Functions
 
 1.7 -Types of Autoencoders
 
2 - TinyML Implementation


## 1 - Introduction to AutoEncoder

Autoencoders are a specific type of artificial neural network that are primarily used for learning efficient representations of data. These representations can be used for various tasks such as dimensionality reduction, data compression, and even for generating new data similar to the original input. Unlike supervised learning models that require labeled data, autoencoders are unsupervised, meaning they learn from input data alone without needing explicit output labels.


![Figure 1](/19_Autoencoder/figures/fig1.png)
Figure 1: AutoEncoder Architecture.



At their core, autoencoders are designed to compress the input data into a latent space representation and then reconstruct the input from this compressed form. The goal is for the reconstruction to be as close as possible to the original input. The idea is that by learning this compression and decompression process, the autoencoder will capture the most important features or patterns within the data.

An autoencoder consists of two main components:

- **Encoder**: The encoder is the part of the network responsible for compressing the input data into a lower-dimensional space. It does this by applying a series of transformations, typically linear or nonlinear, to reduce the dimensionality of the data. This compressed representation is known as the latent space or bottleneck. The latent space is a lower-dimensional representation of the input that ideally captures the most significant features of the data.

- **Decoder**: The decoder takes the latent space representation produced by the encoder and attempts to reconstruct the original input data from it. This reconstruction process is designed to be the inverse of the encoding process. The objective of the decoder is to produce an output that closely matches the original input data, thereby validating the quality of the latent space representation.


### 1.1 - Mathematical Foundations of Autoencoders


#### 1.1.1 - Input Data

Consider an input vector $\mathbf{x} \in \mathbb{R}^n$, where \( n \) represents the dimensionality of the input data. This vector \( \mathbf{x} \) can represent any kind of data, such as an image, a sound wave, or a set of numerical features.


#### 1.1.2 - Encoder Function

The encoder is a function \( f: \mathbb{R}^n \rightarrow \mathbb{R}^m \) that maps the input \( \mathbf{x} \) to a latent space vector \( \mathbf{z} \in \mathbb{R}^m \), where \( m \) is the dimensionality of the latent space, typically much smaller than \( n \). Mathematically, the encoder is represented as:


$\mathbf{z} = f(\mathbf{x}) = \sigma(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)$


In this equation:
- \( \mathbf{W}_e \) is the weight matrix associated with the encoder.
- \( \mathbf{b}_e \) is the bias vector for the encoder.
- \( \sigma \) is an activation function, such as ReLU, sigmoid, or tanh.



#### 1.1.3 -  Latent Space Representation

The vector \( \mathbf{z} \) is the compressed version of the input data. This latent space representation is crucial because it holds the encoded information that the decoder will use to reconstruct the original data. The dimensionality of \( \mathbf{z} \) (denoted as \( m \)) is usually smaller than that of the input \( \mathbf{x} \), enforcing the network to learn only the most critical features of the data.

#### 1.1.4 - Decoder Function

The decoder is a function \( g: \mathbb{R}^m \rightarrow \mathbb{R}^n \) that maps the latent space representation \( \mathbf{z} \) back to the original data space, attempting to reconstruct the input \( \mathbf{x} \). The decoder is mathematically represented as:

\[
\hat{\mathbf{x}} = g(\mathbf{z}) = \sigma(\mathbf{W}_d \mathbf{z} + \mathbf{b}_d)
\]

Where:
- \( \mathbf{W}_d \) is the weight matrix for the decoder.
- \( \mathbf{b}_d \) is the bias vector for the decoder.


The output \( \hat{\mathbf{x}} \) is the reconstructed version of the original input \( \mathbf{x} \). The goal of the decoder is to make \( \hat{\mathbf{x}} \) as close as possible to \( \mathbf{x} \).





### 1.2 - Loss Function and Optimization


The quality of the autoencoder's reconstruction is measured by a loss function, which quantifies the difference between the original input \( \mathbf{x} \) and the reconstructed output \( \hat{\mathbf{x}} \). A commonly used loss function for autoencoders is the Mean Squared Error (MSE), defined as:

\[
\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{x}_i)^2
\]

Where:
- \( n \) is the number of features in the input data.
- \( x_i \) and \( \hat{x}_i \) are the components of the original and reconstructed data vectors, respectively.

The objective during the training of an autoencoder is to minimize this loss function. Minimizing the loss function involves adjusting the weights \( \mathbf{W}_e \), \( \mathbf{W}_d \), and the biases \( \mathbf{b}_e \), \( \mathbf{b}_d \) in the network to achieve the smallest possible reconstruction error.



### 1.3 - The Training Process

During training, the autoencoder receives a set of input data, processes it through the encoder to produce the latent space representation, and then reconstructs the input using the decoder. The difference between the input and the reconstructed output is computed using the loss function. The network then backpropagates this error, adjusting the weights and biases to reduce the reconstruction error.


This process is repeated over many iterations, gradually improving the network's ability to encode and decode the data effectively. As the training progresses, the autoencoder becomes increasingly proficient at capturing the most significant features of the input data in the latent space.

### 1.4 - Understanding the Latent Space

The latent space z is a lower-dimensional representation of the original input data. It serves as a compressed encoding that preserves the most critical information required to reconstruct the input accurately. The dimensionality of the latent space (i.e., the size of z) is a crucial hyperparameter in designing an autoencoder. If the latent space is too large, the autoencoder may simply memorize the input data, failing to learn meaningful patterns (this is known as overfitting). Conversely, if the latent space is too small, the autoencoder may not capture enough information to reconstruct the input accurately.

The latent space is not just a mathematical abstraction; it has practical implications. For instance, in image processing tasks, the latent space might capture abstract features like shapes, edges, or textures. These features can then be manipulated to achieve various goals, such as generating new images, removing noise, or detecting anomalies.


### 1.5 - Visualizing the Autoencoder Process
Let’s visualize the autoencoder process with a simple example:

1. **Input Layer**: Imagine we have an input data vector \( \mathbf{x} \) that represents an image. Each element in \( \mathbf{x} \) corresponds to a pixel intensity value in the image.

2. **Hidden Layers (Encoder)**: The input is passed through several layers of neurons, where each layer performs a linear transformation followed by a non-linear activation. The data’s dimensionality is progressively reduced until it reaches the latent space \( \mathbf{z} \).

3. **Latent Space**: The latent space \( \mathbf{z} \) is a compact representation of the input image. It contains the essential features that the decoder will use to reconstruct the original image.

4. **Hidden Layers (Decoder)**: The latent representation \( \mathbf{z} \) is passed through another set of layers that progressively increase the dimensionality until the output layer is reached.

5. **Output Layer**: The final output \( \hat{\mathbf{x}} \) is the reconstructed version of the original input image. The network’s goal is for this output to be as close as possible to the original image.



### 1.6 - The Role of Activation Functions
Activation functions play a critical role in the autoencoder's ability to learn complex, non-linear representations. By introducing non-linearity through functions like ReLU, sigmoid, or tanh, the autoencoder can capture more complex patterns in the data that a purely linear transformation might miss. This non-linearity is particularly important in scenarios where the relationships between features in the data are non-linear, such as in image or speech processing.


### 1.7 - Types of Autoencoders

In practice, autoencoders can be extended and modified in various ways to improve their performance or adapt them to specific tasks. Some common extensions include:

#### 1.7.1 - Vanilla Autoencoder
The simplest form of autoencoder, often referred to as a "vanilla" autoencoder, consists of a symmetric network with equal numbers of neurons in the encoder and decoder layers. It is primarily used for dimensionality reduction.

#### 1.7.2 - Sparse Autoencoder
A sparse autoencoder includes an additional sparsity constraint on the latent representation. This encourages the model to learn a more compact and informative representation, where only a small number of neurons are active at any given time. This can be mathematically expressed by adding a sparsity penalty term \( \Omega(\mathbf{z}) \) to the loss function:

\[
\mathcal{L}_{\text{sparse}}(\mathbf{x}, \hat{\mathbf{x}}) = \mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) + \lambda \Omega(\mathbf{z})
\]

where \( \lambda \) controls the trade-off between reconstruction accuracy and sparsity.

#### 1.7.3 - Denoising Autoencoder
A denoising autoencoder is designed to reconstruct the original input from a corrupted version of it. During training, the input data is intentionally corrupted (e.g., by adding noise), and the autoencoder is tasked with recovering the clean input. The loss function remains the same, but the input \( \mathbf{x} \) is replaced with a corrupted version \( \mathbf{x}_{\text{noisy}} \).

#### 1.7.4 - Variational Autoencoder (VAE)
The Variational Autoencoder (VAE) is a probabilistic approach that models the latent space as a distribution, typically Gaussian. Instead of learning a fixed representation, the VAE learns the parameters of the distribution (mean and variance) from which the latent vector is sampled. The VAE optimizes a loss function that includes a reconstruction term and a regularization term based on the Kullback-Leibler (KL) divergence between the learned latent distribution and a prior distribution (typically standard normal):

\[
\mathcal{L}_{\text{VAE}}(\mathbf{x}, \hat{\mathbf{x}}) = \mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) + \text{KL}(q(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))
\]
















## 2 - TinyML Implementation

With this example you can implement the machine learning algorithm in ESP32, Arduino, Arduino Portenta H7 with Vision Shield, Raspberry and other different microcontrollers or IoT devices.


### 2.0 - Install the libraries listed in the requirements.txt file

```python
!pip install -r requirements.txt
```

### 2.1 - Importing libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses, Model
from eloquent_tensorflow import convert_model

import warnings
warnings.filterwarnings('ignore')
```

### 2.2 - Load Dataset

```python
(x_train, _), (x_test, _)=tf.keras.datasets.mnist.load_data()
```


### 2.3 - Splitting the data

```python
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)
```

![Figure 2](/19_Autoencoder/figures/fig2.png)


```python
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
```


### 2.4 - AutoEncoder


#### 2.4.1 - Create the Encoder Model

```python
reduction_size = 32

encoder = tf.keras.Sequential()
encoder.add(layers.Input(shape= x_train.shape[1:]))
encoder.add(layers.Dense(128, activation='relu'))
encoder.add(layers.Dense(64, activation='relu'))
encoder.add(layers.Dense(reduction_size, activation='relu'))  
encoder.summary()
```

![Figure 3](/19_Autoencoder/figures/fig3.png)



#### 2.4.2 - Create the Decoder Model

```python
decoder = tf.keras.Sequential()
decoder.add(layers.Input(shape=(reduction_size,)))
decoder.add(layers.Dense(64, activation='relu'))
decoder.add(layers.Dense(128, activation='relu'))
decoder.add(layers.Dense(x_train.shape[1:][0], activation='relu'))
decoder.summary()
```

![Figure 4](/19_Autoencoder/figures/fig4.png)


#### 2.4.3 - Create the AutoEncoder Model

```python
autoencoder = Model(inputs = encoder.input, outputs = decoder(encoder.output))
autoencoder.summary()
```

![Figure 5](/19_Autoencoder/figures/fig5.png)


### 2.5 - Train Model

```python
autoencoder.compile(optimizer='adamax', loss=losses.MeanSquaredError())
```


```python
history = autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))
```


```python
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r.', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.savefig('.\\figures\\history_traing.png', dpi=300, bbox_inches='tight')
plt.show()
```

![Figure 6](/19_Autoencoder/figures/fig6.png)



### 2.6 - Model evaluation


#### 2.6.1 - Evaluating the model with train data

```python
autoencoder.evaluate(x_train, x_train)
```

![Figure 7](/19_Autoencoder/figures/fig7.png)


#### 2.6.2 - Evaluating the model with test data

```python
autoencoder.evaluate(x_train, x_train)
```

![Figure 8](/19_Autoencoder/figures/fig8.png)


```python
encoded_imgs = encoder.predict(x_test) 
decoded_imgs = decoder.predict(encoded_imgs)
```


```python
n = 10  # Quantidade de dígitos para mostrar
plt.figure(figsize=(20, 6))
for i in range(n):
    # Imagens Originais
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Imagens Codificadas
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(8, 4))  # Ajustando para visualizar a codificação (8x4=32)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Imagens Reconstruídas
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('.\\figures\\autoencoder_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

![Figure 9](/19_Autoencoder/figures/fig9.png)



### 2.7 - Obtaining the model to be implemented in the microcontroller

#### 2.7.1 - Encoder

```python
code_encoder = convert_model(encoder)
```


#### 2.7.2 - Decoder

```python
code_decoder = convert_model(decoder)
```


#### 2.7.3 - AutoEncoder

```python
code_autoencoder = convert_model(autoencoder)
```


### 2.8 - Saves the template in a .h file

#### 2.8.1 - Encoder

```python
with open('./Autoencoder/model_encoder.h', 'w') as file:
    file.write(code_encoder)
```


#### 2.8.2 - Decoder

```python
with open('./Autoencoder/model_decoder.h', 'w') as file:
    file.write(code_decoder)
```


#### 2.8.3 - AutoEncoder

```python
with open('./Autoencoder/model_autoencoder.h', 'w') as file:
    file.write(code_autoencoder)
```


### 2.9 - Deploy Model
Import the libraries into your Arduino sketch under libraries -> ESP32 -> EloquentTinyML-main.zip and tflm_esp32–2.0.0.zip

#### 2.9.1 - Complete Arduino Sketch

```cpp
#include <Arduino.h>
// replace with your own model
//"model_encoder.h"
//"model_decoder.h"
//"model_autoencoder.h"
#include "model_autoencoder.h"
// include the runtime specific for your board
// either tflm_esp32 or tflm_cortexm
#include <tflm_esp32.h>
// now you can include the eloquent tinyml wrapper
#include <eloquent_tinyml.h>

// this is trial-and-error process
// when developing a new model, start with a high value
// (e.g. 10000), then decrease until the model stops
// working as expected
#define ARENA_SIZE 30000

Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf;

float X_1[32] = {...};
float expected_values[784] = {...};

void predictSample(float *input, float *expectedOutput) {

      while (!tf.begin(tfModel).isOk())
    {
        Serial.println(tf.exception.toString());
        delay(1000);
    }
  
    // classify class 0
    if (!tf.predict(input).isOk()) {
        Serial.println(tf.exception.toString());
        return;
    }

      for (int i = 0; i < 784; i++) {
        Serial.print("Expcted = ");
        Serial.print(expectedOutput[i]);
        Serial.print(", predicted = ");
        Serial.println(tf.outputs[i]);
        }

    
}

void setup()
{
    Serial.begin(115200);
    delay(3000);
    Serial.println("__TENSORFLOW AUTOENCODER DECODER__");

    // configure input/output
    // (not mandatory if you generated the .h model
    // using the eloquent_tensorflow Python package)
    tf.setNumInputs(TF_NUM_INPUTS);
    tf.setNumOutputs(TF_NUM_OUTPUTS);

    registerNetworkOps(tf);

    delay(500);

}

void loop()
{
    /**
     * Run prediction
     */
    
    predictSample(X_1,     expected_values);
    delay(2000);

}
```


### 2.10 - Results
 
#### 2.10.1 - Encoder

![Figure 10](/19_Autoencoder/figures/fig10.png)


#### 2.10.2 - Decoder

![Figure 11](/19_Autoencoder/figures/fig11.png)


#### 2.10.3 - AutoEncoder

![Figure 12](/19_Autoencoder/figures/fig12.png)

