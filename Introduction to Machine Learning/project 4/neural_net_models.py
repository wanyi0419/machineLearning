import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        """
        Initialize a Logistic Regression model.
        - The model uses softmax function for prediction

        Parameters:
        - learning_rate (float): The learning rate used for gradient descent (default: 0.01).
        - num_epochs (int): The number of training epochs (default: 1000).
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def softmax(self, z):
        """
        Compute the softmax function for a given input vector 'z'.

        Parameters:
        - z (numpy.ndarray): Input vector.

        Returns:
        - numpy.ndarray: Softmax probabilities.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # To avoid overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        """
        Train the Logistic Regression model on the given input data and labels.

        Parameters:
        - X (numpy.ndarray): Input data matrix with shape (num_samples, num_features).
        - y (numpy.ndarray): Target labels with shape (num_samples,).

        Returns:
        - None
        """
        # Initialize weights and bias based on the input data and number of classes
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        self.weights = np.random.randn(num_features, num_classes)
        self.bias = np.zeros((1, num_classes))

        # Convert 'y' to one-hot encoding
        y_one_hot = np.zeros((num_samples, num_classes))
        y_one_hot[np.arange(num_samples), y] = 1

        # Gradient descent for 'num_epochs'
        for _ in range(self.num_epochs):
            # Compute the linear model (scores)
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(linear_model)

            # Calculate gradients of weights and bias
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y_one_hot))
            db = (1 / num_samples) * np.sum(y_pred - y_one_hot, axis=0)

            # Update weights and bias using the gradients
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Make predictions for the input data using the trained Logistic Regression model.

        Parameters:
        - X (numpy.ndarray): Input data matrix with shape (num_samples, num_features).

        Returns:
        - numpy.ndarray: Predicted class labels.
        """
        # Compute the linear model (scores) for predictions
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(linear_model)
        
        # Get the class with the highest probability as the predicted class
        y_pred_cls = np.argmax(y_pred, axis=1)
        return y_pred_cls

    def update(self, params):
        self.learning_rate = params.get('learning_rate')

class LinearNetwork:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        """
        Initialize a Linear Network (Ordinary Least Squares Regression) model.

        Parameters:
        - learning_rate (float): The learning rate used for gradient descent (default: 0.01).
        - num_epochs (int): The number of training epochs (default: 1000).
        """
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the Linear Network (Ordinary Least Squares Regression) model on the given input data and labels.

        Parameters:
        - X (numpy.ndarray): Input data matrix with shape (num_samples, num_features).
        - y (numpy.ndarray): Target labels with shape (num_samples,).

        Returns:
        - None
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for i in range(self.num_epochs):
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update parameters using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Make predictions for the input data using the trained Linear Network (Ordinary Least Squares Regression) model.

        Parameters:
        - X (numpy.ndarray): Input data matrix with shape (num_samples, num_features).

        Returns:
        - numpy.ndarray: Predicted values.
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def update(self, params):
        self.learning_rate = params.get('learning_rate')

"""
===========
Helper functions for NeuralNet Class
===========
"""

# Activation functions

def tanh(x):
    """
    Compute the hyperbolic tangent (tanh) activation function for a given input.

    Parameters:
    - x (numpy.ndarray): Input data.

    Returns:
    - numpy.ndarray: Output after applying the tanh activation function.
    """
    return np.tanh(x)

def d_tanh(x):
    """
    Compute the derivative of the hyperbolic tangent (tanh) activation function.

    Parameters:
    - x (numpy.ndarray): Input data.

    Returns:
    - numpy.ndarray: Derivative of the tanh activation function.
    """
    return 1 - np.square(np.tanh(x))

def sigmoid(x):
    """
    Compute the sigmoid activation function for a given input.

    Parameters:
    - x (numpy.ndarray): Input data.

    Returns:
    - numpy.ndarray: Output after applying the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    """
    Compute the derivative of the sigmoid activation function.

    Parameters:
    - x (numpy.ndarray): Input data.

    Returns:
    - numpy.ndarray: Derivative of the sigmoid activation function.
    """
    return (1 - sigmoid(x)) * sigmoid(x)


# Loss Functions 

def mean_squared_error(y, a):
    """
    Compute the mean squared error (MSE) loss between true labels 'y' and predicted values 'a'.

    Parameters:
    - y (numpy.ndarray): True labels.
    - a (numpy.ndarray): Predicted values.

    Returns:
    - float: Mean squared error loss.
    """
    return np.mean((y - a) ** 2)

def d_mean_squared_error(y, a):
    """
    Compute the derivative of the mean squared error (MSE) loss with respect to predicted values 'a'.

    Parameters:
    - y (numpy.ndarray): True labels.
    - a (numpy.ndarray): Predicted values.

    Returns:
    - float: Derivative of the MSE loss with respect to 'a'.
    """
    return np.mean(-2 * (y - a))

def cross_entropy_loss(y_pred, y_true):
    """
    Compute the cross-entropy loss between predicted probabilities 'y_pred' and true label distribution 'y_true'.

    Parameters:
    - y_pred (numpy.ndarray): Predicted probabilities.
    - y_true (numpy.ndarray): True label distribution (one-hot encoded).

    Returns:
    - float: Cross-entropy loss.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred))
    return loss

def d_cross_entropy_loss(y_pred, y_true):
    """
    Compute the derivative of the cross-entropy loss with respect to y_pred.

    Parameters:
    - y_pred (numpy.ndarray): Predicted probability distribution (e.g., softmax output).
    - y_true (numpy.ndarray): True label distribution (one-hot encoded).

    Returns:
    - numpy.ndarray: Derivative of the loss with respect to y_pred.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    derivative = -np.mean(y_true / y_pred)
    return derivative



class NeuralNetwork:
    # Dictionary of loss functions and their derivatives for different data types
    lossFunctions = {
        'regression': (mean_squared_error, d_mean_squared_error),
        'classification': (cross_entropy_loss, d_cross_entropy_loss)
    }

    def __init__(self, num_neuron=1, num_epochs=100, activation='sigmoid', learning_rate=0.01, dataType='classification', autoencoder=False):
        """
        Initialize a Neural Network model.

        Parameters:
        - num_epochs (int): The number of training epochs (default: 100).
        - activation (str): Activation function type ('sigmoid' or 'tanh', default: 'sigmoid').
        - learning_rate (float): The learning rate used for gradient descent (default: 0.01).
        - dataType (str): Type of data ('classification' or 'regression', default: 'classification').
        - autoencoder (bool): Whether the network is used for autoencoder (default: False).
        """
        self.learning_rate = learning_rate
        self.classification = (dataType == 'classification')
        self.num_epochs = num_epochs
        self.num_neuron = num_neuron
        self.activation = activation
        self.loss, self.d_loss = self.lossFunctions.get(dataType)
        self.autoencoder = autoencoder

    def update(self, params):
        self.learning_rate = params.get('learning_rate')
        self.num_neuron = params.get('num_neurons')

    def softmax(self, x):
        """
        Compute the softmax function for a given input vector 'x'.

        Parameters:
        - x (numpy.ndarray): Input vector.

        Returns:
        - numpy.ndarray: Softmax probabilities.
        """
        exp_x = np.exp(x - np.max(x))  # To avoid overflow
        return exp_x / np.sum(exp_x)

    def fit(self, X, y):
        """
        Train the Neural Network model on the given input data and labels.

        Parameters:
        - X (numpy.ndarray): Input data matrix with shape (num_samples, num_features).
        - y (numpy.ndarray): Target labels or values with shape (num_samples, num_classes).

        Returns:
        - None
        """
        X = X.T
        y = y.T
        num_features, num_samples = X.shape
        num_classes = len(np.unique(y))

        # Initialize layers
        self.layers = []

        if self.classification:
            num_classes = len(np.unique(y))
            # Convert y to one-hot encoding
            y_one_hot = np.zeros((num_samples, num_classes))
            y_one_hot[np.arange(num_samples), y.T] = 1
            y = y_one_hot.T
            if self.autoencoder:
                self.layers.append(AutoencoderLayer(num_features, num_classes, self.activation, self.learning_rate))
                self.layers.append(Layer(num_classes, num_classes, self.activation, self.learning_rate))
            else:
                self.layers.append(Layer(num_features, num_classes, self.activation, self.learning_rate))
                self.layers.append(Layer(num_classes, num_classes, self.activation, self.learning_rate))
        else:
            if self.autoencoder:
                self.layers.append(AutoencoderLayer(num_features, self.num_neuron, self.activation, self.learning_rate))
                self.layers.append(Layer(self.num_neuron, 1, self.activation, self.learning_rate))
            else:
                self.layers.append(Layer(num_features, self.num_neuron, self.activation, self.learning_rate))
                self.layers.append(Layer(self.num_neuron, 1, self.activation, self.learning_rate))

        for epoch in range(self.num_epochs):
            # Feedforward
            A = X
            for layer in self.layers:
                A = layer.feedforward(A)

            # Compute loss and accuracy (for classification)
            loss = self.loss(A, y)

            """
            if epoch%100 == 0:
                if self.classification:
                    accuracy = self.compute_accuracy(A, y)
                    print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%')
                else:
                    print(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss:.4f}')
            """
            # Backpropagation
            dA = self.d_loss(y, A)
            for layer in reversed(self.layers):
                dA = layer.backprop(dA)
            

    def compute_accuracy(self, y_pred, y_true):
        """
        Compute the accuracy of classification predictions.

        Parameters:
        - y_pred (numpy.ndarray): Predicted probabilities.
        - y_true (numpy.ndarray): True label distribution (one-hot encoded).

        Returns:
        - float: Classification accuracy.
        """
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_true, axis=1)
        accuracy = np.mean(y_pred_class == y_true_class) * 100
        return accuracy

    def predict(self, X):
        """
        Make predictions using the trained Neural Network model.

        Parameters:
        - X (numpy.ndarray): Input data matrix with shape (num_samples, num_features).

        Returns:
        - numpy.ndarray: Predicted class labels or values.
        """
        X = X.T
        A = X
        for layer in self.layers:
            A = layer.feedforward(A)
        if self.classification:
            prob = self.softmax(A)
            A = np.argmax(prob, axis=0)
        return A


class Layer:
    # Dictionary of activation functions and their derivatives
    activationFunctions = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid)
    }

    def __init__(self, inputs, neurons, activation='sigmoid', learning_rate=0.01):
        """
        Initialize a neural network layer.

        Parameters:
        - inputs (int): Number of input features.
        - neurons (int): Number of neurons in the layer.
        - activation (str): Activation function type ('sigmoid' or 'tanh', default: 'sigmoid').
        - learning_rate (float): The learning rate used for gradient descent (default: 0.01).
        """
        self.W = np.random.randn(neurons, inputs)
        self.b = np.zeros((neurons, 1))
        self.act, self.d_act = self.activationFunctions.get(activation)
        self.learning_rate = learning_rate

    def feedforward(self, A_prev):
        """
        Perform the feedforward computation in the layer.

        Parameters:
        - A_prev (numpy.ndarray): Input activations from the previous layer.

        Returns:
        - numpy.ndarray: Output activations after applying the activation function.
        """
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.act(self.Z)
        return self.A

    def backprop(self, dA):
        """
        Perform backpropagation to update weights and propagate gradients to the previous layer.

        Parameters:
        - dA (numpy.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
        - numpy.ndarray: Gradient of the loss with respect to the layer's input activations.
        """
        dZ = np.multiply(self.d_act(self.Z), dA)
        dW = 1/dZ.shape[1] * np.dot(dZ, self.A_prev.T)
        db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return dA_prev


class AutoencoderLayer:
    # Dictionary of activation functions and their derivatives
    activationFunctions = {
        'tanh': (tanh, d_tanh),
        'sigmoid': (sigmoid, d_sigmoid)
    }

    def __init__(self, input_size, encoding_size, activation='sigmoid', learning_rate=0.01):
        """
        Initialize an Autoencoder layer.

        Parameters:
        - input_size (int): Number of input features.
        - encoding_size (int): Number of neurons in the encoding layer.
        - activation (str): Activation function type ('sigmoid' or 'tanh', default: 'sigmoid').
        - learning_rate (float): The learning rate used for gradient descent (default: 0.01).
        """
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.activation = activation
        self.learning_rate = learning_rate

        # Encoder weights and biases
        self.encoder_W = np.random.randn(encoding_size, input_size)
        self.encoder_b = np.zeros((encoding_size, 1))

        # Decoder weights and biases
        self.decoder_W = np.random.randn(input_size, encoding_size)
        self.decoder_b = np.zeros((input_size, 1))

        self.encoder_act, self.encoder_d_act = self.activationFunctions.get(activation)

    def encode(self, X):
        """
        Encode input data using the encoder.

        Parameters:
        - X (numpy.ndarray): Input data to be encoded.

        Returns:
        - numpy.ndarray: Encoded representation of the input data.
        """
        # Encoder feedforward
        self.encoder_input = X
        self.encoder_Z = np.dot(self.encoder_W, self.encoder_input) + self.encoder_b
        self.encoder_output = self.encoder_act(self.encoder_Z)
        return self.encoder_output

    def decode(self, encoded_data):
        """
        Decode encoded data using the decoder.

        Parameters:
        - encoded_data (numpy.ndarray): Encoded data to be decoded.

        Returns:
        - numpy.ndarray: Decoded representation of the encoded data.
        """
        # Decoder feedforward
        self.decoder_input = encoded_data
        self.decoder_Z = np.dot(self.decoder_W, self.decoder_input) + self.decoder_b
        self.decoder_output = self.encoder_act(self.decoder_Z)  # Same activation as encoder
        return self.decoder_output

    def feedforward(self, X):
        """
        Perform a full feedforward pass through the autoencoder.

        Parameters:
        - X (numpy.ndarray): Input data to be encoded and decoded.

        Returns:
        - numpy.ndarray: Encoded representation and decoded output.
        """
        # Pass the input through the encoder
        encoded = self.encode(X)
        # Pass the encoded representation through the decoder
        decoded = self.decode(encoded)
        # Return both the encoded representation and the reconstructed output
        return encoded

    def backprop(self, dA):
        """
        Perform backpropagation to update weights and propagate gradients through the autoencoder.

        Parameters:
        - dA (numpy.ndarray): Gradient of the loss with respect to the autoencoder's output.

        Returns:
        - numpy.ndarray: Gradient of the loss with respect to the autoencoder's input.
        """
        # Backpropagation in decoder
        dZ = np.dot(self.encoder_d_act(self.decoder_Z), np.mean(dA))
        dW_decoder = 1 / dZ.shape[1] * np.dot(dZ, self.decoder_input.T)
        db_decoder = 1 / dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_decoder_prev = np.dot(self.decoder_W.T, dZ)

        # Backpropagation in encoder
        dZ = np.multiply(self.encoder_d_act(self.encoder_Z), dA_decoder_prev)
        dW_encoder = 1 / dZ.shape[1] * np.dot(dZ, self.encoder_input.T)
        db_encoder = 1 / dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        dA_encoder_prev = np.dot(self.encoder_W.T, dZ)

        # Update weights and biases in both encoder and decoder
        self.encoder_W -= self.learning_rate * dW_encoder
        self.encoder_b -= self.learning_rate * db_encoder
        self.decoder_W -= self.learning_rate * dW_decoder
        self.decoder_b -= self.learning_rate * db_decoder

        return dA_encoder_prev
