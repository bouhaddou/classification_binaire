import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Fonction d'activation ReLU
def Relu(z):
    """
    La fonction ReLU produit le maximum entre son entrée et zéro.
    Si z < 0, retourne 0, sinon retourne z.
    
    Paramètres:
    - z : peut être un scalaire, un vecteur ou une matrice (par exemple, une entrée d'un neurone)
    
    Retour:
    - Un tableau numpy avec les activations, où toutes les valeurs inférieures à zéro sont remplacées par zéro.
    """
    return np.maximum(0, z)

# Dérivée de la fonction d'activation ReLU
def relu_derivative(z):
     """
    La dérivée de la fonction ReLU est utilisée lors de la rétropropagation d'un réseau neuronal.
    Si x > 0, la dérivée est 1, sinon 0.
    
    Paramètres:
    - x : peut être un scalaire, un vecteur ou une matrice
    
    Retour:
    - Un tableau numpy de la même forme que x avec des 1 où x > 0 et des 0 où x <= 0.
    """
    resultat = (z > 0).astype(z.dtype)
    assert np.all((resultat == 0) | (resultat == 1)), "resultat doit être 0 ou 1"
    return resultat

def sigmiode(z):
    """ la fonction sigmiode permet de transformer un nombre a une valeur entre 0 et 1    """
   
    assert isinstance(z, (int, float, list, np.ndarray)), "l'entrée doit être un nombre, une liste ou un tableau numpy"
    resultat = 1 / (1 + np.exp(-z))
    return resultat

def sigmiode_derivative(z):
    """
    une fonction d'activation (ou sa dérivée) à l'entrée z.
    
    Paramètres :
        * x : scalaire, liste ou np.array
        * fonction : nom de la fonction d'activations sigmoid,tanh,relu)
        * derivative : boolean par defaut est false , si True retourne la dérivée

    Retourn :
        Résultat de la fonction (ou sa dérivée)
    """
    assert isinstance(z, (int, float, list, np.ndarray)), "L'entrée doit être un nombre, une liste ou un tableau numpy."
    result = sigmiode(z) * (1 - sigmiode(z))
    assert np.all((result >= 0) & (result <= 0.25)), "resultat doit être dans l'intervalle [0, 0.25]"
    return result

# Classe NeuralNetwork
class NeuralNetwork:
    """ Initialize the neural network with given layer sizes and learning rate .
    50 layer_sizes : List of integers [ input_size , hidden1_size ,
    ... , output_size ]
    51 
    assert isinstance ( layer_sizes , list ) and len ( layer_sizes ) >= 2, " layer_sizes must be a list with at least 2 elements "
    assert all( isinstance (size , int ) and size > 0 for size in layer_sizes ), "All layer sizes must be positive integers " 
    assert isinstance ( (learning_rate , (int , float )) and learning_rate > 0), " Learning rate must be a positive number" 
    """
    def __init__(self, layer_sizes, learning_rate=0.01):
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2, "layer_sizes must be a list with at least 2 elements"
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes), "All layer sizes must be positive integers"
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, "Learning rate must be a positive number"

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            assert w.shape == (layer_sizes[i], layer_sizes[i + 1]), f"Weight matrix {i+1} has incorrect shape"
            assert b.shape == (1, layer_sizes[i + 1]), f"Bias vector {i+1} has incorrect shape"
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """
        Forward propagation : Z ^{[ l]} = A ^{[l -1]} W ^{[ l]} + b ^{[ l]}, A^{[ l]} = g(Z ^{[ l ]})
        """
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        self.activations = [X]
        self.z_values = []
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            assert z.shape == (X.shape[0], self.layer_sizes[i + 1]), f"Z^{i+1} has incorrect shape"
            A = Relu(z)
            self.activations.append(A)
            self.z_values.append(z)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        output = sigmiode(z)
        self.z_values.append(z)
        self.activations.append(output)
        return output

    def compute_loss(self, y_true, y_pred):
        """
        Binary Cross - Entropy : J = -1/m * sum (y * log ( y_pred ) + (1- y) * log (1- y_pred )) 
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to loss must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        assert np.all((y_true == 0) | (y_true == 1)), "y_true must contain only 0s and 1s"
        m = y_true.shape[0]
        loss = -(1 / m) * np.sum(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        assert not np.isnan(loss), "Loss computation resulted in NaN"
        return loss

    def compute_accuracy(self, y_true, y_pred):
        """
        Compute accuracy : proportion of correct predictions
        """
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to accuracy must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        preds = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(preds == y_true)
        assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"
        return accuracy

    def backward(self, X, y, outputs):
        """
        Backpropagation : compute dW ^{[l]}, db ^{[ l]} for each layer
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and isinstance(outputs, np.ndarray), "Inputs to backward must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape == outputs.shape, "y and outputs must have the same shape"
        m = X.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]
        dZ = outputs - y
        assert dZ.shape == outputs.shape, "dZ for output layer has incorrect shape"
        self.d_weights[-1] = self.activations[-2].T @ dZ / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        for i in range(len(self.weights) - 2, -1, -1):
            dA = dZ @ self.weights[i + 1].T
            dZ = dA * relu_derivative(self.z_values[i])
            self.d_weights[i] = self.activations[i].T @ dZ / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - self.learning_rate * self.d_weights[i]
            self.biases[i] = self.biases[i] - self.learning_rate * self.d_biases[i]

    def train(self, X, y, X_val, y_val, epochs, batch_size):
        """
         Train the neural network using mini - batch SGD , with validation
        """
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays"
        assert isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray), "X_val and y_val must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape[1] == self.layer_sizes[-1], f"Output dimension ({y.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert X_val.shape[1] == self.layer_sizes[0], f"Validation input dimension ({X_val.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y_val.shape[1] == self.layer_sizes[-1], f"Validation output dimension ({y_val.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert isinstance(epochs, int) and epochs > 0, "Epochs must be a positive integer"
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size must be a positive integer"
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X, y = X[indices], y[indices]
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                outputs = self.forward(X_batch)
                epoch_loss += self.compute_loss(y_batch, outputs)
                self.backward(X_batch, y_batch, outputs)
            y_train_pred = self.forward(X)
            y_val_pred = self.forward(X_val)
            train_loss = self.compute_loss(y, y_train_pred)
            val_loss = self.compute_loss(y_val, y_val_pred)
            train_accuracy = self.compute_accuracy(y, y_train_pred)
            val_accuracy = self.compute_accuracy(y_val, y_val_pred)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        """ Predict class labels (0 or 1) """
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        output = self.forward(X)
        predictions = (output >= 0.5).astype(int)
        assert predictions.shape == (X.shape[0], self.layer_sizes[-1]), "Predictions have incorrect shape"
        return predictions

# Chargement et préparation des données
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values.reshape(-1, 1)
assert X.shape[0] == y.shape[0], "Number of samples in X and y must match"
assert X.shape[1] == 8, "Expected 8 features in input data"
X = (X - X.mean(axis=0)) / X.std(axis=0)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == X.shape[0], "Train-val-test split sizes must sum to total samples"

# Création et entraînement du modèle
layer_sizes = [8, 16, 8, 1]
nn = NeuralNetwork(layer_sizes, learning_rate=0.01)
train_losses, val_losses, train_accuracies, val_accuracies = nn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

# Prédictions et évaluation
y_pred = nn.predict(X_test)
print("\nRapport de classification (Test set) :")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion")
print(cm)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="train loss")
plt.plot(val_losses, label="val loss")
plt.title("Courbe de perte")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="train accuracies")
plt.plot(val_accuracies, label="val accuracies")
plt.title("Courbe d'accuracies")
plt.xlabel("Époque")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
