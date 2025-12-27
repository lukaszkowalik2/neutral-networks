import numpy as np

class Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, 1, 0)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, num_epochs=100, activation_type='sigmoid', verbose=False, early_stopping=False, patience=10, min_delta=0.001):
        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.hidden_bias = np.zeros(hidden_size)
        self.output_weights = np.random.randn(hidden_size, output_size)
        self.output_bias = np.zeros(output_size)

        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.activation = self.__get_activation(activation_type)
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta

    def __get_activation(self, activation_type):
        if activation_type == 'sigmoid':
            return Sigmoid()
        elif activation_type == 'relu':
            return ReLU()
        else:
            raise ValueError(f"Activation type {activation_type} not supported")

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.hidden_weights) + self.hidden_bias
        self.hidden_output = self.activation.forward(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.output_weights) + self.output_bias
        self.output = self.softmax(self.output_input)
        return self.output

    def backward(self, X, y, output):
        output_error = output - y
        hidden_error = np.dot(output_error, self.output_weights.T) * self.activation.backward(self.hidden_input)

        self.output_weights -= self.learning_rate * np.dot(self.hidden_output.T, output_error)
        self.output_bias -= self.learning_rate * np.sum(output_error, axis=0)
        self.hidden_weights -= self.learning_rate * np.dot(X.T, hidden_error)
        self.hidden_bias -= self.learning_rate * np.sum(hidden_error, axis=0)
        
    def train(self, X, y, batch_size=32, X_val=None, y_val=None):
        y_onehot = np.zeros((len(y), self.output_size))
        y_onehot[np.arange(len(y)), y] = 1
        
        if y_val is not None:
            y_val_onehot = np.zeros((len(y_val), self.output_size))
            y_val_onehot[np.arange(len(y_val)), y_val] = 1

        best_loss = float('inf')
        patience_counter = 0
        self.history = {'loss': [], 'val_loss': [], 'stopped_early': False, 'final_epoch': 0}

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y_onehot[i:i+batch_size]
                output_batch = self.forward(X_batch)
                
                epoch_loss += -np.sum(y_batch * np.log(output_batch + 1e-8))
                self.backward(X_batch, y_batch, output_batch)

            epoch_loss /= len(X)
            self.history['loss'].append(epoch_loss)
            self.history['final_epoch'] = epoch + 1
            
            # Compute validation loss
            val_loss = None
            if X_val is not None:
                val_output = self.forward(X_val)
                val_loss = -np.sum(y_val_onehot * np.log(val_output + 1e-8)) / len(X_val)
                self.history['val_loss'].append(val_loss)

            if self.verbose:
                msg = f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

            # Early stopping based on validation loss (or train loss if no val data)
            if self.early_stopping:
                monitor_loss = val_loss if val_loss is not None else epoch_loss
                if monitor_loss < best_loss - self.min_delta:
                    best_loss = monitor_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self.history['stopped_early'] = True
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == np.argmax(y, axis=1))