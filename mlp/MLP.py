import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, num_epochs=100):
        self.hidden_weights = np.random.randn(input_size, hidden_size)
        self.hidden_bias = np.zeros(hidden_size)
        self.output_weights = np.random.randn(hidden_size, output_size)
        self.output_bias = np.zeros(output_size)

        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.hidden_weights) + self.hidden_bias
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.output_weights) + self.output_bias
        self.output = self.softmax(self.output_input)
        return self.output

    def backward(self, X, y, output):
        output_error = output - y
        hidden_error = np.dot(output_error, self.output_weights.T) * self.sigmoid_derivative(self.hidden_input)

        self.output_weights -= self.learning_rate * np.dot(self.hidden_output.T, output_error)
        self.output_bias -= self.learning_rate * np.sum(output_error, axis=0)
        self.hidden_weights -= self.learning_rate * np.dot(X.T, hidden_error)
        self.hidden_bias -= self.learning_rate * np.sum(hidden_error, axis=0)
        
    def train(self, X, y, batch_size=32):
        y_onehot = np.zeros((len(y), self.output_size))
        y_onehot[np.arange(len(y)), y] = 1

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y_onehot[i:i+batch_size]
                output_batch = self.forward(X_batch)
                
                # Cross-entropy loss
                epoch_loss += -np.sum(y_batch * np.log(output_batch + 1e-8))
                self.backward(X_batch, y_batch, output_batch)

            epoch_loss /= len(X)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")


    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def evaluate(self, X, y):
        return np.mean(self.predict(X) == np.argmax(y, axis=1))