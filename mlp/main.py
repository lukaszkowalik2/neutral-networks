import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from MLP import MLP


def load_mnist(subset_size=10000):
    print("Loading MNIST...")
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = x / 255.0
    Y = y.astype(int)
    
    if subset_size and subset_size < len(X):
        idx = np.random.choice(len(X), subset_size, replace=False)
        X, Y = X[idx], Y[idx]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    np.random.seed(42)
    
    X_train, X_test, Y_train, Y_test = load_mnist(subset_size=5000)
    
    mlp = MLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=0.01,
        num_epochs=20
    )
    
    print("\nTraining...")
    mlp.train(X_train, Y_train, batch_size=32)
    
    train_acc = np.mean(mlp.predict(X_train) == Y_train)
    test_acc = np.mean(mlp.predict(X_test) == Y_test)
    
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
