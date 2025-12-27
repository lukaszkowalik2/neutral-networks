import numpy as np
import matplotlib.pyplot as plt
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
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def test_activations(X_train, X_test, Y_train, Y_test, X_val, Y_val):
    print("\n" + "="*50)
    print("TEST: Activation Functions")
    print("="*50)
    
    results = {}
    for activation in ['sigmoid', 'relu']:
        np.random.seed(42)
        mlp = MLP(
            input_size=784,
            hidden_size=128,
            output_size=10,
            learning_rate=0.01,
            num_epochs=50,
            activation_type=activation,
            verbose=False
        )
        print(f"\n{activation.upper()}:")
        mlp.train(X_train, Y_train, batch_size=32, X_val=X_val, y_val=Y_val)
        
        train_acc = np.mean(mlp.predict(X_train) == Y_train)
        test_acc = np.mean(mlp.predict(X_test) == Y_test)
        val_acc = np.mean(mlp.predict(X_val) == Y_val)
        
        print(f"  Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}, Val acc: {val_acc:.4f}")
        print(f"  Final loss: {mlp.history['loss'][-1]:.4f}")
        
        results[activation] = {'model': mlp, 'train_acc': train_acc, 'test_acc': test_acc, 'val_acc': val_acc}
    
    return results


def test_early_stopping(X_train, X_test, Y_train, Y_test, X_val, Y_val):
    print("\n" + "="*50)
    print("TEST: Early Stopping")
    print("="*50)
    
    results = {}
    
    # Without early stopping
    np.random.seed(42)
    mlp_no_es = MLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=0.01,
        num_epochs=200,
        activation_type='relu',
        early_stopping=False,
        verbose=False
    )
    print("\nWithout early stopping:")
    mlp_no_es.train(X_train, Y_train, batch_size=32, X_val=X_val, y_val=Y_val)
    train_acc = np.mean(mlp_no_es.predict(X_train) == Y_train)
    test_acc = np.mean(mlp_no_es.predict(X_test) == Y_test)
    val_acc = np.mean(mlp_no_es.predict(X_val) == Y_val)
    print(f"  Epochs: {mlp_no_es.history['final_epoch']}")
    print(f"  Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}, Val acc: {val_acc:.4f}")
    results['no_es'] = mlp_no_es
    
    # With early stopping
    np.random.seed(42)
    mlp_es = MLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        learning_rate=0.01,
        num_epochs=200,
        activation_type='relu',
        early_stopping=True,
        patience=5,
        min_delta=0.001,
        verbose=False
    )
    print("\nWith early stopping (patience=5):")
    mlp_es.train(X_train, Y_train, batch_size=32, X_val=X_val, y_val=Y_val)
    print(f"  Epochs: {mlp_es.history['final_epoch']}")
    print(f"  Stopped early: {mlp_es.history['stopped_early']}")
    print(f"  Train acc: {np.mean(mlp_es.predict(X_train) == Y_train):.4f}")
    print(f"  Test acc: {np.mean(mlp_es.predict(X_test) == Y_test):.4f}")
    print(f"  Val acc: {np.mean(mlp_es.predict(X_val) == Y_val):.4f}")
    results['es'] = mlp_es
    
    return results

def plot_results(act_results, es_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Activation comparison - train loss
    ax = axes[0, 0]
    for name, res in act_results.items():
        ax.plot(res['model'].history['loss'], label=f'{name} train')
        ax.plot(res['model'].history['val_loss'], label=f'{name} val', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Activation Functions - Train vs Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Early stopping - train vs val loss
    ax = axes[0, 1]
    ax.plot(es_results['no_es'].history['loss'], label='Train (no ES)', alpha=0.7)
    ax.plot(es_results['no_es'].history['val_loss'], label='Val (no ES)', alpha=0.7, linestyle='--')
    ax.plot(es_results['es'].history['loss'], label='Train (ES)', linewidth=2)
    ax.plot(es_results['es'].history['val_loss'], label='Val (ES)', linewidth=2, linestyle='--')
    if es_results['es'].history['stopped_early']:
        stop_epoch = es_results['es'].history['final_epoch']
        ax.axvline(x=stop_epoch-1, color='red', linestyle=':', label=f'Stopped at {stop_epoch}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Early Stopping Effect')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Overfitting gap
    ax = axes[1, 0]
    for name, res in act_results.items():
        train = res['model'].history['loss']
        val = res['model'].history['val_loss']
        gap = [v - t for t, v in zip(train, val)]
        ax.plot(gap, label=name)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss - Train Loss')
    ax.set_title('Overfitting Gap (higher = more overfitting)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final accuracy comparison
    ax = axes[1, 1]
    names = list(act_results.keys())
    x = np.arange(len(names))
    width = 0.25
    ax.bar(x - width, [act_results[n]['train_acc'] for n in names], width, label='Train')
    ax.bar(x, [act_results[n]['val_acc'] for n in names], width, label='Val')
    ax.bar(x + width, [act_results[n]['test_acc'] for n in names], width, label='Test')
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Accuracy by Activation')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('mlp_results.png', dpi=150)
    plt.show()
    print("\nPlot saved to mlp_results.png")


if __name__ == "__main__":
    np.random.seed(42)
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_mnist(subset_size=10000)
    
    act_results = test_activations(X_train, X_test, Y_train, Y_test, X_val, Y_val)
    es_results = test_early_stopping(X_train, X_test, Y_train, Y_test, X_val, Y_val)
    
    plot_results(act_results, es_results)
