import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class HybridLoss:
    """
    Implements a hybrid loss function combining Softmax and Hinge loss.
    """

    @staticmethod
    def softmax_loss(scores: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute softmax (cross-entropy) loss and gradient.
        """
        N, _ = scores.shape
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        eps = 1e-12
        loss = -np.mean(np.log(probs[np.arange(N), y] + eps))
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N
        return loss, dscores

    @staticmethod
    def hinge_loss(
        scores: np.ndarray,
        y: np.ndarray,
        margin: float = 1.0,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute multi-class SVM (hinge) loss and gradient.
        """
        N, _ = scores.shape
        correct_scores = scores[np.arange(N), y].reshape(N, 1)
        margins = np.maximum(0, scores - correct_scores + margin)
        margins[np.arange(N), y] = 0
        loss = np.mean(np.sum(margins, axis=1))
        dscores = np.zeros_like(scores, dtype=np.float64)
        mask = margins > 0
        dscores[mask] = 1.0
        violations = np.sum(mask, axis=1)
        dscores[np.arange(N), y] = -violations.astype(np.float64)
        dscores /= N
        return loss, dscores

    @staticmethod
    def hybrid_loss_forward(
        scores: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.5,
        beta: float = 0.5,
        margin: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        softmax_loss_val, _ = HybridLoss.softmax_loss(scores, y)
        hinge_loss_val, _ = HybridLoss.hinge_loss(scores, y, margin)
        total_loss = alpha * softmax_loss_val + beta * hinge_loss_val
        return total_loss, {"softmax": softmax_loss_val, "hinge": hinge_loss_val}

    @staticmethod
    def hybrid_loss_backward(
        scores: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.5,
        beta: float = 0.5,
        margin: float = 1.0,
    ) -> np.ndarray:
        _, ds_soft = HybridLoss.softmax_loss(scores, y)
        _, ds_hinge = HybridLoss.hinge_loss(scores, y, margin)
        return alpha * ds_soft + beta * ds_hinge


class RegularizedNetwork:
    """
    Two-layer neural network with various regularization options.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        weight_scale: float = 1e-3,
    ):
        self.params = {
            "W1": weight_scale * np.random.randn(input_dim, hidden_dim),
            "b1": np.zeros(hidden_dim),
            "W2": weight_scale * np.random.randn(hidden_dim, output_dim),
            "b2": np.zeros(output_dim),
        }

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        z1 = X.dot(W1) + b1
        h1 = np.maximum(0, z1)
        scores = h1.dot(W2) + b2
        cache = {"X": X, "z1": z1, "h1": h1, "W1": W1, "W2": W2, "b1": b1, "b2": b2}
        return scores, cache

    def backward(self, dscores: np.ndarray, cache: Dict) -> Dict[str, np.ndarray]:
        X = cache["X"]
        h1 = cache["h1"]
        z1 = cache["z1"]
        W2 = cache["W2"]
        dW2 = h1.T.dot(dscores)
        db2 = np.sum(dscores, axis=0)
        dh1 = dscores.dot(W2.T)
        dz1 = dh1 * (z1 > 0)
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def update_params(self, grads: Dict[str, np.ndarray], learning_rate: float) -> None:
        for key in self.params:
            self.params[key] -= learning_rate * grads[key]

    def accuracy(self, scores: np.ndarray, y: np.ndarray) -> float:
        preds = np.argmax(scores, axis=1)
        return np.mean(preds == y)

    def compute_loss_and_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reg_type: str = "l2",
        reg_strength: float = 0.0,
        loss_type: str = "hybrid",
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> Tuple[float, Dict]:
        scores, cache = self.forward(X)
        if loss_type == "softmax":
            loss, dscores = HybridLoss.softmax_loss(scores, y)
        elif loss_type == "hinge":
            loss, dscores = HybridLoss.hinge_loss(scores, y)
        else:
            loss, _ = HybridLoss.hybrid_loss_forward(scores, y, alpha, beta)
            dscores = HybridLoss.hybrid_loss_backward(scores, y, alpha, beta)

        W1, W2 = self.params["W1"], self.params["W2"]
        reg_loss = 0.0
        reg_grad = {"W1": np.zeros_like(W1), "W2": np.zeros_like(W2)}

        if reg_type == "l1":
            loss1, reg_grad["W1"] = l1_regularization(W1, reg_strength)
            loss2, reg_grad["W2"] = l1_regularization(W2, reg_strength)
            reg_loss = loss1 + loss2
        elif reg_type == "l2":
            loss1, reg_grad["W1"] = l2_regularization(W1, reg_strength)
            loss2, reg_grad["W2"] = l2_regularization(W2, reg_strength)
            reg_loss = loss1 + loss2
        elif reg_type == "elastic":
            l1_strength, l2_strength = reg_strength / 2, reg_strength / 2
            loss1, reg_grad["W1"] = elastic_net_regularization(
                W1, l1_strength, l2_strength
            )
            loss2, reg_grad["W2"] = elastic_net_regularization(
                W2, l1_strength, l2_strength
            )
            reg_loss = loss1 + loss2

        loss += reg_loss
        grads = self.backward(dscores, cache)
        grads["W1"] += reg_grad["W1"]
        grads["W2"] += reg_grad["W2"]
        return loss, grads


def l1_regularization(W: np.ndarray, reg_strength: float) -> Tuple[float, np.ndarray]:
    loss = reg_strength * np.sum(np.abs(W))
    dW = reg_strength * np.sign(W)
    dW[W == 0] = 0.0
    return loss, dW


def l2_regularization(W: np.ndarray, reg_strength: float) -> Tuple[float, np.ndarray]:
    loss = reg_strength * np.sum(W * W)
    dW = 2 * reg_strength * W
    return loss, dW


def elastic_net_regularization(
    W: np.ndarray, l1_strength: float, l2_strength: float
) -> Tuple[float, np.ndarray]:
    l1_loss, l1_grad = l1_regularization(W, l1_strength)
    l2_loss, l2_grad = l2_regularization(W, l2_strength)
    return l1_loss + l2_loss, l1_grad + l2_grad


def numerical_gradient(f, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        x[idx] = old_value + h
        fxh1 = f(x.copy())
        x[idx] = old_value - h
        fxh2 = f(x.copy())
        grad[idx] = (fxh1 - fxh2) / (2.0 * h)
        x[idx] = old_value
        it.iternext()
    return grad


def check_gradients(scores: np.ndarray, y: np.ndarray, alpha: float = 0.5, beta: float = 0.5) -> Dict[str, float]:
    analytical = HybridLoss.hybrid_loss_backward(scores, y, alpha, beta)
    numerical = numerical_gradient(
        lambda s: HybridLoss.hybrid_loss_forward(s, y, alpha, beta)[0],
        scores.copy(),
    )
    return {
        "max_diff": float(np.max(np.abs(analytical - numerical))),
        "avg_diff": float(np.mean(np.abs(analytical - numerical))),
    }


def train_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dim: int = 100,
    learning_rate: float = 1e-3,
    reg_type: str = "l2",
    reg_strength: float = 0.01,
    loss_type: str = "hybrid",
    num_epochs: int = 100,
    batch_size: int = 200,
    alpha: float = 0.5,
    beta: float = 0.5,
    verbose: bool = True,
) -> Tuple[RegularizedNetwork, Dict]:
    input_dim = X_train.shape[1]
    output_dim = np.max(y_train) + 1
    net = RegularizedNetwork(input_dim, hidden_dim, output_dim)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(num_epochs):
        idx = np.random.permutation(len(X_train))
        X_perm, y_perm = X_train[idx], y_train[idx]
        for i in range(0, len(X_perm), batch_size):
            Xb = X_perm[i : i + batch_size]
            yb = y_perm[i : i + batch_size]
            loss, grads = net.compute_loss_and_gradients(
                Xb,
                yb,
                reg_type,
                reg_strength,
                loss_type,
                alpha,
                beta,
            )
            net.update_params(grads, learning_rate)

        train_scores, _ = net.forward(X_perm)
        val_scores, _ = net.forward(X_val)
        train_loss, _ = net.compute_loss_and_gradients(
            X_perm, y_perm, reg_type, reg_strength, loss_type, alpha, beta
        )
        val_loss, _ = net.compute_loss_and_gradients(
            X_val, y_val, reg_type, reg_strength, loss_type, alpha, beta
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(net.accuracy(train_scores, y_perm))
        history["val_acc"].append(net.accuracy(val_scores, y_val))

        if verbose:
            print(
                f"Epoch {ep+1}/{num_epochs} | "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | "
                f"Train Acc={history['train_acc'][-1]:.3f}, Val Acc={history['val_acc'][-1]:.3f}"
            )

    return net, history


def analyze_weight_distribution(net: RegularizedNetwork, reg_type: str) -> Dict[str, np.ndarray]:
    W1, W2 = net.params["W1"], net.params["W2"]
    all_weights = np.concatenate([W1.flatten(), W2.flatten()])
    hist_counts, hist_bins = np.histogram(all_weights, bins=10)
    return {
        "sparsity_percent": np.mean(np.abs(all_weights) < 0.01),
        "l1_norm": np.sum(np.abs(all_weights)),
        "l2_norm": np.sqrt(np.sum(all_weights**2)),
        "mean": np.mean(all_weights),
        "std": np.std(all_weights),
        "histogram": (hist_counts, hist_bins),
    }


def compare_regularizations(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    reg_types: List[str] = None,
    reg_strength: float = 0.01,
    train_kwargs: Dict = None,
) -> Dict:
    if reg_types is None:
        reg_types = ["none", "l1", "l2", "elastic"]
    if train_kwargs is None:
        train_kwargs = {}
    results = {}
    for reg_type in reg_types:
        print(f"\nTraining with {reg_type} regularization...")
        net, history = train_network(
            X_train,
            y_train,
            X_val,
            y_val,
            reg_type=reg_type,
            reg_strength=reg_strength,
            **train_kwargs,
        )
        stats = analyze_weight_distribution(net, reg_type)
        results[reg_type] = (net, history, stats)
        print(f"Weight statistics for {reg_type}: {stats}")
    return results


def _save_or_show(save_path, name):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{name}.png"), bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(results: Dict, save_path: str = None):
    reg_types = list(results.keys())
    train_loss_data = {r: results[r][1]["train_loss"] for r in reg_types}
    val_loss_data = {r: results[r][1]["val_loss"] for r in reg_types}
    val_acc_data = {r: results[r][1]["val_acc"] for r in reg_types}
    sparsity_data = [results[r][2]["sparsity_percent"] for r in reg_types]
    l1_norm_data = [results[r][2]["l1_norm"] for r in reg_types]
    l2_norm_data = [results[r][2]["l2_norm"] for r in reg_types]

    plt.figure(figsize=(8, 5))
    for r in reg_types:
        plt.plot(train_loss_data[r], label=f"{r.capitalize()} Train Loss", linestyle="-")
        plt.plot(val_loss_data[r], label=f"{r.capitalize()} Val Loss", linestyle="--")
    plt.title("1. Training and Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    _save_or_show(save_path, "1_loss_curves")

    plt.figure(figsize=(8, 5))
    for r in reg_types:
        plt.plot(val_acc_data[r], label=f"{r.capitalize()}")
    plt.title("2. Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    _save_or_show(save_path, "2_val_accuracy_curves")

    plt.figure(figsize=(12, 8))
    plt.suptitle("3. Weight Distribution Histograms", fontsize=16)
    for i, r in enumerate(reg_types):
        stats = results[r][2]
        plt.subplot(2, 2, i + 1)
        hist_counts, hist_bins = stats["histogram"]
        bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2.0
        plt.bar(
            bin_centers,
            hist_counts,
            width=(hist_bins[1] - hist_bins[0]),
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
        )
        plt.title(f"{r.capitalize()} (Sparsity: {stats['sparsity_percent']:.2%})")
        plt.xlabel("Weight Value")
        plt.ylabel("Count")
        plt.xlim(-0.1, 0.1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_or_show(save_path, "3_weight_histograms")

    plt.figure(figsize=(6, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(reg_types)))
    plt.bar(reg_types, sparsity_data, color=colors)
    plt.title("4. Model Sparsity Comparison (|w| < 0.01)")
    plt.xlabel("Regularization Type")
    plt.ylabel("Sparsity Percentage")
    for i, sp in enumerate(sparsity_data):
        plt.text(i, sp + 0.005, f"{sp:.2%}", ha="center")
    plt.grid(axis="y", alpha=0.3)
    _save_or_show(save_path, "4_sparsity_comparison")

    x = np.arange(len(reg_types))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, l1_norm_data, width, label="L1 Norm", color="c")
    plt.bar(x + width / 2, l2_norm_data, width, label="L2 Norm", color="darkorange")
    plt.title("5. Weight Norm Comparison")
    plt.ylabel("Norm Value")
    plt.xticks(x, reg_types)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path, "5_l1_l2_norm_comparison")

    plt.figure(figsize=(8, 5))
    for r in reg_types:
        plt.plot(results[r][1]["train_acc"], label=f"{r.capitalize()} Train Acc")
    plt.title("6. Training Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    _save_or_show(save_path, "6_train_accuracy_curves")


def create_toy_dataset(N: int = 400, D: int = 50, C: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.randn(N, D)
    y = np.random.randint(0, C, N)
    return X, y


def run_experiment(seed: int = 42) -> Dict:
    np.random.seed(seed)
    X_train, y_train = create_toy_dataset(N=400, D=50, C=3)
    X_val, y_val = create_toy_dataset(N=120, D=50, C=3)
    train_kwargs = {
        "hidden_dim": 64,
        "learning_rate": 5e-3,
        "loss_type": "hybrid",
        "num_epochs": 60,
        "batch_size": 64,
        "alpha": 0.6,
        "beta": 0.4,
        "verbose": False,
    }
    results = compare_regularizations(
        X_train,
        y_train,
        X_val,
        y_val,
        reg_types=["none", "l1", "l2", "elastic"],
        reg_strength=0.01,
        train_kwargs=train_kwargs,
    )
    return results


def main():
    print("=" * 70)
    print("Running Hybrid Loss + Regularization experiment on a toy dataset")
    print("=" * 70)
    results = run_experiment()
    save_dir = "./plots"
    print(f"\nSaving plots to {save_dir} ...")
    plot_training_curves(results, save_path=save_dir)
    print("Plots generated:")
    for fname in sorted(os.listdir(save_dir)):
        print(f" - {os.path.join(save_dir, fname)}")


if __name__ == "__main__":
    main()
