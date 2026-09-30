"""
Solved implementation for:
Laboratory – Neural Models: Learning, Depth, Activations, and Output Layers

Covers:
1. Binary XOR problem specification and linear-vs-nonlinear demonstration.
2. 2-2-1 XOR network using sigmoid/tanh/ReLU hidden activations.
3. BCEWithLogitsLoss, backpropagation, gradient inspection, and predictions.
4. Zero-initialisation symmetry experiment.
5. Activation experiment: sigmoid, tanh, ReLU.
6. Three-class extension using 3 logits + CrossEntropyLoss.
7. Stable-softmax shift diagnostic.

Requirements:
    Python 3
    PyTorch
    NumPy (not strictly required by the implementation, but listed by the lab)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility and dataset
# ---------------------------------------------------------------------------

SEED = 11
LEARNING_RATE = 0.5
STEPS = 5000

torch.manual_seed(SEED)

# XOR:
# (0,0) -> 0
# (0,1) -> 1
# (1,0) -> 1
# (1,1) -> 0
X = torch.tensor(
    [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]]
)

Y_BINARY = torch.tensor(
    [[0.0],
     [1.0],
     [1.0],
     [0.0]]
)

# Three-class extension:
# class 0 -> (0,0)
# class 1 -> (0,1), (1,0)
# class 2 -> (1,1)
Y_3CLASS = torch.tensor([0, 1, 1, 2])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class XORNet(nn.Module):
    """2 inputs -> 2 hidden units -> 1 output logit."""

    def __init__(self, activation):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.activation = activation

    def forward(self, x):
        hidden = self.activation(self.fc1(x))
        return self.fc2(hidden)


class ThreeClassNet(nn.Module):
    """2 inputs -> 2 hidden units -> 3 output logits."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 3)

    def forward(self, x):
        hidden = torch.tanh(self.fc1(x))
        return self.fc2(hidden)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def linear_xor_demo():
    """
    Train a single affine layer + sigmoid on XOR.

    A sigmoid applied to an affine function is still controlled by one
    linear decision boundary, so it cannot represent XOR exactly.
    """
    torch.manual_seed(SEED)

    model = nn.Linear(2, 1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    initial_loss = loss_fn(model(X), Y_BINARY).item()

    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = loss_fn(model(X), Y_BINARY)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
        predictions = (probabilities >= 0.5).float()

    return initial_loss, loss.item(), probabilities.squeeze(1), predictions.squeeze(1)


def train_binary(activation_name, seed=SEED, steps=STEPS, lr=LEARNING_RATE):
    """
    Train the 2-2-1 binary XOR network.

    The first backward pass is inspected so that the experiment exposes
    an actual gradient tensor rather than relying only on the final loss.
    """
    activations = {
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
    }

    if activation_name not in activations:
        raise ValueError("activation_name must be sigmoid, tanh, or relu")

    torch.manual_seed(seed)

    model = XORNet(activations[activation_name])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    initial_loss = loss_fn(model(X), Y_BINARY).item()

    early_gradient = None
    early_gradient_tensor = None

    for step in range(steps):
        optimizer.zero_grad()

        # Forward pass
        logits = model(X)

        # Scalar loss
        loss = loss_fn(logits, Y_BINARY)

        # Reverse-mode automatic differentiation / backpropagation
        loss.backward()

        if step == 0:
            early_gradient = model.fc1.weight.grad.norm().item()
            early_gradient_tensor = model.fc1.weight.grad.detach().clone()

        # Optimiser updates parameters
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.sigmoid(logits).squeeze(1)
        predictions = (probabilities >= 0.5).float()

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": loss.item(),
        "probabilities": probabilities,
        "predictions": predictions,
        "early_gradient_norm": early_gradient,
        "early_gradient_tensor": early_gradient_tensor,
    }


def symmetry_experiment(steps=8, lr=LEARNING_RATE):
    """
    Set all weights/biases to zero and show that the two hidden units
    receive identical gradients and therefore remain symmetric.

    Because the two hidden units start identically and see the same inputs,
    they compute the same values and receive the same updates.
    """
    torch.manual_seed(SEED)

    model = XORNet(nn.Tanh())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    records = []

    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(X), Y_BINARY)
        loss.backward()

        # Difference between the two hidden-unit rows.
        row_difference = (
            model.fc1.weight.detach()[0] - model.fc1.weight.detach()[1]
        ).abs().max().item()

        gradient_row_difference = (
            model.fc1.weight.grad[0] - model.fc1.weight.grad[1]
        ).abs().max().item()

        records.append(
            {
                "step": step,
                "loss": loss.item(),
                "max_weight_row_difference": row_difference,
                "max_gradient_row_difference": gradient_row_difference,
            }
        )

        optimizer.step()

    return model, records


def train_three_class(seed=SEED, steps=STEPS, lr=LEARNING_RATE):
    """
    Three-class extension:
        hidden representation: 2 units
        output: 3 logits
        loss: CrossEntropyLoss
    """
    torch.manual_seed(seed)

    model = ThreeClassNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    initial_loss = loss_fn(model(X), Y_3CLASS).item()

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, Y_3CLASS)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

    return {
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": loss.item(),
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
    }


def print_binary_result(name, result):
    print(f"\n{name.upper()} ACTIVATION")
    print("-" * 55)
    print(f"Initial loss:              {result['initial_loss']:.8f}")
    print(f"Final loss:                {result['final_loss']:.8f}")
    print(f"Early ||grad W1||_2:       {result['early_gradient_norm']:.8f}")
    print("Early W1 gradient tensor:")
    print(result["early_gradient_tensor"])
    print("Final probabilities:")
    for x, p in zip(X.tolist(), result["probabilities"].tolist()):
        print(f"  {x} -> {p:.8f}")
    print("Predicted labels:", result["predictions"].int().tolist())
    print("Expected labels: ", Y_BINARY.int().squeeze(1).tolist())
    print(
        "All 4 correct:            ",
        bool(torch.equal(result["predictions"].int(),
                         Y_BINARY.int().squeeze(1)))
    )


def main():
    print("=" * 72)
    print("NEURAL MODELS LAB — SOLVED PYTORCH IMPLEMENTATION")
    print("=" * 72)

    print("\nTASK 1: XOR / LINEAR BASELINE")
    print("-" * 55)
    print("Input/output mapping:")
    print("  (0,0) -> 0")
    print("  (0,1) -> 1")
    print("  (1,0) -> 1")
    print("  (1,1) -> 0")

    initial, final, probs, preds = linear_xor_demo()
    print(f"\nSingle affine layer + sigmoid:")
    print(f"Initial loss: {initial:.8f}")
    print(f"Final loss:   {final:.8f}")
    print("Probabilities:", [round(v, 6) for v in probs.tolist()])
    print("Predictions:  ", preds.int().tolist())
    print(
        "\nInterpretation: one affine decision boundary cannot separate "
        "the XOR classes exactly."
    )

    print("\nTASK 2–4: BINARY XOR WITH A 2–2–1 NETWORK")
    print("-" * 55)

    # The same seed, optimizer, learning rate, and number of steps are used
    # across the activation comparison. Only the hidden activation changes.
    results = {}
    for activation in ("sigmoid", "tanh", "relu"):
        results[activation] = train_binary(activation)
        print_binary_result(activation, results[activation])

    print("\nCOMPACT ACTIVATION RESULT TABLE")
    print("-" * 72)
    print(f"{'Activation':<12} {'Final loss':>14} {'4/4 correct':>14} {'Early grad norm':>20}")
    for activation, result in results.items():
        correct = bool(
            torch.equal(
                result["predictions"].int(),
                Y_BINARY.int().squeeze(1)
            )
        )
        print(
            f"{activation:<12} "
            f"{result['final_loss']:>14.8f} "
            f"{str(correct):>14} "
            f"{result['early_gradient_norm']:>20.8f}"
        )

    print("\nTASK 4C: ZERO-INITIALISATION SYMMETRY")
    print("-" * 55)
    _, symmetry_records = symmetry_experiment()

    print(
        "The two hidden-unit rows start identical. The table shows that "
        "their maximum row difference stays zero (up to floating point precision)."
    )
    print(
        f"{'Step':<8} {'Loss':>14} {'Weight row diff':>20} "
        f"{'Gradient row diff':>22}"
    )
    for record in symmetry_records:
        print(
            f"{record['step']:<8} "
            f"{record['loss']:>14.8f} "
            f"{record['max_weight_row_difference']:>20.8e} "
            f"{record['max_gradient_row_difference']:>22.8e}"
        )

    print("\nTASK 5: THREE-CLASS EXTENSION")
    print("-" * 55)
    three = train_three_class()

    print("Class mapping:")
    print("  class 0 -> both sensors inactive: (0,0)")
    print("  class 1 -> sensors disagree:       (0,1), (1,0)")
    print("  class 2 -> both sensors active:    (1,1)")

    print(f"\nFinal CrossEntropyLoss: {three['final_loss']:.8f}")
    print("Output-layer weight shape:", tuple(three["model"].fc2.weight.shape))
    print("Number of logits/example: 3")
    print("\nPredicted class probabilities:")
    for x, p, pred in zip(
        X.tolist(),
        three["probabilities"].tolist(),
        three["predictions"].tolist()
    ):
        print(f"  {x} -> {p} -> class {pred}")

    example_index = 1
    probability_sum = three["probabilities"][example_index].sum().item()
    print(
        f"\nSoftmax check for example {X[example_index].tolist()}: "
        f"sum = {probability_sum:.10f}"
    )

    # Optional numerical-stability diagnostic required by the lab.
    test_logits = torch.tensor([[1.0, 2.0, 3.0]])
    shifted_logits = test_logits + 100.0
    p1 = torch.softmax(test_logits, dim=1)
    p2 = torch.softmax(shifted_logits, dim=1)

    print("\nSoftmax shift diagnostic:")
    print("softmax([1,2,3])       =", p1.squeeze(0).tolist())
    print("softmax([101,102,103]) =", p2.squeeze(0).tolist())
    print("Maximum absolute difference:",
          (p1 - p2).abs().max().item())

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
