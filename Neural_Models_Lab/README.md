# Solved Neural Models Lab

This folder contains a complete PyTorch implementation for the laboratory:

**“Laboratory – Neural Models: Learning, Depth, Activations, and Output Layers”**

The implementation follows the exercise specification: XOR is used to study nonlinear representation learning, a 2–2–1 network is trained with a binary output, gradients and losses are inspected, zero initialisation is used for the symmetry experiment, sigmoid/tanh/ReLU are compared, and the task is extended to three classes with three logits and cross-entropy.

## Files

- `solved_neural_models_lab.py` — complete executable solution.
- `README.md` — explanation, design choices, and how to run it.

## Requirements

- Python 3
- PyTorch
- NumPy

CPU is sufficient; no GPU is required by the lab. fileciteturn0file0L47-L54

Install PyTorch if necessary:

```bash
pip install torch numpy
```

## Run

From the folder containing the Python file:

```bash
python solved_neural_models_lab.py
```

The script prints all requested experimental evidence directly to the terminal.

---

# What the code does

## 1. XOR problem

The four training examples are:

| x1 | x2 | Binary target |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

This is the sensor-disagreement rule specified in the lab. fileciteturn0file0L55-L64

The script first trains a **single affine layer + sigmoid** as a linear baseline. It demonstrates that this model cannot represent XOR exactly because an affine transformation provides only one linear decision boundary.

The lab specifically asks students to predict what happens with a single affine transformation followed by a sigmoid. fileciteturn0file0L69-L80

## 2. Binary neural network

The main model is:

```text
2 inputs → 2 hidden units → 1 output logit
```

The hidden activation is changed between:

- Sigmoid
- Tanh
- ReLU

The output uses a **logit** rather than explicitly applying sigmoid inside the model, followed by:

```text
BCEWithLogitsLoss
```

This is the numerically stable PyTorch implementation of the sigmoid + binary-cross-entropy pairing specified by the lab. The lab itself recommends logits with `BCEWithLogitsLoss`. fileciteturn0file0L88-L94 fileciteturn0file0L106-L115

## 3. Training

The script uses:

- random parameter initialisation;
- a fixed seed for reproducibility;
- full-batch training;
- SGD;
- learning rate = `0.5`;
- `5000` training steps.

The lab allows engineering settings such as the learning rate, number of steps, seed, and optimiser to be adjusted when necessary, while keeping the scientific task unchanged. fileciteturn0file0L137-L142

## 4. Backpropagation / gradient check

During the first training step, the script records:

```python
model.fc1.weight.grad
```

and its Euclidean norm.

Conceptually, this tensor represents:

```text
∂L / ∂W(1)
```

That is, how the loss changes with respect to each first-layer weight.

The code explicitly separates the four important stages:

```text
forward pass
      ↓
loss calculation
      ↓
loss.backward()
      ↓
optimizer.step()
```

The lab asks for this connection between the PyTorch backward pass and the chain-rule/backpropagation calculation. fileciteturn0file0L143-L148

## 5. Activation experiment

The same dataset, architecture, seed, optimiser, learning rate and number of steps are used while changing only the hidden activation.

The output table contains:

```text
Hidden activation
Final loss
Whether all 4 examples are correct
Early ||∇W(1)L||₂
```

This matches the requested result table in the lab. fileciteturn0file0L154-L164

### Important interpretation

The results from four XOR points should **not** be used to claim that one activation is universally better. They are observations about this particular experiment.

The lab explicitly asks for observed differences rather than a universal “best activation” claim. fileciteturn0file0L161-L168

## 6. Zero-initialisation symmetry experiment

A second experiment sets all weights and biases to zero before training.

The script records the difference between:

```text
hidden-unit row 1
hidden-unit row 2
```

and also the difference between their gradients.

The expected result is that the two hidden units remain identical. They start identically, compute identical values, and receive identical gradients, so gradient descent gives them identical updates.

This directly implements the symmetry experiment specified on page 4 of the lab. fileciteturn0file0L149-L153

## 7. Three-class extension

The binary target is converted into:

```text
Class 0: (0,0)  both sensors inactive
Class 1: (0,1) or (1,0)  sensors disagree
Class 2: (1,1)  both sensors active
```

The architecture becomes:

```text
2 inputs → 2 hidden units → 3 logits
```

The loss becomes:

```python
nn.CrossEntropyLoss()
```

The script then applies softmax to the final logits for human-readable class probabilities.

The lab requires exactly this change to the output/loss portion. fileciteturn0file0L169-L190

## 8. Softmax checks

The script verifies that the three probabilities for one example sum to approximately 1.

It also performs the optional numerical-stability experiment:

```text
softmax([1, 2, 3])
softmax([101, 102, 103])
```

The two probability vectors should be effectively identical because adding the same constant to every logit does not change softmax probabilities.

This follows the optional diagnostic described in the lab. fileciteturn0file0L183-L189

---

# Expected conceptual answers

## Why is a nonlinear hidden layer needed for XOR?

A stack of affine layers without nonlinear hidden activations collapses to a single affine transformation. Therefore, simply adding more affine layers does not create the nonlinear representation required by XOR.

The nonlinear hidden layer allows the network to transform the input into a representation in which the final decision can separate the two classes. fileciteturn0file0L34-L38

## Why sigmoid + BCE for the binary output?

The target represents one yes/no decision. A sigmoid converts a logit into a value between 0 and 1, which can be interpreted as a binary probability. Binary cross-entropy is therefore the corresponding loss.

In the implementation, `BCEWithLogitsLoss` combines these operations in a numerically stable form. fileciteturn0file0L30-L33

## What does a useful learning signal look like?

A nonzero gradient by itself is not enough. The stronger evidence is:

1. loss decreases substantially;
2. the final predictions correctly classify all four XOR examples;
3. the gradients are nonzero during learning;
4. the behaviour is reproducible under the fixed seed.

These checks follow the laboratory's requirement to inspect losses, predictions and gradients rather than trusting one final number. fileciteturn0file0L135-L148

## Why does zero/identical initialisation cause symmetry?

If both hidden units start with identical parameters, they receive the same input and therefore compute the same output. Because their contributions to the loss are identical, backpropagation gives them identical gradients. They consequently remain identical after every update.

The network therefore fails to develop two distinct hidden features.

## What is the three-class output gradient?

For softmax probabilities `p` and one-hot target `y`, the gradient of cross-entropy with respect to the logits is:

```text
p - y
```
