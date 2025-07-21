# Neural Network from Scratch

A comprehensive implementation of a 2-layer Neural Network for MNIST digit classification built entirely from scratch using NumPy, along with an equivalent PyTorch implementation for comparison.

## 🎯 Project Overview

This project demonstrates the fundamental concepts of deep learning by implementing a neural network without relying on high-level frameworks like TensorFlow or PyTorch. The from-scratch implementation provides deep insights into the mathematical foundations and computational mechanics that power modern deep learning frameworks.

## 🚀 Why Build from Scratch?

While powerful frameworks like TensorFlow and PyTorch exist, building neural networks from scratch offers several educational advantages:

- **Mathematical Understanding**: Gain deep insights into backpropagation, gradient descent, and optimization algorithms
- **Implementation Clarity**: Understand exactly what happens behind the scenes in high-level frameworks
- **Debugging Skills**: Learn to identify and fix issues at the algorithmic level
- **Foundation Building**: Establish a solid base for understanding advanced architectures
- **Performance Insights**: Appreciate the optimizations that frameworks provide

## 🏗️ Architecture

### Network Structure
```
Input Layer (784 neurons) → Hidden Layer (128 neurons) → Output Layer (10 neurons)
        ↓                           ↓                         ↓
   28×28 pixels              ReLU Activation           Softmax Activation
```

### Mathematical Formulation

#### Forward Propagation
1. **Hidden Layer:**
   ```
   Z₁ = W₁ * X + b₁
   A₁ = ReLU(Z₁) = max(0, Z₁)
   ```

2. **Output Layer:**
   ```
   Z₂ = W₂ * A₁ + b₂
   A₂ = Softmax(Z₂) = exp(Z₂ᵢ - max(Z₂)) / Σⱼ exp(Z₂ⱼ - max(Z₂))
   ```

#### Loss Function
**Cross-Entropy Loss:**
```
Loss = -1/m * Σᵢ Σⱼ yᵢⱼ * log(ŷᵢⱼ + ε)
```
where:
- m = number of samples
- yᵢⱼ = true label (one-hot encoded)
- ŷᵢⱼ = predicted probability
- ε = small constant (1e-8) to prevent log(0)

#### Backward Propagation
1. **Output Layer Gradients:**
   ```
   dZ₂ = A₂ - Y
   dW₂ = 1/m * dZ₂ * A₁ᵀ
   db₂ = 1/m * Σ(dZ₂)
   ```

2. **Hidden Layer Gradients:**
   ```
   dA₁ = W₂ᵀ * dZ₂
   dZ₁ = dA₁ * ReLU'(Z₁)
   dW₁ = 1/m * dZ₁ · Xᵀ
   db₁ = 1/m * Σ(dZ₁)
   ```

#### Parameter Updates
**Gradient Descent:**
```
W₁ := W₁ - α * dW₁
b₁ := b₁ - α * db₁
W₂ := W₂ - α * dW₂
b₂ := b₂ - α * db₂
```
where α is the learning rate.

#### Weight Initialization
**He Initialization (for ReLU):**
```
W ~ N(0, √(2/n_in))
```
where n_in is the number of input neurons to the layer.

## 📊 Implementation Details

### Data Preprocessing
- **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
- **One-Hot Encoding**: Labels converted to 10-dimensional vectors
- **Train-Test Split**: 80% training, 20% testing

### Hyperparameters
- **Learning Rate**: 0.001
- **Epochs**: 30
- **Hidden Layer Size**: 128 neurons
- **Batch Processing**: Full dataset (can be modified for mini-batch)

## 🔧 Usage

### Running the From-Scratch Implementation
```bash
# Open the Jupyter notebook
jupyter notebook neural_network_scratch.ipynb

# Or run directly if converted to Python script
python neural_network_scratch.py
```



## 🔍 Key Differences: From-Scratch vs PyTorch

| Aspect | From-Scratch (NumPy) | PyTorch |
|--------|----------------------|---------|
| **Weight Initialization** | Manual He initialization | `torch.nn.init.kaiming_normal_()` |
| **Forward Pass** | Manual matrix operations | `nn.Linear()` + `F.relu()` |
| **Loss Calculation** | Manual cross-entropy | `nn.CrossEntropyLoss()` |
| **Backward Pass** | Manual gradient calculation | `loss.backward()` |
| **Parameter Updates** | Manual gradient descent | `optimizer.step()` |
| **Data Loading** | Manual preprocessing | `DataLoader` with transforms |
| **Code Length** | Lengthy | Shorter |
| **Learning Value** | High (understand internals) | Medium (framework usage) |

## 📈 Results

Both implementations achieve similar performance on MNIST:
- **Accuracy**: ~95-98% on test set
- **Training Time**: From-scratch is slower but educational
- **Memory Usage**: PyTorch is more optimized

## 🎓 Learning Outcomes

After completing this project, you will understand:

1. **Mathematical Foundations**: The math behind neural networks
2. **Implementation Details**: How frameworks work internally
3. **Optimization Techniques**: Why certain choices are made
4. **Debugging Skills**: How to identify and fix algorithmic issues
5. **Framework Appreciation**: Why PyTorch/TensorFlow are powerful


## 📝 License

This project is open source and available under the [MIT License](LICENSE).

