
# 🌸 Week 3: Shallow Neural Networks & Planar Data Classification

Welcome to Week 3 of **Neural Networks and Deep Learning** (Course 1 of the **DeepLearning.AI Deep Learning Specialization**)! This module transitions from single-neuron models to fully realized 2-layer shallow neural networks with hidden layers: defining non-linear activation functions ($\text{tanh}$, $\text{ReLU}$, Sigmoid), understanding the necessity of non-linearity, performing full forward and backward propagation matrix calculus, breaking symmetry via Gaussian random initialization, and evaluating non-linear classification boundaries on non-linearly separable planar datasets.

---

## 📝 Core Technical Objectives
* **Neural Network Layer Representations:** Structuring 2-layer architectures with input vector $\mathbf{x} = \mathbf{a}^{[0]} \in \mathbb{R}^{n_x}$, hidden activation layer $\mathbf{a}^{[1]} \in \mathbb{R}^{n_h}$, and output activation $\mathbf{a}^{[2]} = \hat{y} \in \mathbb{R}^{n_y}$.
* **Non-Linear Activation Dynamics:** Evaluating non-linear transformations ($\text{tanh}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, $\text{ReLU}(z) = \max(0, z)$, Leaky $\text{ReLU}$, and Sigmoid) and proving why linear activations cause multi-layer networks to collapse into simple linear regression equivalents regardless of depth.
* **Vectorized Forward & Backpropagation:** Computing explicit layer-by-layer linear updates $\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]}$ and activation updates $\mathbf{A}^{[1]} = g^{[1]}(\mathbf{Z}^{[1]})$, followed by exact backpropagation derivative calculations using element-wise Hadamard products: $d\mathbf{Z}^{[1]} = \left(\mathbf{W}^{[2]T} d\mathbf{Z}^{[2]}\right) * g^{[1]'}\left(\mathbf{Z}^{[1]}\right)$.
* **Symmetry Breaking via Random Initialization:** Demonstrating why initializing weight matrices to zeros causes all hidden units to compute identical features ($\mathbf{a}_1^{[1]} = \mathbf{a}_2^{[1]}$) and receive identical gradient updates, requiring small non-zero random initialization ($\mathbf{W} \sim \mathcal{N}(0, 1) \times 0.01$).

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture slides, Jupyter notebooks, and assignments are mapped directly to their targeted analytical focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Week 3 Lecture Slides](./C1_W3.pdf)** | Formal visual and mathematical reference covering layer index notation, vectorized array dimensions, and derivation of gradient update equations. |
| **[Planar Data Classification Assignment](./Planar_data_classification_with_one_hidden_layer.ipynb)** | Graded programming project building a 2-layer neural network from scratch in pure NumPy to classify complex red/blue flower petal distributions where logistic regression fails. |
| **[Ian Goodfellow Interview](./Ian%20Goodfellow%20Interview)** | Strategic discussion on Generative Adversarial Networks (GANs), adversarial machine learning robustness, and early breakthroughs in deep learning. |

---

## 💡 Visual Pipeline Reference

The shallow neural network vector dimensions, forward activation pass, and backward gradient flow implemented across this module:

* **Matrix Initialization & Shape Safety**:
  * $\mathbf{W}^{[1]} \in \mathbb{R}^{(n_h \times n_x)}$, $\mathbf{b}^{[1]} \in \mathbb{R}^{(n_h \times 1)}$ initialized via `np.random.randn(n_h, n_x) * 0.01` and `np.zeros((n_h, 1))`.
  * $\mathbf{W}^{[2]} \in \mathbb{R}^{(n_y \times n_h)}$, $\mathbf{b}^{[2]} \in \mathbb{R}^{(n_y \times 1)}$ initialized similarly.
* **Vectorized Forward Propagation Across $m$ Samples**:
  $$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]} \quad \longrightarrow \quad \mathbf{A}^{[1]} = \text{tanh}\left(\mathbf{Z}^{[1]}\right)$$
  $$\mathbf{Z}^{[2]} = \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]} \quad \longrightarrow \quad \mathbf{A}^{[2]} = \sigma\left(\mathbf{Z}^{[2]}\right)$$
* **Vectorized Backpropagation Loop**:
  $$d\mathbf{Z}^{[2]} = \mathbf{A}^{[2]} - \mathbf{Y}$$
  $$d\mathbf{W}^{[2]} = \frac{1}{m} d\mathbf{Z}^{[2]} \mathbf{A}^{[1]T}, \quad db^{[2]} = \frac{1}{m} \text{np.sum}\left(d\mathbf{Z}^{[2]}, \text{axis}=1, \text{keepdims}=\text{True}\right)$$
  $$d\mathbf{Z}^{[1]} = \left(\mathbf{W}^{[2]T} d\mathbf{Z}^{[2]}\right) * \left(1 - \left(\mathbf{A}^{[1]}\right)^2\right)$$
  $$d\mathbf{W}^{[1]} = \frac{1}{m} d\mathbf{Z}^{[1]} \mathbf{X}^T, \quad db^{[1]} = \frac{1}{m} \text{np.sum}\left(d\mathbf{Z}^{[1]}, \text{axis}=1, \text{keepdims}=\text{True}\right)$$

---

## 🎯 Technical Skills Architecture

### 📊 Deep Learning Mathematical Foundations
* **Activation Calculus:** Deriving explicit derivatives for activations: $g'(\mathbf{z}) = 1 - \tanh^2(\mathbf{z})$ for $\text{tanh}$ and $g'(\mathbf{z}) = \sigma(\mathbf{z})(1 - \sigma(\mathbf{z}))$ for Sigmoid.
* **Non-Linear Decision Boundaries:** Constructing multi-dimensional decision hyperplanes capable of wrapping non-linearly separable cluster topographies.
* **Cross-Entropy Cost Evaluation:** Computing numerical loss stability over $m$ samples: $J = -\frac{1}{m} \sum \left[ \mathbf{Y}\log(\mathbf{A}^{[2]}) + (1-\mathbf{Y})\log(1-\mathbf{A}^{[2]}) \right]$.

### 🤖 Applied Engineering Strategy
* **Hidden Unit Capacity Tuning:** Experimenting with hidden unit size $n_h \in \{1, 2, 3, 4, 5, 20, 50\}$ to observe the transition from underfitting to precise fitting and potential overfitting.
* **Hadamard vs. Matrix Products:** Managing matrix multiplication (`np.dot`) versus element-wise operations (`*`) during gradient chain updates.
* **Strict Shape Consistency:** Leveraging `keepdims=True` during reduction sums to maintain rank-2 column vectors `(n, 1)` and prevent silent broadcasting bugs.

---

## 🛠️ Production Tech Stack & Ecosystem

| Vector Linear Algebra | Data Visualization | Development Environment |
| :---: | :---: | :---: |
| ![NumPy](https://img.shields.io/badge/NumPy-Matrix_Calculus-013243?style=flat&logo=numpy&logoColor=white) | ![Matplotlib](https://img.shields.io/badge/Matplotlib-Planar_Boundaries-11557c?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |

