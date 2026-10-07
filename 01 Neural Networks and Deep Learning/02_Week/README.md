
# 🐈 Week 2: Neural Networks Basics & Vectorized Logistic Regression

Welcome to Week 2 of **Neural Networks and Deep Learning** (Course 1 of the **DeepLearning.AI Deep Learning Specialization**)! This module establishes the mathematical, vectorization, and computational foundations of deep learning: formulating binary classification through a neural network mindset, computing derivatives via forward/backward computation graphs, executing vectorized operations across $m$ training examples without explicit Python loops, and leveraging NumPy broadcasting mechanics.

---

## 📝 Core Technical Objectives
* **Logistic Regression as a Single-Neuron Network:** Formulating prediction $\hat{y} = a = \sigma(\mathbf{w}^T \mathbf{x} + b)$ using the Sigmoid activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ and binary cross-entropy loss $\mathcal{L}(a, y) = -\left[y \log(a) + (1-y) \log(1-a)\right]$.
* **Computation Graph & Backpropagation:** Tracking forward-pass operations and executing backward-pass chain rule calculus to compute parameter derivatives $\frac{\partial \mathcal{L}}{\partial z} = dz = a - y$, $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = d\mathbf{w} = \mathbf{x} dz$, and $\frac{\partial \mathcal{L}}{\partial b} = db = dz$.
* **Vectorized Training Across $m$ Examples:** Replacing explicit `for` loops with matrix calculations: linear activations $\mathbf{Z} = \mathbf{w}^T \mathbf{X} + b \in \mathbb{R}^{1 \times m}$, predicted output vector $\mathbf{A} = \sigma(\mathbf{Z})$, and averaged gradients $d\mathbf{w} = \frac{1}{m} \mathbf{X} d\mathbf{Z}^T$, $db = \frac{1}{m} \sum d\mathbf{Z}$.
* **NumPy Vector Mechanics & Broadcasting:** Utilizing NumPy matrix operations, avoiding rank-1 arrays `(n,)` in favor of explicit column/row vectors `(n, 1)`, and leveraging element-wise broadcasting across mismatched tensor dimensions.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, interactive Jupyter notebooks, and practice assignments are mapped directly to their targeted analytical focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Python Basics with NumPy](./Python_Basics_with_Numpy.ipynb)** | Practice programming lab covering vectorized sigmoid functions, matrix normalization, softmax activation implementations, and L1/L2 loss calculations. |
| **[Logistic Regression Assignment](./Logistic_Regression_with_a_Neural_Network_mindset.ipynb)** | Graded programming project building a complete cat-vs-non-cat binary image classifier from scratch using pure NumPy matrix forward/backward passes. |
| **[Pieter Abbeel Interview](./Pieter%20Abbeel%20Interview)** | Strategic discussion on robotics, deep reinforcement learning, apprenticeship learning, and early research developments in AI. |

---

## 💡 Visual Pipeline Reference

The forward propagation, backward computation graph, and vectorized parameter update lifecycle implemented across this module:

* **Matrix Data Ingestion** ➔ Unroll input images $\mathbf{X} \in \mathbb{R}^{(n_x \times m)}$ and target labels $\mathbf{Y} \in \mathbb{R}^{(1 \times m)}$.
* **Vectorized Forward Pass** ➔ Compute linear combination $\mathbf{Z} = \mathbf{w}^T \mathbf{X} + b$ ➔ Apply non-linear transformation $\mathbf{A} = \sigma(\mathbf{Z})$ ➔ Calculate overall cost $J(\mathbf{w}, b) = -\frac{1}{m} \sum \left[ \mathbf{Y} \log(\mathbf{A}) + (1 - \mathbf{Y}) \log(1 - \mathbf{A}) \right]$.
* **Vectorized Backpropagation** ➔ Calculate activation errors $d\mathbf{Z} = \mathbf{A} - \mathbf{Y}$ ➔ Compute weight gradients $d\mathbf{w} = \frac{1}{m} \mathbf{X} d\mathbf{Z}^T$ and bias gradient $db = \frac{1}{m} \text{np.sum}(d\mathbf{Z})$.
* **Gradient Descent Update Loop** ➔ Update parameters simultaneously: $\mathbf{w} \leftarrow \mathbf{w} - \alpha \, d\mathbf{w}$ and $b \leftarrow b - \alpha \, db$.

---

## 🎯 Technical Skills Architecture

### 📊 Deep Learning Mathematical Foundations
* **Derivation of Binary Cross-Entropy:** Proving why $\frac{\partial \mathcal{L}}{\partial z} = a - y$ simplifies cleanly through derivative chain rule components $\frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z}$.
* **Vector Algebra & Memory Layout:** Structuring feature matrices where columns represent individual training samples to enable efficient parallel linear algebra computations.
* **Broadcasting Execution Rules:** Understanding how NumPy automatically expands dimension size $1$ arrays to match target tensor shapes during arithmetic operations.

### 🤖 Applied Engineering Strategy
* **High-Performance Vectorization:** Eliminating nested loops in Python to achieve $100\times+$ computational speedups using SIMD instruction sets via NumPy underlying C implementations.
* **Explicit Tensor Shaping:** Utilizing `assert(w.shape == (n_x, 1))` and explicit `reshape()` commands to eliminate subtle matrix rank-1 array bugs.
* **Image Preprocessing Pipelines:** Flattening $(H, W, C)$ image arrays into $1\text{D}$ vectors and scaling pixel values by dividing by $255$.

---

## 🛠️ Production Tech Stack & Ecosystem

| Numerical Vectorization | Image Preprocessing | Development Environment |
| :---: | :---: | :---: |
| ![NumPy](https://img.shields.io/badge/NumPy-Broadcasting_Vectors-013243?style=flat&logo=numpy&logoColor=white) | ![PIL / SciPy](https://img.shields.io/badge/PIL%2FSciPy-Image_Flattening-3776AB?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |

