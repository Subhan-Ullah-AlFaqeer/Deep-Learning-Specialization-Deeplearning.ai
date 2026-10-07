
# 🐾 Week 4: Deep Neural Networks & Image Classification

Welcome to Week 4 of **Neural Networks and Deep Learning** (Course 1 of the **DeepLearning.AI Deep Learning Specialization**)! This final module transitions from shallow models to multi-layer deep $L$-layer neural networks: formalizing general forward and backward propagation building blocks, passing linear and activation caches across explicit computational blocks, performing rigorous matrix dimension verification, exploring hierarchical deep representations, and tuning primary model parameters versus hyperparameters on a binary cat image classifier.

---

## 📝 Core Technical Objectives
* **General Deep $L$-Layer Architecture:** Generalizing neural network representations across arbitrary depths $l \in \{1, 2, \dots, L\}$, mapping inputs $\mathbf{a}^{[0]} = \mathbf{X}$ to output predictions $\mathbf{a}^{[L]} = \hat{\mathbf{Y}}$.
* **Forward and Backward Cache Propagation:** Storing linear tuples $(\mathbf{A}^{[l-1]}, \mathbf{W}^{[l]}, \mathbf{b}^{[l]})$ in linear caches and linear/activation values $(\mathbf{Z}^{[l]}, \text{cache--linear})$ in activation caches during forward pass to feed exact variables into backpropagation derivative formulas.
* **Matrix Dimension Verification Protocols:** Formulating strict dimensional invariant rules: $\mathbf{W}^{[l]} \in \mathbb{R}^{(n_l \times n_{l-1})}$, $\mathbf{b}^{[l]} \in \mathbb{R}^{(n_l \times 1)}$, $\mathbf{Z}^{[l]}, \mathbf{A}^{[l]} \in \mathbb{R}^{(n_l \times m)}$, and gradient tensors $d\mathbf{W}^{[l]}, d\mathbf{b}^{[l]}, d\mathbf{Z}^{[l]}$ matching their parameter counterparts exactly.
* **Hierarchical Deep Representation Intuition:** Understanding how early layers compute low-level primitive features (edges, orientations), mid-layers compose structural parts (eyes, noses, contours), and deeper layers form complex object concepts (faces, full bodies).
* **Parameters vs. Hyperparameters:** Distinguishing explicitly learned weights $(\mathbf{W}, \mathbf{b})$ from governing hyperparameters: learning rate $\alpha$, epoch iterations, depth $L$, layer hidden units $n_l$, and activation function choices ($\text{ReLU}$ vs. Sigmoid).

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, reading notes, and interactive Jupyter assignments are mapped directly to their targeted analytical focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Building your Deep Neural Network: Step by Step](./001%20Assingment/Building_your_Deep_Neural_Network_Step_by_Step.ipynb)** | Graded modular programming project building flexible linear/activation forward functions (`linear_forward`, `linear_activation_forward`, `L_model_forward`) and backprop modules (`linear_backward`, `linear_activation_backward`, `L_model_backward`) in pure NumPy. |
| **[Deep Neural Network - Application](./002%20Assingment/Deep%20Neural%20Network%20-%20Application.ipynb)** | Applied deep learning programming assignment assembling a 2-layer NN and an $L$-layer deep neural network to classify cat images ($64 \times 64 \times 3$), demonstrating accuracy gains with increased network depth. |

---

## 💡 Visual Pipeline Reference

The general $L$-layer forward computation, cached data flow, backpropagation loop, and parameter updates implemented in this module:

* **Forward Propagation Loop ($l = 1 \dots L$)**:
  $$\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$$
  $$\mathbf{A}^{[l]} = g^{[l]}\left(\mathbf{Z}^{[l]}\right) \quad \text{where } g^{[l]} = \text{ReLU for } l < L, \quad g^{[L]} = \text{Sigmoid}$$
  $$\text{Cache }^{[l]} \leftarrow \left(\left(\mathbf{A}^{[l-1]}, \mathbf{W}^{[l]}, \mathbf{b}^{[l]}\right), \mathbf{Z}^{[l]}\right)$$
* **Cost Evaluation**:
  $$J(\mathbf{W}, \mathbf{b}) = -\frac{1}{m} \sum \left[ \mathbf{Y}\log\left(\mathbf{A}^{[L]}\right) + (1-\mathbf{Y})\log\left(1-\mathbf{A}^{[L]}\right) \right]$$
* **Backward Propagation Loop ($l = L \dots 1$)**:
  $$d\mathbf{Z}^{[l]} = d\mathbf{A}^{[l]} * g^{[l]'}\left(\mathbf{Z}^{[l]}\right)$$
  $$d\mathbf{W}^{[l]} = \frac{1}{m} d\mathbf{Z}^{[l]} \mathbf{A}^{[l-1]T}, \quad d\mathbf{b}^{[l]} = \frac{1}{m} \text{np.sum}\left(d\mathbf{Z}^{[l]}, \text{axis}=1, \text{keepdims}=\text{True}\right)$$
  $$d\mathbf{A}^{[l-1]} = \mathbf{W}^{[l]T} d\mathbf{Z}^{[l]}$$
* **Gradient Descent Parameter Update**:
  $$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \alpha \, d\mathbf{W}^{[l]}, \quad \mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \alpha \, d\mathbf{b}^{[l]}$$

---

## 🎯 Technical Skills Architecture

### 📊 Deep Learning Modular Engineering
* **Cached Computational Graphs:** Encapsulating linear and activation parameters into clean dictionary tuples during execution passes to keep backpropagation functions purely functional and scalable.
* **Vector Dimension Invariants:** Conducting structural debugging checks using tensor shapes to guarantee zero implicit broadcasting during complex multi-layer backpropagation passes.
* **Initialization Scaling:** Applying He initialization scaling factor $\sqrt{\frac{2}{n_{l-1}}}$ for $\text{ReLU}$ networks to keep parameter variance stable across deep layers.

### 🤖 Applied Computer Vision Engineering
* **Multi-Layer Perception on Raw Pixels:** Flattening color images of shape $(64, 64, 3)$ into feature vectors of length $n_x = 12288$ and scaling values to $[0, 1]$ range.
* **Depth Generalization Benchmarking:** Comparing empirical performance between a 2-layer baseline network ($72\%$ test accuracy) and a deeper 4-layer model ($80\%$ test accuracy) on non-linear image classification tasks.
* **Hyperparameter Tuning Strategies:** Managing iterative experimental workflows for learning rates $\alpha \in \{0.075, 0.009, 0.001\}$ and epoch thresholds to avoid vanishing/exploding activations.

---

## 🛠️ Production Tech Stack & Ecosystem

| Vector Computing & Linear Algebra | Computer Vision & Data Handling | Development Environment |
| :---: | :---: | :---: |
| ![NumPy](https://img.shields.io/badge/NumPy-Modular_NN_Engine-013243?style=flat&logo=numpy&logoColor=white) | ![HDF5 / PIL](https://img.shields.io/badge/H5py%2FPIL-Dataset_Loading-3776AB?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |

