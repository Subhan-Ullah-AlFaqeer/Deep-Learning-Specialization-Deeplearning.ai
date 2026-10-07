
# 🧱 Week 1: Foundations of Convolutional Neural Networks

Welcome to Week 1 of **Convolutional Neural Networks** (Course 4 of the **DeepLearning.AI Deep Learning Specialization**)! This module establishes the core mathematical and architectural foundation of 2D Computer Vision. You will move from explicit zero-padded array operations in NumPy to high-level graph execution in TensorFlow 2.x—implementing 2D convolutions, max/average pooling forward passes from scratch, parameter sharing mechanisms, and building multi-class vision architectures using both TF Keras Sequential and Functional APIs.

---

## 📝 Core Technical Objectives
- **2D Spatial Convolution & Tensor Transformations:** Transforming input feature tensors $A^{[l-1]} \in \mathbb{R}^{n_H^{[l-1]} \times n_W^{[l-1]} \times n_C^{[l-1]}}$ into output activations $A^{[l]} \in \mathbb{R}^{n_H^{[l]} \times n_W^{[l]} \times n_C^{[l]}}$ using $n_C^{[l]}$ learned 3D filters $K \in \mathbb{R}^{f \times f \times n_C^{[l-1]}}$, zero-padding $p$, and stride $s$:

  $$n_H^{[l]} = \left\lfloor \frac{n_H^{[l-1]} + 2p - f}{s} \right\rfloor + 1, \quad n_W^{[l]} = \left\lfloor \frac{n_W^{[l-1]} + 2p - f}{s} \right\rfloor + 1$$

- **Explicit NumPy Forward Implementations:** Constructing step-by-step vector convolution loops (`conv_single_step`) and spatial window sliding passes over zero-padded feature volumes (`conv_forward`), as well as Max Pooling and Average Pooling downsampling layers (`pool_forward`).

- **Convolutional Advantages & Parameter Efficiency:** Leveraging **Parameter Sharing** (a feature detector useful in one region is useful elsewhere) and **Sparsity of Connections** (each output activation depends only on a local receptive field), drastically reducing parameter counts compared to fully connected layers.

- **TensorFlow 2.x Keras API Architectural Paradigms:** Constructing vision networks using both Keras paradigms:
  - **Sequential API:** Linear layer stacking for baseline feature extractors (e.g., Happy House Mood Classifier).
  - **Functional API:** Non-linear DAG network structures supporting multiple inputs, intermediate feature extraction, and residual connections (e.g., SIGNS Digit Recognition).

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, vector NumPy assignments, and Keras vision models are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C4_W1.pdf)** | Formal visual and mathematical reference covering 2D edge detection filters, padding strategies ("Valid" vs. "Same"), strided convolutions, receptive fields, and Yann LeCun interview insights. |
| **[Convolutional Model Step-by-Step](./01_Week/01%20Assingment/Convolution_model_Step_by_Step_v1.ipynb)** | Ground-up implementation of zero-padding (`zero_pad`), single-step convolution slicing, multi-channel 2D conv forward passes, and pooling forward passes in pure NumPy. |
| **[Convolution Model Application](./01_Week/02%20Assingment/Convolution_model_Application.ipynb)** | End-to-end TensorFlow 2.x Keras model construction: building a Happy House mood classifier via the Sequential API and a 6-class Sign Language Digit ConvNet using the Functional API (`Conv2D` $\to$ `BatchNormalization` $\to$ `ReLU` $\to$ `MaxPool2D` $\to$ `Flatten` $\to$ `Dense`). |
| **[Yann LeCun Interview](./Yann_LeCun_Interview.mp4)** | In-depth dialogue with Turing Award laureate Yann LeCun on the history of LeNet, energy-based models, self-supervised learning, and the future of computer vision architectures. |

---

## 💡 Visual Pipeline Reference

The forward propagation computational pass across 2D Convolution and Pooling layers:

- **Single-Step 3D Filter Cross-Correlation**:

  $$Z_k^{(i)}(h, w) = \sum_{v=1}^{f} \sum_{u=1}^{f} \sum_{c=1}^{n_C^{[l-1]}} \left( A_{\text{slice}}^{(i)}(u, v, c) \cdot K_k(u, v, c) \right) + b_k$$

- **Functional Keras Feature Extraction Pipeline (SIGNS Dataset)**:

  $$\begin{aligned}   X \longrightarrow &\text{Conv2D}(f=8, s=1, p=\text{"same"}) \longrightarrow \text{ReLU} \longrightarrow \text{MaxPool2D}(f=8, s=8, p=\text{"same"}) \\   \longrightarrow &\text{Conv2D}(f=16, s=1, p=\text{"same"}) \longrightarrow \text{ReLU} \longrightarrow \text{MaxPool2D}(f=4, s=4, p=\text{"same"}) \\   \longrightarrow &\text{Flatten} \longrightarrow \text{Dense}(6, \text{activation}=\text{"softmax"})   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Computer Vision Theory & Spatial Tensor Math
- **Receptive Field Dynamics:** Calculating output volume dimensions under "Valid" ($p=0$) and "Same" ($p = \frac{f-1}{2}$) padding configurations across arbitrary stride lengths.

- **Feature Map Dimensionality Tracking:** Managing tensor volume shifts across sequential Conv, Max Pool, and Average Pool layers to prevent spatial information collapse before classification layers.

- **Parameter Optimization:** Exploiting spatial weight sharing to enable scale-invariant feature extraction while maintaining compact parameter footprints.


### 🤖 Applied TensorFlow Engineering
- **NumPy Matrix Slicing:** Constructing high-performance array indexing loops (`a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]`) to isolate spatial sub-tensors for dot product evaluation.

- **Keras Functional Architecture:** Constructing flexible computation graphs using explicit tensor inputs (`tensors_in = tf.keras.Input(shape=...)`) and output mappings (`model = tf.keras.Model(inputs=..., outputs=...)`).

- **Categorical Vision Optimization:** Compiling and training ConvNets using `CategoricalCrossentropy` loss, `Adam` optimizer, mini-batch iterators, and categorical evaluation metrics.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Tensor Mechanics & Math | Vector Visualization | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![NumPy](https://img.shields.io/badge/NumPy-Array_Operations-013243?style=flat&logo=numpy&logoColor=white) | ![Matplotlib](https://img.shields.io/badge/Matplotlib-Feature_Maps-11557c?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |

