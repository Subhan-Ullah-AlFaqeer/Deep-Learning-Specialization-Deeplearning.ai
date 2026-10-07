
# 🏛️ Week 2: Deep Convolutional Models: Case Studies & Transfer Learning

Welcome to Week 2 of **Convolutional Neural Networks** (Course 4 of the **DeepLearning.AI Deep Learning Specialization**)! This module synthesizes foundational research-paper architectures—from classic networks (LeNet-5, AlexNet, VGG-16) to modern residual networks (ResNet-50), bottleneck $1 \times 1$ convolutions, Inception modules, depthwise separable convolutions (MobileNetV1/V2), and Compound Scaling (EfficientNet). You will apply these architectural concepts in TensorFlow 2.x Keras by building skip connections from scratch, engineering automated data augmentation pipelines, and adapting pre-trained MobileNet models via fine-tuning for custom vision applications.

---

## 📝 Core Technical Objectives
- **Classic Architectures & Evolution:** Tracking the evolution of computer vision networks from LeNet-5 ($f=5, s=1 \to \text{AvgPool}$), AlexNet ($11 \times 11$ strided filters with ReLU and Local Response Normalization), to VGG-16 ($3 \times 3$ convolutions with stride 1 and $2 \times 2$ max pooling with stride 2).

- **Residual Skip Connections & Vanishing Gradients:** Bypassing vanishing gradient bottlenecks in deep networks ($L > 50$) using residual identity blocks $\mathbf{a}^{[l+2]} = g\left(\mathbf{z}^{[l+2]} + \mathbf{a}^{[l]}\right)$ and convolutional blocks (applying $1 \times 1$ projections to match spatial dimensions when $s \ne 1$).

- **$1 \times 1$ Convolutions & Inception Bottlenecks:** Leveraging $1 \times 1$ convolutions to shrink or expand channel volume dimensions without altering spatial resolution, reducing computational bottleneck costs within Inception modules.

- **Depthwise Separable Convolutions & MobileNet:** Reducing floating-point operations (FLOPs) by splitting standard 3D convolutions into a two-stage process: **Depthwise Convolution** (per-channel spatial filtering) followed by **Pointwise Convolution** ($1 \times 1$ cross-channel combination), cutting computational requirements by approximately $\frac{1}{N} + \frac{1}{f^2}$.

- **Data Pipelines & Transfer Learning Protocols:** Utilizing `tf.keras.utils.image_dataset_from_directory` to stream batches, applying Keras preprocessing layers (`RandomFlip`, `RandomRotation`) directly into data graphs, and executing two-stage transfer learning: freezing base features for initial head training, followed by unfreezing top layers for low learning rate fine-tuning.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, residual network implementations, and pre-trained MobileNet projects are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C4_W2.pdf)** | Formal visual and structural reference covering classic networks, ResNet identity/convolutional blocks, Inception bottlenecks, depthwise separable convolutions, and EfficientNet compound scaling vectors. |
| **[Residual Networks Assignment](./04%20Convolutional%20Neural%20Networks/02_Week/01%20Assingment/Residual_Networks.ipynb)** | Step-by-step implementation of ResNet-50 in TensorFlow Keras: building `identity_block` (shortcut connections) and `convolutional_block` ($1 \times 1$ projection shortcuts) to train a 50-layer deep sign language digit classifier. |
| **[Transfer Learning with MobileNet](./04%20Convolutional%20Neural%20Networks/02_Week/02%20Assingment/Transfer_learning_with_MobileNet_v1.ipynb)** | End-to-end computer vision workflow: streaming alpaca vs. non-alpaca image directories, engineering Keras data augmentation pipelines, loading pre-trained MobileNetV2 weights, training custom classification heads, and unfreezing top layers for fine-tuning. |

---

## 💡 Visual Pipeline Reference

The mathematical shortcut structure of ResNet skip connections alongside Depthwise Separable Convolution mechanics:

- **Residual Block Shortcut Mapping**:

  $$\mathbf{z}^{[l+1]} = \mathbf{W}^{[l+1]} \mathbf{a}^{[l]} + \mathbf{b}^{[l+1]} \quad \longrightarrow \quad \mathbf{a}^{[l+1]} = g\left(\mathbf{z}^{[l+1]}\right)$$
  $$\mathbf{z}^{[l+2]} = \mathbf{W}^{[l+2]} \mathbf{a}^{[l+1]} + \mathbf{b}^{[l+2]} \quad \longrightarrow \quad \mathbf{a}^{[l+2]} = g\left(\mathbf{z}^{[l+2]} + \mathbf{a}^{[l]}\right)$$

- **Depthwise Separable Convolution Decomposition**:

  $$\text{Standard Conv Cost}: n_H \cdot n_W \cdot f^2 \cdot n_C^{[l-1]} \cdot n_C^{[l]}$$
  $$\text{Depthwise + Pointwise Cost}: n_H \cdot n_W \cdot n_C^{[l-1]} \left( f^2 + n_C^{[l]} \right)$$
  $$\text{Computational Reduction Ratio}: \frac{\text{Separable Cost}}{\text{Standard Cost}} = \frac{1}{n_C^{[l]}} + \frac{1}{f^2}$$

---

## 🎯 Technical Skills Architecture

### 📊 Deep Vision Theory & Research Architecture
- **Residual Identity Mechanics:** Proving why residual skip connections preserve signal propagation by allowing the identity function $\mathbf{a}^{[l+2]} = \mathbf{a}^{[l]}$ to be easily learned when weights $\mathbf{W} \to 0$.

- **Inception Bottleneck Optimization:** Applying $1 \times 1$ convolutions before computationally expensive $3 \times 3$ and $5 \times 5$ filters to reduce total floating-point operations while preserving feature capacity.

- **Compound Model Scaling:** Balancing network depth, width, and image resolution using EfficientNet scaling coefficients ($\phi$) to maximize accuracy within strict compute budgets.


### 🤖 Applied TensorFlow Engineering
- **Keras Add Layer Integration:** Constructing functional DAG skip connections using `tf.keras.layers.Add()([X, X_shortcut])` within custom modular network functions.

- **Dynamic Data Augmentation Pipelines:** Building preprocessing graphs using `tf.keras.Sequential` with `RandomFlip("horizontal")` and `RandomRotation(0.2)` layers to execute on-the-fly GPU tensor transformations.

- **Two-Stage Fine-Tuning Execution:** Toggling parameter trainability (`base_model.trainable = False`), compiling classification heads, and subsequently setting `base_model.trainable = True` for target layers with reduced learning rates ($\alpha \approx 10^{-5}$).


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Pre-trained Models | Interactive Environment |
| :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![Keras Applications](https://img.shields.io/badge/Keras_Apps-ResNet_&_MobileNet-D00000?style=flat&logo=keras&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |

