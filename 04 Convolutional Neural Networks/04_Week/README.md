# 🎨 Week 4: Special Applications: Face Recognition & Neural Style Transfer

Welcome to Week 4 of **Convolutional Neural Networks** (Course 4 of the **DeepLearning.AI Deep Learning Specialization**)! This module explores two milestone computer vision applications: 128-dimensional spatial face embedding verification (FaceNet/Siamese Networks) and arbitrary image neural synthesis (Neural Style Transfer). You will implement One-Shot Learning with Triplet Loss for identity verification, and construct total cost functions optimization graphs incorporating Gram matrix style representations to transfer artistic style onto target content images.

---

## 📝 Core Technical Objectives
- **Face Verification vs. Recognition Architecture:** Distinguishing binary 1-to-1 matching $\text{Verification}: \text{d}(x^{(1)}, x^{(2)})\le\tau$ from $1\text{-to-}N$ database searching $\text{Recognition}: \text{Match } x \text{ across } N \text{ identities}$.

- **One-Shot Learning & Siamese Networks:** Overcoming small-sample classification limits via metric learning, embedding input images into low-dimensional vector representations $f(x^{(i)}) \in \mathbb{R}^{128}$ using shared-weight deep ConvNet backbones (Inception/FaceNet).

- **Triplet Loss Mechanics:** Training embedding functions $f(x)$ by optimizing spatial distances across Anchor ($A$), Positive ($P$), and Negative ($N$) image triplets using margin parameter $\alpha$:

  $$\mathcal{L}(A, P, N) = \max\left( \Vert{}f(A) - f(P)\Vert{}_2^2 - \Vert{}f(A) - f(N)\Vert{}_2^2 + \alpha, \; 0 \right)$$

- **Content Cost Function in Style Transfer:** Quantifying structural content similarity at hidden layer $l$ using L2 loss between content tensor $A^{(C)}$ and generated tensor $A^{(G)}$:

  $$J_{\text{content}}(C, G) = \frac{1}{4 \cdot n_H^{[l]} \cdot n_W^{[l]} \cdot n_C^{[l]}} \sum_{i, j, k} \left( A_{i, j, k}^{(C)[l]} - A_{i, j, k}^{(G)[l]} \right)^2$$

- **Gram Matrix & Style Cost Function:** Encoding artistic style (channel feature correlations) via un-normalized Gram matrices $G_{k, k'}^{[l]} = \sum_{i=1}^{n_H^{[l]} n_W^{[l]}} a_{i, k}^{[l]} a_{i, k'}^{[l]}$, evaluating layer style cost $J_{\text{style}}^{[l]}(S, G)$, and computing total loss:

  $$J(G) = \alpha J_{\text{content}}(C, G) + \beta J_{\text{style}}(S, G)$$

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, FaceNet embedding verification notebooks, and Neural Style Transfer art generation pipelines are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C4_W4.pdf)** | Formal visual reference covering Siamese network weight-sharing graphs, hard triplet selection mechanics, deep feature activation maps, and Gram matrix spatial correlation math. |
| **[Face Recognition Assignment](./04%20Convolutional%20Neural%20Networks/04_Week/Face_Recognition.ipynb)** | Implementing FaceNet in TensorFlow Keras: building Triplet Loss (`triplet_loss`), mapping image inputs to 128-dimensional encodings, executing 1-to-1 Face Verification (`verify`), and 1-to-N Face Recognition (`who_is_it`). |
| **[Art Generation with Neural Style Transfer Assignment](./04%20Convolutional%20Neural%20Networks/04_Week/Art_Generation_with_Neural_Style_Transfer_v1.ipynb)** | Building an artistic synthesis engine with pre-trained VGG-19: computing content cost (`compute_content_cost`), Gram matrices (`gram_matrix`), layer style costs (`compute_layer_style_cost`), and optimizing generated pixel tensors $G$ via `tf.GradientTape`. |

---

## 💡 Visual Pipeline Reference

The mathematical structure of Triplet Loss alongside Neural Style Transfer optimization:

- **Triplet Loss Optimization Distance Boundary**:

$$\Vert{}f(A) - f(P)\Vert{}_2^2 + \alpha \le \Vert{}f(A) - f(N)\Vert{}_2^2 \quad \implies \quad \Vert{}f(A) - f(P)\Vert{}_2^2 - \Vert{}f(A) - f(N)\Vert{}_2^2 + \alpha \le 0$$

- **Neural Style Transfer Total Gradient Optimization**:

$$
\begin{aligned}   \mathbf{G}_{\text{pixels}} &\leftarrow \mathbf{G}_{\text{pixels}} - \eta \cdot \nabla_{\mathbf{G}} J(\mathbf{G}) \\   J(\mathbf{G}) &= \alpha J_{\text{content}}(C, \mathbf{G}) + \beta \sum_{l} w^{[l]} J_{\text{style}}^{[l]}(S, \mathbf{G})   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Deep Metric Learning & Feature Correlation Theory
- **Metric Distance Learning:** Mapping input pixel manifolds into Euclidean spaces where intra-class variance is minimized and inter-class distance exceeds margin threshold $\alpha$.

- **Hard Triplet Selection:** Proving why selecting online triplets where $\Vert{}f(A) - f(P)\Vert{}_2^2 \approx \Vert{}f(A) - f(N)\Vert{}_2^2$ is necessary to prevent vanishing gradients during FaceNet training.

- **Gram Matrix Channel Cross-Correlation:** Capturing spatial style co-occurrence by computing inner products between feature map channel activations across input dimensions.


### 🤖 Applied TensorFlow Engineering
- **L2 Vector Normalization:** Normalizing feature embedding output vectors using `tf.math.l2_normalize(x, axis=-1)` for invariant Euclidean distance calculation.

- **Pixel Optimization via Automatic Differentiation:** Managing trainable pixel state tensors (`tf.Variable(initial_image)`) and optimizing raw RGB values directly using `tf.GradientTape()` gradient updates.

- **Pre-trained Feature Extraction:** Extracting intermediate feature representations from pre-trained VGG-19 models using custom output layer subsets (`tf.keras.Model(inputs, outputs)`).


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Pre-trained Models | Interactive Environment |
| :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![Pre-trained Networks](https://img.shields.io/badge/VGG19_&_FaceNet-Deep_Features-0056D2?style=flat&logo=keras&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
