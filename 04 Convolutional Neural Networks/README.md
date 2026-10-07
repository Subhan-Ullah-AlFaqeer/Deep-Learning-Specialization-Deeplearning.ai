# 👁️ Course 4: Convolutional Neural Networks

Welcome to **Course 4** of the **DeepLearning.AI Deep Learning Specialization**! This course provides a comprehensive exploration of modern Computer Vision, taking you from foundational 2D spatial convolution mathematics to state-of-the-art visual perception pipelines. You will master the building blocks of deep ConvNets, residual architectures, single-shot real-time object detection systems, semantic segmentation models, low-dimensional facial metric learning, and neural synthesis engines.

Across 4 intensive modules, you will construct end-to-end computer vision systems in TensorFlow 2.x and Keras—implementing foundational operations from scratch in NumPy, leveraging pre-trained backbones (MobileNet, ResNet, VGG-19) via transfer learning, and training spatial networks for real-world applications including autonomous driving perception and automated biometric verification.

---

## 📝 Core Technical Objectives
- **Spatial Tensor Operators & Convolutions:** Mastering 2D cross-correlation, zero-padding strategies ("Valid" vs. "Same"), strided window sliding, spatial dimensions calculations, and pooling downsampling mechanics (Max and Average Pooling).

- **Deep Network Architectures & Skip Connections:** Implementing milestone ConvNet paradigms including classic networks (LeNet-5, AlexNet, VGG-16), bottleneck $1 \times 1$ convolutions, Inception modules, Residual Networks (ResNet-50) using skip connections to eliminate vanishing gradients, and depthwise separable convolutions (MobileNetV1/V2).

- **Dense Visual Detection & Segmentation:** Engineering single-shot multi-box object detection pipelines (YOLO v2) with spatial anchor boxes, Intersection over Union (IoU) metrics, and Non-Max Suppression (NMS), alongside pixel-wise semantic segmentation using transpose convolutions and U-Net encoder-decoder skip architectures.

- **Deep Metric Learning & Identity Verification:** Mapping high-dimensional image tensors into compact 128-dimensional Euclidean embedding spaces using Siamese networks and FaceNet backbones, optimizing Triplet Loss with hard-negative mining for one-shot face verification and recognition.

- **Neural Art Synthesis & Optimization:** Implementing Neural Style Transfer using pre-trained VGG-19 feature extractors, formulating L2 content loss and Gram matrix channel cross-correlation style cost functions, and optimizing raw RGB pixel tensors via explicit automatic differentiation.

---

## 🧪 Course Architecture & Module Matrix

This course is structured into 4 sequential modules covering core vision mechanics, advanced research architectures, real-time perception systems, and specialized neural synthesis applications:

| Module / Directory | Operational Focus & Technical Topics | Key Notebooks & Deliverables |
| :--- | :--- | :--- |
| **[01_Week](./01_Week)** | **Foundations of Convolutional Neural Networks**: Spatial 2D convolutions, zero-padding math, strided max/average pooling, Keras Sequential & Functional APIs, parameter sharing, and receptive field dynamics. | • `Convolution_model_Step_by_Step_v1.ipynb`<br>• `Convolution_model_Application.ipynb`<br>• `Yann_LeCun_Interview.mp4` |
| **[02_Week](./02_Week)** | **Deep Convolutional Models: Case Studies & Transfer Learning**: Classic networks (LeNet, AlexNet, VGG-16), ResNet-50 skip connections, $1 \times 1$ bottleneck convolutions, Inception blocks, MobileNet depthwise separable convolutions, EfficientNet scaling, and fine-tuning transfer protocols. | • `Residual_Networks.ipynb`<br>• `Transfer_learning_with_MobileNet_v1.ipynb` |
| **[03_Week](./03_Week)** | **Object Detection, YOLO & U-Net Segmentation**: Spatial grid encoding, bounding box regression, anchor boxes, IoU calculation, Non-Max Suppression (NMS), single-shot YOLO v2 inference, transpose convolutions, and CARLA U-Net semantic segmentation. | • `Autonomous_driving_application_Car_detection.ipynb`<br>• `Image_segmentation_Unet_v2.ipynb` |
| **[04_Week](./04_Week)** | **Special Applications: Face Recognition & Neural Style Transfer**: One-Shot Learning, Siamese networks, 128D FaceNet embeddings, Triplet Loss optimization, VGG-19 feature correlation, Gram matrices, and pixel-level neural style transfer optimization. | • `Face_Recognition.ipynb`<br>• `Art_Generation_with_Neural_Style_Transfer_v1.ipynb` |

---

## 💡 Visual Pipeline Reference

The unified diagnostic equations and mathematical foundations governing Course 4:

- **2D Spatial Convolution Activation & Dimensionality**:

  $$n_H^{[l]} = \left\lfloor \frac{n_H^{[l-1]} + 2p - f}{s} \right\rfloor + 1, \quad n_W^{[l]} = \left\lfloor \frac{n_W^{[l-1]} + 2p - f}{s} \right\rfloor + 1$$

- **Residual Block Shortcut Mapping (ResNet)**:

  $$\mathbf{a}^{[l+2]} = g\left( \mathbf{z}^{[l+2]} + \mathbf{a}^{[l]} \right) = g\left( \mathbf{W}^{[l+2]} \mathbf{a}^{[l+1]} + \mathbf{b}^{[l+2]} + \mathbf{a}^{[l]} \right)$$

- **Triplet Loss Metric Learning Optimization**:

  $$\mathcal{L}(A, P, N) = \max\left( \Vert{}f(A) - f(P)\Vert{}_2^2 - \Vert{}f(A) - f(N)\Vert{}_2^2 + \alpha, \; 0 \right)$$

- **Neural Style Transfer Gram Matrix Correlation & Total Loss**:

  $$G_{k, k'}^{[l]} = \sum_{i=1}^{n_H^{[l]} n_W^{[l]}} a_{i, k}^{[l]} a_{i, k'}^{[l]}, \quad J(G) = \alpha J_{\text{content}}(C, G) + \beta J_{\text{style}}(S, G)$$

---

## 🎯 Technical Skills Architecture

### 📊 Deep Vision Theory & Mathematical Foundations
- **Spatial Feature Map Tracking:** Computing spatial tensor shape alterations across sequential Conv2D, MaxPool2D, and Conv2DTranspose layers.

- **Depthwise Separable Operations:** Decomposing full 3D convolutions into spatial depthwise filters and cross-channel pointwise $1 \times 1$ convolutions to reduce computational complexity by $\sim \frac{1}{f^2}$.

- **Metric Learning Geometry:** Formulating distance functions in latent embedding spaces to solve one-shot verification problems without re-training target networks.


### 🤖 Applied TensorFlow & Computer Vision Engineering
- **End-to-End YOLO Processing:** Building multi-stage prediction filtering algorithms combining class confidence thresholding, IoU evaluation, and GPU-accelerated Non-Max Suppression (`tf.image.non_max_suppression`).

- **Dense Segmentation Architectures:** Constructing expansive decoder networks using `Conv2DTranspose` layers concatenated with contracting encoder skip features (`tf.keras.layers.concatenate`).

- **Custom Gradient Synthesis:** Executing direct pixel updates on input images via `tf.GradientTape()` to synthesize artistic imagery from feature loss targets.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Computer Vision Engine | Pre-trained Models | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![OpenCV](https://img.shields.io/badge/Vision-YOLO_v2_&_UNet-0056D2?style=flat&logo=opencv&logoColor=white) | ![Keras Apps](https://img.shields.io/badge/Keras_Apps-ResNet_MobileNet_VGG-D00000?style=flat&logo=keras&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
