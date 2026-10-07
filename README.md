# 🚀 DeepLearning.AI: Deep Learning Specialization

Welcome to the root repository for the **DeepLearning.AI Deep Learning Specialization**, instructed by Andrew Ng! This repository serves as a production-grade, fully vectorized, copy-paste intact portfolio containing all theoretical foundations, mathematical derivations, Jupyter notebook implementations, and specialized neural network architectures across the entire 5-course series.

Master the mechanics of modern Artificial Intelligence—from scalar single-neuron backpropagation and deep fully-connected networks to residual computer vision backbones, single-shot real-time object detection, sequence attention mechanisms, and fine-tuned Hugging Face Transformer models.

---

## 📝 Program Architecture & Core Objectives

Across 5 comprehensive courses, this specialization provides a complete pathway for mastering deep learning theory and production deployment in Python, TensorFlow 2.x, and PyTorch:

- **Vectorized Neural Fundamentals:** Implementing dynamic forward and backward propagation algorithms entirely from scratch in NumPy, mastering gradient descent optimization, and understanding hyperparameter tuning mechanics.
- **Regularization & Optimization Dynamics:** Mitigating high bias/variance regimes using $L_2$ regularization, Dropout, Xavier/He weight initializations, Batch Normalization, and advanced optimizers (RMSprop, Adam).
- **Machine Learning System Strategy:** Structuring machine learning workflows, conducting systematic error analysis, setting up orthogonal metrics ($F_1$, satisficing/optimizing metrics), and leveraging Transfer, Multi-Task, and End-to-End Learning.
- **Computer Vision & Spatial Models:** Engineering 2D Convolutional Neural Networks (ConvNets), residual skip architectures (ResNet-50), single-shot object detection (YOLO v2), semantic segmentation (U-Net), Siamese metric learning, and Neural Style Transfer.
- **Sequence Models & Attention Topologies:** Building Recurrent Neural Networks (RNNs, GRUs, LSTMs), training character-level language generators, constructing dense vector embeddings (GloVe, Word2Vec), implementing Bahdanau additive attention, and subclassing multi-head Transformer networks.

---

## 🧪 Specialization Curriculum & Course Matrix

The specialization is organized into 5 sequential courses, each targeting a critical domain of modern deep learning engineering:

| Course / Directory | Core Technical Focus | Primary Deliverables & Key Technologies |
| :--- | :--- | :--- |
| **[01_Neural_Networks_and_Deep_Learning](./Course%201%20-%20Neural%20Networks%20and%20Deep%20Learning)** | **Foundations of Deep Learning**: Logistic regression as a neural network, vectorization, hidden layer activation functions ($\text{ReLU}, \text{Sigmoid}, \text{tanh}$), and $L$-layer deep network backpropagation from scratch in NumPy. | • `Planar_data_classification_with_one_hidden_layer.ipynb`<br>• `Building_your_Deep_Neural_Network_Step_by_Step.ipynb`<br>• `Deep_Neural_Network_Application_Image_Classification.ipynb` |
| **[02_Improving_Deep_Neural_Networks](./Course%202%20-%20Improving%20Deep%20Neural%20Networks)** | **Hyperparameter Tuning, Regularization & Optimization**: Train/Dev/Test splits, Bias/Variance trade-offs, $L_2$ regularization, Inverted Dropout, Gradient Checking, Momentum, RMSprop, Adam, Learning Rate Decay, and TensorFlow 2.x Keras basics. | • `Initialization.ipynb`<br>• `Regularization.ipynb`<br>• `Gradient_Checking.ipynb`<br>• `Optimization_methods.ipynb`<br>• `TensorFlow_tutorial.ipynb` |
| **[03_Structuring_Machine_Learning_Projects](./Course%203%20-%20Structuring%20Machine%20Learning%20Projects)** | **ML Strategy & Error Analysis**: Orthogonalization, single-number evaluation metrics, train/dev/test distribution mismatches, error analysis tables, ceiling analysis, and strategic end-to-end vs. component learning choices. | • `Autonomous_Driving_Error_Analysis_Case_Study.md`<br>• `Machine_Learning_Flight_Simulator.ipynb` |
| **[04_Convolutional_Neural_Networks](./Course%204%20-%20Convolutional%20Neural%20Networks)** | **Computer Vision Pipelines**: 2D spatial convolutions, pooling layers, ResNet-50 skip connections, MobileNet depthwise separable convolutions, YOLO v2 car detection, U-Net image segmentation, FaceNet 128D embeddings, and Neural Style Transfer. | • `Convolution_model_Step_by_Step_v1.ipynb`<br>• `Residual_Networks.ipynb`<br>• `Autonomous_driving_application_Car_detection.ipynb`<br>• `Image_segmentation_Unet_v2.ipynb`<br>• `Face_Recognition.ipynb` |
| **[05_Sequence_Models](./Course%205%20-%20Sequence%20Models)** | **Temporal Models, NLP & Transformers**: RNN/LSTM/GRU cells from scratch, character-level dinosaur language generator, Keras jazz music composer, GloVe debiased embeddings, Bahdanau attention NMT, trigger word detection, and Hugging Face Transformers. | • `Building_a_Recurrent_Neural_Network_Step_by_Step.ipynb`<br>• `Operations_on_word_vectors_v2a.ipynb`<br>• `Neural_machine_translation_with_attention_v4a.ipynb`<br>• `C5_W4_A1_Transformer_Subclass_v1.ipynb` |

---

## 💡 Visual Pipeline Reference

The unified diagnostic equations governing the 5-course specialization architecture:

- **General Backpropagation Gradient Step & Adam Optimizer Updates**:

  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\frac{v_t}{1 - \beta_2^t}} + \epsilon} \frac{m_t}{1 - \beta_1^t}$$

- **Residual Network Skip Connection Mapping (Course 4)**:

  $$\mathbf{a}^{[l+2]} = g\left( \mathbf{z}^{[l+2]} + \mathbf{a}^{[l]} \right) = g\left( \mathbf{W}^{[l+2]} \mathbf{a}^{[l+1]} + \mathbf{b}^{[l+2]} + \mathbf{a}^{[l]} \right)$$

- **Scaled Dot-Product Multi-Head Self-Attention (Course 5)**:

  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \quad \text{where} \quad \text{head}_i = \text{softmax}\left( \frac{Q_i K_i^T}{\sqrt{d_k}} + M \right) V_i$$

---

## 🎯 Technical Skills Architecture

### 📊 Deep Learning Theory & Mathematical Foundations
- **Vectorized Computation Calculus:** Computing exact partial derivatives $\frac{\partial \mathcal{L}}{\partial \mathbf{W}}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}}$ across vectorized multi-layer matrices using matrix chain rule operations.

- **Spatial & Temporal Modeling Physics:** Formulating spatial receptive fields in ConvNets alongside time-unrolled memory states ($c^{\langle t \rangle}, a^{\langle t \rangle}$) in gated recurrent networks to solve temporal credit assignment problems.

- **Attention & Latent Vector Spaces:** Projecting high-dimensional inputs into compact continuous metric spaces (128D FaceNet, 50D GloVe, Multi-Head Transformer attention projections).


### 🤖 Production Machine Learning Engineering
- **End-to-End Deep Learning Frameworks:** Constructing custom models using TensorFlow 2.x Keras Functional/Subclassing APIs alongside PyTorch and Hugging Face ecosystems (`transformers`, `datasets`, `evaluate`).

- **Computer Vision & Signal Processing:** Processing 2D/3D visual tensors with OpenCV and converting 1D acoustic audio waveforms into 2D time-frequency spectrograms using PyDub and STFTs.

- **Systematic ML Operations:** Designing rigorous error analysis pipelines, optimizing inference speed via depthwise separable convolutions, and fine-tuning pre-trained models via transfer learning.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Frameworks | Computer Vision & Audio | NLP & Transformer Ecosystem | Environment & Math |
| :---: | :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-1.x_&_2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) | ![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat&logo=opencv&logoColor=white) ![PyDub](https://img.shields.io/badge/PyDub-Audio_Spectrograms-0056D2?style=flat&logo=python&logoColor=white) | ![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers_&_Tokenizers-FFD21E?style=flat&logo=huggingface&logoColor=black) | ![NumPy](https://img.shields.io/badge/NumPy-Vectorized_Math-013243?style=flat&logo=numpy&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Notebooks-FA0F00?style=flat&logo=jupyter&logoColor=white) |
