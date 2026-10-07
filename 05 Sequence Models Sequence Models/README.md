# 🔤 Course 5: Sequence Models

Welcome to **Course 5** of the **DeepLearning.AI Deep Learning Specialization**! This course provides a comprehensive exploration of sequential and temporal data modeling—from classical recurrent architectures to modern attention-driven Transformer networks. You will master the mechanisms underlying natural language processing (NLP), speech recognition, polyphonic music generation, time-series modeling, and neural machine translation.

Across 4 intensive modules, you will construct end-to-end sequence processing pipelines in TensorFlow 2.x, Keras, and PyTorch—building Recurrent Neural Networks (RNNs), Gated Recurrent Units (GRUs), and Long Short-Term Memory (LSTM) cells from scratch in NumPy, engineering dense vector word embeddings (Word2Vec, GloVe), implementing dynamic Bahdanau attention mechanisms, and fine-tuning state-of-the-art Hugging Face Transformer models (BERT, DistilBERT) for downstream NLP tasks.

---

## 📝 Core Technical Objectives
- **Recurrent Dynamics & Gated Architectures:** Mastering temporal recurrence $a^{\langle t \rangle} = g(W_{aa}a^{\langle t-1 \rangle} + W_{ax}x^{\langle t \rangle} + b_a)$, Backpropagation Through Time (BPTT), gradient clipping for exploding gradients, and multi-gate memory mechanics in GRUs and LSTMs to preserve long-range dependencies.

- **Character Language Modeling & Polyphonic Synthesis:** Training character-level language models for text generation, sampling from output Softmax probability distributions, and composing polyphonic jazz music by constructing shared-parameter Keras LSTM loops.

- **Dense Word Vector Spaces & Bias Neutralization:** Learning distributed word representations (Skip-Gram, Negative Sampling, GloVe), evaluating semantic similarity via Cosine Similarity, solving structural word analogies, and applying geometric subspace projections to remove demographic bias.

- **Dynamic Attention & Audio Signal Processing:** Building Sequence-to-Sequence models with additive Bahdanau attention for Neural Machine Translation (NMT), optimizing decoding via Beam Search with length normalization, evaluating translations using BLEU scores, and processing audio spectrograms for trigger word detection.

- **Transformer Architectures & Transfer Learning:** Constructing Positional Encodings, Scaled Dot-Product Self-Attention, Causal Look-Ahead Masking, and Multi-Head Attention blocks from scratch via Keras subclassing, alongside fine-tuning Hugging Face Transformers for Named Entity Recognition (NER) and extractive Question Answering (QA).

---

## 🧪 Course Architecture & Module Matrix

This course is structured into 4 sequential modules covering foundational recurrent models, vector embedding representations, attention-based seq2seq systems, and state-of-the-art Transformer networks:

| Module / Directory | Operational Focus & Technical Topics | Key Notebooks & Deliverables |
| :--- | :--- | :--- |
| **[01_Week](./01_Week)** | **Recurrent Neural Networks, LSTMs & GRUs**: RNN/LSTM math from scratch in NumPy, Backpropagation Through Time (BPTT), character-level language modeling, gradient clipping, Softmax sampling, Bidirectional/Deep RNN topologies, and Keras LSTM music generation. | • `Building_a_Recurrent_Neural_Network_Step_by_Step.ipynb`<br>• `Dinosaurus_Island_Character_level_language_model.ipynb`<br>• `Improvise_a_Jazz_Solo_with_an_LSTM_Network_v4.ipynb` |
| **[02_Week](./02_Week)** | **Natural Language Processing & Word Embeddings**: Word vector spaces, Cosine Similarity, Word2Vec Skip-Gram with Negative Sampling, GloVe co-occurrence matrix optimization, geometric bias neutralization, and deep LSTM sentiment classification (Emojifier). | • `Operations_on_word_vectors_v2a.ipynb`<br>• `Emoji_v3a.ipynb` |
| **[03_Week](./03_Week)** | **Sequence Models & Attention Mechanism**: Encoder-Decoder seq2seq pipelines, Beam Search decoding with length normalization, BLEU score evaluation, Bahdanau additive attention for NMT, audio spectrogram processing, and trigger word detection. | • `Neural_machine_translation_with_attention_v4a.ipynb`<br>• `Trigger_word_detection_v1a.ipynb` |
| **[04_Week](./04_Week)** | **Transformer Networks & Transfer Learning**: Positional encodings, Scaled Dot-Product Self-Attention, Multi-Head Attention (MHA) subclassing, causal masking, complete Transformer Encoder/Decoder, and Hugging Face fine-tuning for NER and Question Answering. | • `C5_W4_A1_Transformer_Subclass_v1.ipynb`<br>• `Transformer_Network_Application_Named_Entity_Recognition.ipynb`<br>• `Transformer_Network_Application_Question_Answering.ipynb` |

---

## 💡 Visual Pipeline Reference

The unified diagnostic equations and mathematical foundations governing Course 5:

- **LSTM Cell Internal Gated Memory Updates**:

$$\begin{aligned}   \mathbf{x}_{\text{concat}} &= \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} \\   \Gamma_f^{\langle t \rangle} &= \sigma\left(W_f \mathbf{x}_{\text{concat}} + b_f\right), \quad \Gamma_u^{\langle t \rangle} = \sigma\left(W_u \mathbf{x}_{\text{concat}} + b_u\right) \\   c^{\langle t \rangle} &= \Gamma_f^{\langle t \rangle} \odot c^{\langle t-1 \rangle} + \Gamma_u^{\langle t \rangle} \odot \tanh\left(W_c \mathbf{x}_{\text{concat}} + b_c\right) \\   \Gamma_o^{\langle t \rangle} &= \sigma\left(W_o \mathbf{x}_{\text{concat}} + b_o\right), \quad a^{\langle t \rangle} = \Gamma_o^{\langle t \rangle} \odot \tanh\left(c^{\langle t \rangle}\right)   \end{aligned}$$

- **Vector Embedding Bias Neutralization Projection**:

$$e{\text{bias\component}} = \frac{e \cdot g}{\Vert{}g\Vert{}2^2} \cdot g \quad \implies \quad e^{\perp} = e - e{\text{bias\component}}$$

- **Bahdanau Additive Attention Context Vector**:

$$e^{\langle t, t' \rangle} = \text{Dense}\left(\left[s^{\langle t-1 \rangle}, a^{\langle t' \rangle}\right]\right), \quad \alpha^{\langle t, t' \rangle} = \text{Softmax}\left(e^{\langle t, t' \rangle}\right), \quad c^{\langle t \rangle} = \sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}$$

- **Scaled Dot-Product Multi-Head Self-Attention**:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \quad \text{where} \quad \text{head}_i = \text{softmax}\left( \frac{Q_i K_i^T}{\sqrt{d_k}} + M \right) V_i$$

---

## 🎯 Technical Skills Architecture

### 📊 Sequence Theory & Natural Language Mathematics
- **Recurrent Gradient Stability:** Understanding backpropagation through time dynamics, diagnosing vanishing/exploding gradients, and applying gating mechanisms or gradient clipping.

- **High-Dimensional Vector Geometry:** Computing spatial similarity metrics, deriving orthogonal projections for embedding debiasing, and leveraging GloVe co-occurrence matrices.

- **Attention & Transformer Dynamics:** Formulating scaled dot-product queries, keys, and values, applying causal masks for autoregressive decoders, and encoding positional order via sinusoidal functions.


### 🤖 Applied TensorFlow, PyTorch & NLP Engineering
- **Custom Keras Subclassing & Layer Design:** Implementing custom Transformer layers (`MultiHeadAttention`, `PositionalEncoding`) and end-to-end models using TensorFlow Keras Subclassing API.

- **Audio Spectrogram Feature Extraction:** Processing 1D raw audio waveforms into 2D spectrograms using PyDub and STFTs for 1D Conv-GRU trigger word models.

- **Hugging Face Ecosystem Integration:** Utilizing `transformers`, `datasets`, and `evaluate` libraries to tokenize, align subtokens, and fine-tune pre-trained models for NER and SQuAD Question Answering.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | NLP & Transformer Ecosystem | Signal & Audio Processing | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers_&_Datasets-FFD21E?style=flat&logo=huggingface&logoColor=black) | ![PyDub](https://img.shields.io/badge/PyDub-Audio_&_MIDI-0056D2?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
