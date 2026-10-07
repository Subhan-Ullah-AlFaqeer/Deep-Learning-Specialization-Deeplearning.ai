# ⚡ Week 4: Transformer Networks, Self-Attention & Hugging Face Fine-Tuning

Welcome to Week 4 of **Sequence Models** (Course 5 of the **DeepLearning.AI Deep Learning Specialization**)! This module covers modern attention-based architectures, transitioning from recurrent networks to the Transformer architecture (*"Attention Is All You Need"*). You will implement Positional Encodings, Scaled Dot-Product Attention, Causal Look-Ahead Masking, and Multi-Head Attention blocks from scratch in TensorFlow 2.x Keras via model subclassing. Additionally, you will explore downstream transfer learning using pre-trained Hugging Face Transformers (`transformers`, `datasets`, `evaluate`) for Named Entity Recognition (NER) and extractive Question Answering (QA).

---

## 📝 Core Technical Objectives
- **Positional Encoding Mechanics:** Injecting positional order into non-recurrent parallel inputs using deterministic sinusoidal functions across embedding dimension $d_{\text{model}}$:

  $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{\text{model}}}}}\right)$$

- **Scaled Dot-Product Self-Attention:** Mapping Query ($Q$), Key ($K$), and Value ($V$) projection matrices to output representations, scaling by $\sqrt{d_k}$ to prevent soft-max gradient saturation at high dimensions, and applying causal masks $M$ for autoregressive decoder generation:

  $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V$$

- **Multi-Head Attention (MHA) Parallelism:** Projecting queries, keys, and values into $h$ distinct subspace heads ($d_v = d_k = d_{\text{model}} / h$), computing scaled dot-product attention in parallel, and concatenating outputs:

  $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \quad \text{where} \quad \text{head}_i = \text{Attention}\left(Q W_i^Q, K W_i^K, V W_i^V\right)$$

- **Complete Transformer Architecture:** Stacking Encoder blocks (Multi-Head Attention, Residual Connections, Layer Normalization, Position-wise Feed-Forward Networks) and Decoder blocks (Masked Causal Multi-Head Attention, Encoder-Decoder Cross-Attention) for sequence tasks.

- **Transfer Learning & Downstream Fine-Tuning:** Leveraging pre-trained Transformer models (e.g., DistilBERT, BERT) via Hugging Face `TFAutoModelForTokenClassification` and `TFAutoModelForQuestionAnswering` for Named Entity Recognition token classification and extractive span prediction.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, custom subclassed Transformer notebooks, and Hugging Face fine-tuning application labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C5_W4.pdf)** | Formal visual reference covering self-attention $Q, K, V$ matrix operations, multi-head projection splitting, positional sine/cosine wave encoding, Transformer Encoder-Decoder block diagrams, and BERT/GPT paradigms. |
| **[Transformer Architecture Subclassing](./05%20Sequence%20Models%20Sequence%20Models/04_Week/01%20Assingment/C5_W4_A1_Transformer_Subclass_v1.ipynb)** | End-to-end TensorFlow implementation using Keras Subclassing: building `PositionalEncoding`, `ScaledDotProductAttention`, `MultiHeadAttention`, `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, and complete `Transformer` models. |
| **[Transformer Network Application: NER](./05%20Sequence%20Models%20Sequence%20Models/04_Week/Transformer_Network_Application_Named_Entity_Recognition.ipynb)** | Fine-tuning pre-trained DistilBERT for token-level classification: aligning word-piece subtoken labels, configuring `DataCollatorForTokenClassification`, and evaluating precision/recall/F1 metrics. |
| **[Transformer Network Application: QA](./05%20Sequence%20Models%20Sequence%20Models/04_Week/Transformer_Network_Application_Question_Answering.ipynb)** | Implementing extractive Question Answering on SQuAD: tokenizing question-context pairs, mapping start/end answer character positions to token offsets, and fine-tuning Hugging Face models in TensorFlow and PyTorch. |

---

## 💡 Visual Pipeline Reference

The Scaled Dot-Product Attention matrix computational flow alongside the complete Transformer Encoder-Decoder architecture:

- **Scaled Dot-Product Attention Pipeline**:

$$\begin{aligned}   \begin{bmatrix} Q \\ K \\ V \end{bmatrix} &\longrightarrow S = \frac{Q K^T}{\sqrt{d_k}} \longrightarrow \text{Apply Mask } M \text{ (if causal)} \\   &\longrightarrow \mathbf{A} = \text{Softmax}(S + M) \longrightarrow \text{Output} = \mathbf{A} \cdot V   \end{aligned}$$

- **Transformer Block & Sublayer Pipeline**:

$$\begin{aligned}   \mathbf{X} &\longrightarrow \mathbf{X} + \text{PE} \longrightarrow \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X}) \longrightarrow \text{Add \ LayerNorm} \\   \longrightarrow \text{PositionwiseFeedForward}(\text{ReLU}(W_1 x + b_1) W_2 + b_2) \longrightarrow \text{Add \ LayerNorm} \longrightarrow \mathbf{H}_{\text{encoder}}   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Transformer Mathematics & Self-Attention Theory
- **Sequence Parallelization Dynamics:** Explaining how self-attention eliminates recurrent sequential bottlenecks ($O(1)$ sequential operations vs $O(T_x)$ in RNNs) to enable parallel training over long sequences.

- **Causal Look-Ahead Masking Math:** Setting upper-triangular elements of the attention logit matrix to $-\infty$ (or $-1\text{e}9$) to ensure decoder predictions at time step $t$ depend only on tokens $1 \dots t-1$.

- **Subtoken Label Alignment:** Resolving label misalignment in token classification when tokenizers split words into sub-word tokens by applying special label masks (e.g., `-100`).


### 🤖 Applied TensorFlow & Hugging Face Engineering
- **TensorFlow Keras Subclassing:** Overriding `tf.keras.layers.Layer` and `tf.keras.Model` methods (`build()`, `call()`) to implement custom multi-head projection layers and attention blocks.

- **Multi-Head Tensor Reshaping:** Splitting feature dimensions into multiple heads using `tf.reshape` and swapping axes via `tf.transpose` to run parallel matrix multiplication: `(batch_size, num_heads, seq_len, depth)`.

- **Hugging Face Ecosystem Integration:** Utilizing `AutoTokenizer`, `TFAutoModel`, `DataCollatorWithPadding`, and the `Trainer` / `Native TF2` training loops for NLP tasks.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Transformer Library | Tokenization & Datasets | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black) | ![Datasets](https://img.shields.io/badge/Hugging_Face-Datasets_&_Evaluate-0056D2?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
