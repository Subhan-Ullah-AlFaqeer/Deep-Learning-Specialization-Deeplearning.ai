# 🔄 Week 1: Recurrent Neural Networks, LSTMs, GRUs & Music Generation

Welcome to Week 1 of **Sequence Models** (Course 5 of the **DeepLearning.AI Deep Learning Specialization**)! This module establishes the mathematical foundation for temporal data processing. You will build Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) cells from scratch in NumPy, address vanishing/exploding gradients via gradient clipping, generate novel text using character-level language models, and construct deep LSTM networks in TensorFlow 2.x Keras to compose original jazz music.

---

## 📝 Core Technical Objectives
- **Sequence Notation & RNN Cell Architecture:** Representing sequence inputs $x^{\langle t \rangle}$ and targets $y^{\langle t \rangle}$ across time steps $T_x$ and $T_y$. Processing temporal hidden states $a^{\langle t \rangle} = g_1(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$ and predictions $\hat{y}^{\langle t \rangle} = g_2(W_{ya} a^{\langle t \rangle} + b_y)$.

- **Gated Architectures (GRU & LSTM):** Controlling information flow and long-range dependencies using update gates ($\Gamma_u$), forget gates ($\Gamma_f$), output gates ($\Gamma_o$), and memory cell vectors ($\tilde{c}^{\langle t \rangle}, c^{\langle t \rangle}$):

$$
\begin{aligned}   \tilde{c}^{\langle t \rangle} &= \tanh\left(W_c [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c\right), \quad &\Gamma_u &= \sigma\left(W_u [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u\right) \\   \Gamma_f &= \sigma\left(W_f [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f\right), \quad &\Gamma_o &= \sigma\left(W_o [a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o\right) \\   c^{\langle t \rangle} &= \Gamma_f \odot c^{\langle t-1 \rangle} + \Gamma_u \odot \tilde{c}^{\langle t \rangle}, \quad &a^{\langle t \rangle} &= \Gamma_o \odot \tanh\left(c^{\langle t \rangle}\right)   \end{aligned}$$

- **Character-Level Language Modeling & Sampling:** Training character-level language models on text corpuses (e.g., dinosaur names), performing forward propagation to calculate cross-entropy loss, applying gradient clipping ($\text{clip}(g, -\text{max\_val}, \text{max\_val})$) to prevent exploding gradients, and sampling novel sequences using Softmax probability distributions.

- **Bidirectional & Deep RNN Networks:** Stacking multiple recurrent layers to capture hierarchical sequence representations, and utilizing Bidirectional RNNs (BRNNs) to incorporate past ($a^{\rightarrow \langle t \rangle}$) and future ($a^{\leftarrow \langle t \rangle}$) contextual information simultaneously.

- **Polyphonic Music Generation:** Constructing flexible computation graphs with shared Keras LSTM layers and custom loop structures to sample discrete musical values (pitches, durations) and improvise jazz solos.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, vector NumPy assignments, and Keras music generation notebooks are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C5_W1.pdf)** | Formal visual and mathematical reference covering standard RNN unrolling, Backpropagation Through Time (BPTT), GRU gate mechanisms, 4-gate LSTM cell dynamics, Bidirectional topologies, and Deep RNN structures. |
| **[Building an RNN - Step by Step](./05%20Sequence%20Models%20Sequence%20Models/01_Week/01%20Assingment/Building_a_Recurrent_Neural_Network_Step_by_Step.ipynb)** | Ground-up vector implementation in NumPy: writing `rnn_cell_forward`, `rnn_forward` unrolling, `lstm_cell_forward`, and full `lstm_forward` passes along with explicit BPTT gradient calculations (`rnn_backward`, `lstm_backward`). |
| **[Dinosaur Island Language Model](./05%20Sequence%20Models%20Sequence%20Models/01_Week/02%20Assingment/Dinosaurus_Island_Character_level_language_model.ipynb)** | Building a character-level sequence generator: implementing element-wise gradient clipping (`clip`), Softmax sampling (`sample`), and training an unrolled RNN model via SGD to synthesize novel dinosaur names. |
| **[Jazz Improvisation with LSTM](./05%20Sequence%20Models%20Sequence%20Models/01_Week/03%20Assingment/Improvise_a_Jazz_Solo_with_an_LSTM_Network_v4.ipynb)** | Music synthesis pipeline in TensorFlow Keras: instantiating shared `LSTM()` and `Dense()` layer instances, executing $T_x$-step sequence generation loops using the Functional API, and exporting musical outputs to MIDI format. |

---

## 💡 Visual Pipeline Reference

The forward propagation computational pass across standard RNN cells alongside the 4-gate LSTM cell memory pipeline:

- **Standard RNN Cell Forward Step**:

  $$a^{\langle t \rangle} = \tanh\left(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a\right), \quad \hat{y}^{\langle t \rangle} = \text{softmax}\left(W_{ya} a^{\langle t \rangle} + b_y\right)$$

- **LSTM Cell Internal State Update Sequence**:

$$
\begin{aligned}   \mathbf{x}_{\text{concat}} &= \begin{bmatrix} a^{\langle t-1 \rangle} \\ x^{\langle t \rangle} \end{bmatrix} \\   \Gamma_f^{\langle t \rangle} &= \sigma\left(W_f \mathbf{x}_{\text{concat}} + b_f\right), \quad \Gamma_u^{\langle t \rangle} = \sigma\left(W_u \mathbf{x}_{\text{concat}} + b_u\right) \\   c^{\langle t \rangle} &= \Gamma_f^{\langle t \rangle} \odot c^{\langle t-1 \rangle} + \Gamma_u^{\langle t \rangle} \odot \tanh\left(W_c \mathbf{x}_{\text{concat}} + b_c\right) \\   \Gamma_o^{\langle t \rangle} &= \sigma\left(W_o \mathbf{x}_{\text{concat}} + b_o\right), \quad a^{\langle t \rangle} = \Gamma_o^{\langle t \rangle} \odot \tanh\left(c^{\langle t \rangle}\right)   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Sequence Theory & Recurrent Mathematics
- **Backpropagation Through Time (BPTT):** Deriving partial derivatives of loss functions $L^{\langle t \rangle}$ with respect to unrolled weight matrices ($W_{aa}, W_{ax}, W_{ya}$) across arbitrary sequence lengths.

- **Vanishing vs. Exploding Gradients:** Explaining how repeated matrix multiplications $W_{aa}^{T_x}$ lead to exponentially decaying or exploding gradients, and using gating mechanisms ($c^{\langle t \rangle}$) or gradient clipping to preserve gradient stability.

- **Softmax Temperature & Sequence Sampling:** Converting network output logits into probability vectors $P(x^{\langle t \rangle} \mid x^{\langle 1 \rangle}, \dots, x^{\langle t-1 \rangle})$ and sampling next-token indices using `np.random.choice`.


### 🤖 Applied TensorFlow & Sequence Engineering
- **NumPy Matrix Stacking & Slicing:** Manipulating multi-dimensional array shapes $(n_a, m, T_x)$ to efficiently process batched temporal sequences.

- **Keras Shared Layer Loops:** Re-using layer objects (e.g., `LSTMCell`, `Dense`) across iterative Functional API loops to construct models with shared parameters across sequence steps.

- **Musical Data Encoding:** Mapping musical note pitches, durations, and chords into one-hot target vectors for sequence-to-sequence prediction and MIDI generation.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Tensor Mechanics & Math | Interactive Environment | Audio & MIDI Processing |
| :---: | :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![NumPy](https://img.shields.io/badge/NumPy-Array_Operations-013243?style=flat&logo=numpy&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) | ![Music21](https://img.shields.io/badge/Music21-MIDI_Synthesis-0056D2?style=flat&logo=python&logoColor=white) |
