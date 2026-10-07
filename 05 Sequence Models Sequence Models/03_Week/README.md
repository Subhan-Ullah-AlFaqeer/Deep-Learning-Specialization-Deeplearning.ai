# 🎯 Week 3: Sequence Models & Attention Mechanism

Welcome to Week 3 of **Sequence Models** (Course 5 of the **DeepLearning.AI Deep Learning Specialization**)! This module transitions from standard sequence-to-sequence encoder-decoder pipelines to modern dynamic attention mechanisms and audio processing architectures. You will implement Sequence-to-Sequence models with Bahdanau additive attention for Neural Machine Translation (NMT), optimize decoding using Beam Search with length normalization, evaluate translations via BLEU scores, and synthesize audio spectator streams to construct a end-to-end Trigger Word Detection model.

---

## 📝 Core Technical Objectives
- **Encoder-Decoder Architecture & Beam Search Decoding:** Generating target output sequences $y^{\langle 1 \rangle}, \dots, y^{\langle T_y \rangle}$ conditioned on source input sequences $x^{\langle 1 \rangle}, \dots, x^{\langle T_x \rangle}$. Replacing greedy decoding with Beam Search ($\text{beam width } B \approx 3\text{--}10$) using length-normalized log probability targets to mitigate short-sequence penalty bias:

  $$\arg\max_{y} \frac{1}{T_y^\alpha} \sum_{t=1}^{T_y} \log P\left(y^{\langle t \rangle} \mid y^{\langle 1 \rangle}, \dots, y^{\langle t-1 \rangle}, x\right)$$

- **BLEU Score Evaluation Metric:** Quantifying precision between candidate translations and human reference sentences across $n$-grams using modified precision $p_n$ combined with a Brevity Penalty ($\text{BP}$):

  $$\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right), \quad \text{BP} = \begin{cases} 1 & \text{if } \text{length}_{\text{candidate}} > \text{length}_{\text{reference}} \\ \exp\left(1 - \frac{\text{length}_{\text{reference}}}{\text{length}_{\text{candidate}}}\right) & \text{if } \text{length}_{\text{candidate}} \le \text{length}_{\text{reference}} \end{cases}$$

- **Bahdanau Additive Attention Mechanics:** Computing dynamic attention weights $\alpha^{\langle t, t' \rangle}$ to focus decoder step $t$ on encoder hidden states $a^{\langle t' \rangle}$, generating context vectors $c^{\langle t \rangle} = \sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}$:

  $$e^{\langle t, t' \rangle} = \text{Dense}\left(\left[s^{\langle t-1 \rangle}, a^{\langle t' \rangle}\right]\right), \quad \alpha^{\langle t, t' \rangle} = \frac{\exp\left(e^{\langle t, t' \rangle}\right)}{\sum_{t''=1}^{T_x} \exp\left(e^{\langle t, t'' \rangle}\right)}$$

- **Audio Signal Processing & Spectrograms:** Transforming raw 1D acoustic audio waveforms into 2D time-frequency Spectrograms via Short-Time Fourier Transform (STFT), converting raw audio into spatial spectral features for 1D convolutional and recurrent networks.

- **Trigger Word Detection Pipeline:** Synthesizing training audio by overlaying positive trigger words ("Activate") and negative distractor words onto background noise streams, setting target label sequences $y^{\langle t \rangle} = 1$ for a fixed window directly following trigger occurrences, and training a Conv1D-GRU classification architecture.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, attention translation notebooks, and trigger word audio processing labs are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C5_W3.pdf)** | Formal visual reference covering seq2seq architectures, Beam Search tree exploration, BLEU score mathematical formulation, attention weight matrix alignment maps, and audio spectrogram transformations. |
| **[Neural Machine Translation with Attention](./05%20Sequence%20Models%20Sequence%20Models/03_Week/01%20Assingment/Neural_machine_translation_with_attention_v4a.ipynb)** | Building a custom date-formatting NMT engine in TensorFlow Keras: implementing custom Attention layers (`one_step_attention`), constructing shared pre-attention Bidirectional LSTMs and post-attention LSTMs, and visualizing attention alignments $\alpha^{\langle t, t' \rangle}$. |
| **[Trigger Word Detection Assignment](./05%20Sequence%20Models%20Sequence%20Models/03_Week/02%20Assingment/Trigger_word_detection_v1a.ipynb)** | End-to-end speech recognition project: synthesizing audio data by overlaying clipped audio clips onto background tracks, generating ground-truth label targets, computing spectrograms, and training a 1D Convolutional + GRU sequence classification network. |

---

## 💡 Visual Pipeline Reference

The Bahdanau dynamic attention alignment architecture alongside the Audio Spectrogram Trigger Word detection pipeline:

- **Attention Mechanism Context Computation**:

  $$\begin{aligned}   \left[ s^{\langle t-1 \rangle}, a^{\langle t' \rangle} \right] &\longrightarrow \text{Dense}(10, \text{activation}='{\text{tanh}}') \longrightarrow \text{Dense}(1, \text{activation}='{\text{relu}}') \longrightarrow e^{\langle t, t' \rangle} \\   \alpha^{\langle t, t' \rangle} &= \text{Softmax}\left(e^{\langle t, t' \rangle}\right) \quad \implies \quad c^{\langle t \rangle} = \sum_{t'=1}^{T_x} \alpha^{\langle t, t' \rangle} a^{\langle t' \rangle}   \end{aligned}$$

- **Trigger Word Audio Synthesizer Pipeline**:

  $$\begin{aligned}   \text{Background Noise} + \text{Clips ("Activate", "Distractors")} &\longrightarrow \text{1D Raw Audio Waveform } (137,581) \\   &\longrightarrow \text{STFT Spectrogram } (101, 5511) \longrightarrow \text{Conv1D}(f=15, s=4) \\   &\longrightarrow \text{BatchNormalization} \longrightarrow \text{GRU}(128) \longrightarrow \text{Dense}(1, \text{Sigmoid}) \longrightarrow \hat{Y} \in \mathbb{R}^{1375}   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Attention Theory & Sequence Search
- **Beam Search Error Diagnostics:** Conducting systemic error analysis to determine whether translation errors stem from the Beam Search algorithm ($P(y^* \mid x) > P(\hat{y} \mid x)$) or the underlying Sequence-to-Sequence RNN model ($P(y^* \mid x) \le P(\hat{y} \mid x)$).

- **Dynamic Attention Alignment:** Proving how attention layers resolve the bottleneck problem of traditional seq2seq models by enabling decoders to access all encoder time steps directly.

- **Target Label Synthesis for Audio:** Formulating consecutive label assignment strategies ($y^{\langle t \rangle} = 1$ for 50 timesteps after a trigger phrase) to handle class imbalance in temporal audio streams.


### 🤖 Applied TensorFlow & Audio Engineering
- **Custom Keras Attention Models:** Building complex functional models containing custom iteration loops across time steps $T_y$ using shared layer instances.

- **Audio Waveform Manipulation:** Utilizing PyDub to slice, overlay, and adjust decibel volume levels across raw `.wav` audio files.

- **1D Convolutional Feature Extraction:** Processing temporal spectrogram frames using `Conv1D(filters, kernel_size, strides)` to reduce time dimension sequence length before feeding into recurrent GRU blocks.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Signal & Audio Processing | Sequence Architectures | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![PyDub](https://img.shields.io/badge/PyDub-Audio_Processing-0056D2?style=flat&logo=python&logoColor=white) | ![Attention](https://img.shields.io/badge/Architecture-Bahdanau_Attention-D00000?style=flat&logo=keras&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
