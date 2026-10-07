# 🔤 Week 2: Natural Language Processing & Word Embeddings

Welcome to Week 2 of **Sequence Models** (Course 5 of the **DeepLearning.AI Deep Learning Specialization**)! This module explores dense vector representations of language—transitioning from sparse one-hot encodings to continuous $d$-dimensional word embedding spaces (Word2Vec, Skip-Gram, Negative Sampling, GloVe). You will implement vector cosine similarity, solve structural word analogies, quantify and neutralize gender/societal bias in pre-trained embeddings, and construct deep LSTM sentiment analysis models (Emojifier) using custom Keras `Embedding` layers.

---

## 📝 Core Technical Objectives
- **Dense Vector Spaces & Cosine Similarity:** Mapping discrete vocabulary words into continuous feature spaces $e_w \in \mathbb{R}^{d}$ ($d \approx 50\text{--}300$), measuring directional alignment and semantic proximity via Cosine Similarity:

  $$\text{Similarity}(u, v) = \frac{u \cdot v}{\Vert{}u\Vert{}_2 \Vert{}v\Vert{}_2} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^{d} u_i^2} \sqrt{\sum_{i=1}^{d} v_i^2}}$$

- **Word Analogy Formulations:** Solving proportional semantic relationships ($e_{\text{man}} : e_{\text{woman}} :: e_{\text{king}} : e_y$) by identifying candidate vectors $e_y$ that maximize directional similarity:

  $$\arg\max_{y} \text{Similarity}\left(e_y, \; e_{\text{woman}} - e_{\text{man}} + e_{\text{king}}\right)$$

- **Efficient Embedding Learning Algorithms:**
  - **Skip-Gram with Negative Sampling:** Converting expensive $\vert{}V\vert{}$-way softmax updates into $K+1$ binary logistic regression classification tasks ($1$ target context word vs. $K$ sampled noise words).
  - **GloVe (Global Vectors for Word Representation):** Minimizing log-count matrix factorization loss $\mathcal{L} = \sum_{i,j=1}^{\vert{}V\vert{}} f(X_{ij}) \left(\theta_i^T e_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$.

- **Embedding Debiasing Protocols:** Neutralizing unwanted gender/societal bias along specific feature axes by isolating bias directions $g = e_{\text{he}} - e_{\text{she}}$, projecting non-gendered target vectors ($e_{\text{doctor}}$, $e_{\text{programmer}}$) onto orthogonal hyperplanes ($e^{\perp}$), and equalizing equidistant word pairs ($e_{\text{girl}}$, $e_{\text{boy}}$).

- **Keras Embedding Integration & Emojifier Models:** Implementing custom embedding lookups (`Embedding(input_dim=|V|, output_dim=d)`) with fixed pre-trained GloVe weights (`trainable=False`), feeding variable-length token sequences into multi-layer Dropout-regularized Bidirectional LSTMs for fine-grained sentiment classification.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, vector debiasing notebooks, and Keras Emojify sentiment classifiers are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C5_W2.pdf)** | Formal visual and mathematical reference covering word feature spaces, Skip-Gram architecture, Negative Sampling loss formulations, GloVe co-occurrence matrix optimization, and geometric vector debiasing algorithms. |
| **[Operations on Word Vectors](./05%20Sequence%20Models%20Sequence%20Models/02_Week/01%20Assingment/Operations_on_word_vectors_v2a.ipynb)** | Implementing core embedding mechanics in NumPy: computing vector similarity (`cosine_similarity`), solving analogies (`complete_analogy`), and neutralizing bias via projection (`neutralize`) and equalization (`equalize`). |
| **[Emojify Assignment](./05%20Sequence%20Models%20Sequence%20Models/02_Week/02%20Assingment/Emoji_v3a.ipynb)** | Building two sentiment classification architectures: a baseline average word-vector model (`Emojify-V1`) and a deep 2-layer LSTM network (`Emojify-V2`) utilizing pre-trained 50-dimensional GloVe embeddings in TensorFlow Keras. |

---

## 💡 Visual Pipeline Reference

The vector debiasing projection geometry alongside the deep LSTM sentiment network architecture:

- **Vector Neutralization Projection Formula**:

  $$e_{\text{bias\_component}} = \frac{e \cdot g}{\Vert{}g\Vert{}_2^2} \cdot g \quad \implies \quad e^{\perp} = e - e_{\text{bias\_component}}$$

- **Deep LSTM Emojifier Neural Pipeline**:

  $$\begin{aligned}   \text{Text Sequence} &\longrightarrow \text{Tokens } [x_1, x_2, \dots, x_{T_x}] \longrightarrow \text{Embedding Layer (Pre-trained GloVe)} \\   &\longrightarrow \text{LSTM}(128, \text{return\_sequences}=\text{True}) \longrightarrow \text{Dropout}(0.5) \\   &\longrightarrow \text{LSTM}(128, \text{return\_sequences}=\text{False}) \longrightarrow \text{Dropout}(0.5) \\   &\longrightarrow \text{Dense}(5) \longrightarrow \text{Softmax} \longrightarrow \hat{y} \text{ (Emoji Class)}   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Embedding Theory & Vector Geometry
- **Semantic Space Mechanics:** Explaining how high-dimensional embedding dimensions implicitly encode latent semantic features (e.g., gender, age, royalty, tense).

- **Negative Sampling Efficiency:** Proving how replacing full vocabulary softmax denominators with binary logistic losses reduces computational complexity per step from $\mathcal{O}(\vert{}V\vert{})$ to $\mathcal{O}(K)$.

- **Bias Neutralization Math:** Deriving orthogonal vector projections to eliminate systematic demographic bias in downstream NLP systems without destroying core semantic meanings.


### 🤖 Applied TensorFlow & NLP Engineering
- **Pre-trained Matrix Mapping:** Parsing GloVe `.txt` files into dictionary lookup tables and populating Keras `Embedding` weight matrices ($\vert{}V\vert{} \times d$).

- **Tensor Index Mapping:** Converting raw string sentence inputs into padded integer index arrays (`sentences_to_indices`) matching pre-trained vocabulary dictionaries.

- **Sequence Classifier Architecture:** Constructing sequence-to-vector classification models in Keras by stacking recurrent layers with `return_sequences` parameter management.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Vector Math & Mechanics | Pre-trained Embeddings | Interactive Environment |
| :---: | :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![NumPy](https://img.shields.io/badge/NumPy-Array_Operations-013243?style=flat&logo=numpy&logoColor=white) | ![GloVe](https://img.shields.io/badge/GloVe-Word_Embeddings-0056D2?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
