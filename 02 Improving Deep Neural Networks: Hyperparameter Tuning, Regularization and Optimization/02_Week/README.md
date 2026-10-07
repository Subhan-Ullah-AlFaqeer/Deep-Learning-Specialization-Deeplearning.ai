
# 🚀 Week 2: Optimization Algorithms

Welcome to Week 2 of **Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization** (Course 2 of the **DeepLearning.AI Deep Learning Specialization**)! This module equips your neural network toolbox with advanced optimization algorithms and acceleration techniques: partitioning datasets into Mini-Batches, computing Exponentially Weighted Averages with Bias Correction, accelerating parameter updates using Gradient Descent with Momentum, RMSprop, and Adam, and implementing dynamic Learning Rate Decay schedules to navigate complex cost landscapes with local optima and plateaus.

---

## 📝 Core Technical Objectives
* **Mini-Batch Gradient Descent Pipeline:** Partitioning full training set $(X, Y)$ into smaller mini-batches $(X^{\{t\}}, Y^{\{t\}})$ of size $b$ (e.g., $64, 128, 256, 512$) to balance CPU/GPU vector parallelism with faster, iterative parameter update frequency per epoch.
* **Exponentially Weighted Moving Averages:** Tracking continuous state dynamics $v_t = \beta v_{t-1} + (1 - \beta) \theta_t$ with smoothing hyperparameter $\beta$, and applying initialization Bias Correction $\frac{v_t}{1 - \beta^t}$ to prevent underestimation during early iterations ($t \to 1$).
* **Advanced First-Order Optimization Algorithms:**
  * **Gradient Descent with Momentum:** Tracking velocity of past gradients $v_{dW} = \beta v_{dW} + (1 - \beta) dW$ to damp vertical oscillations and accelerate horizontal trajectory toward the minimum.
  * **RMSprop (Root Mean Square Propagation):** Scaling gradients by root of exponentially weighted squared derivatives $s_{dW} = \beta_2 s_{dW} + (1 - \beta_2) dW^2$, dampening updates along high-variance dimensions via $\frac{dW}{\sqrt{s_{dW}} + \epsilon}$.
  * **Adam (Adaptive Moment Estimation):** Combining Momentum (1st moment $v$) and RMSprop (2nd moment $s$) with explicit bias correction ($\hat{v}_t, \hat{s}_t$) for state-of-the-art adaptive learning rates per parameter: $W \leftarrow W - \alpha \frac{\hat{v}_{dW}}{\sqrt{\hat{s}_{dW}} + \epsilon}$.
* **Learning Rate Decay Scheduling:** Decay learning rate $\alpha$ over epochs $e$ ($\alpha = \frac{\alpha_0}{1 + \text{decay\rate} \cdot e}$, exponential decay $\alpha = \alpha_0 \cdot \gamma^e$, or step-wise decay) to allow aggressive early exploration and fine-grained convergence near the global minimum.
* **Cost Topography Navigation:** Navigating local optima, saddle points, and zero-gradient plateaus in high-dimensional non-convex parameter spaces.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture slides, interactive Jupyter assignments, and industry perspectives are mapped directly to their targeted analytical focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C2_W2.pdf)** | Formal visual and mathematical reference covering mini-batch iteration loops, exponentially weighted smoothing derivations, bias correction curves, and combined Adam updates. |
| **[Optimization Methods Assignment](./02_Week/01%20Assingment/Optimization_methods.ipynb)** | Programming lab implementing mini-batch partitioning, Batch GD, Momentum, RMSprop, Adam, and Learning Rate Decay schedules from scratch in pure NumPy, comparing convergence speed on 2D moon-shaped classification boundaries. |
| **[Yuanqing Lin Interview](./Yuanqing%20Lin%20Interview)** | Strategic discussion on industrial computer vision applications, large-scale autonomous driving systems, platform engineering, and high-performance AI deployment. |

---

## 💡 Visual Pipeline Reference

The mathematical formulations and state updates across the advanced optimization algorithms implemented in this module:

* **Random Mini-Batch Shuffling & Partitioning**:
  $$\text{Shuffle } (X, Y) \text{ synchronously } \longrightarrow \text{Slice into } \left(X^{\{1\}}, Y^{\{1\}}\right), \left(X^{\{2\}}, Y^{\{2\}}\right), \dots, \left(X^{\{T\}}, Y^{\{T\}}\right)$$
* **Adam Optimization Update Step (Iterating over Mini-Batch $t$)**:
  $$v_{dW} = \beta_1 v_{dW} + (1 - \beta_1) dW, \quad v_{db} = \beta_1 v_{db} + (1 - \beta_1) db \quad \text{(1st Moment Vector)}$$
  $$s_{dW} = \beta_2 s_{dW} + (1 - \beta_2) dW^2, \quad s_{db} = \beta_2 s_{db} + (1 - \beta_2) db^2 \quad \text{(2nd Moment Vector)}$$
  $$v_{dW}^{\text{corrected}} = \frac{v_{dW}}{1 - \beta_1^t}, \quad v_{db}^{\text{corrected}} = \frac{v_{db}}{1 - \beta_1^t} \quad \text{(Bias Correction)}$$
  $$s_{dW}^{\text{corrected}} = \frac{s_{dW}}{1 - \beta_2^t}, \quad s_{db}^{\text{corrected}} = \frac{s_{db}}{1 - \beta_2^t} \quad \text{(Bias Correction)}$$
  $$W \leftarrow W - \alpha \frac{v_{dW}^{\text{corrected}}}{\sqrt{s_{dW}^{\text{corrected}}} + \epsilon}, \quad b \leftarrow b - \alpha \frac{v_{db}^{\text{corrected}}}{\sqrt{s_{db}^{\text{corrected}}} + \epsilon}$$

---

## 🎯 Technical Skills Architecture

### 📊 Deep Learning Optimization & Mathematical Foundations
* **Moment Approximations:** Deriving why $v_t \approx \frac{1}{1-\beta}$ recent timesteps of data are integrated into the exponentially weighted average.
* **Hyperparameter Selection Defaults:** Establishing industrial baseline defaults: Momentum $\beta_1 = 0.9$, RMSprop/Adam $\beta_2 = 0.999$, stabilization $\epsilon = 10^{-8}$, and tuning primary learning rate $\alpha$.
* **Saddle Point Dynamics:** Understanding why high-dimensional space optimization bottlenecks stem from zero-gradient plateaus and saddle points rather than isolated local minima.

### 🤖 Applied Optimization Engineering
* **Mini-Batch Memory Optimization:** Selecting mini-batch sizes that fit power-of-two GPU/CPU cache boundaries ($64, 128, 256, 512$) to maximize SIMD execution throughput.
* **In-Place Matrix State Updates:** Vectorizing 1st ($v$) and 2nd ($s$) moment dictionary matrices across arbitrary $L$-layer network shapes without memory duplication.
* **Dynamic Schedule Integration:** Implementing learning rate decay within training epoch loops to achieve sub-decimal loss precision near local minima.

---

## 🛠️ Production Tech Stack & Ecosystem

| Numerical Optimization | Vector Visualization | Interactive Environment |
| :---: | :---: | :---: |
| ![NumPy](https://img.shields.io/badge/NumPy-Adaptive_Optimizers-013243?style=flat&logo=numpy&logoColor=white) | ![Matplotlib](https://img.shields.io/badge/Matplotlib-Loss_Convergence-11557c?style=flat&logo=python&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |

