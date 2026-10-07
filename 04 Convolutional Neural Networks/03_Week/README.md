# 🚘 Week 3: Object Detection, YOLO & U-Net Semantic Segmentation

Welcome to Week 3 of **Convolutional Neural Networks** (Course 4 of the **DeepLearning.AI Deep Learning Specialization**)! This module transitions from image-level classification to dense spatial prediction tasks: 2D object localization, bounding box regression, anchor box mechanics, non-max suppression (NMS), the single-shot YOLO algorithm, and pixel-wise semantic segmentation using transpose convolutions and U-Net architectures on autonomous vehicle datasets.

---

## 📝 Core Technical Objectives
- **Spatial Object Localization & Anchor Mechanics:** Encoding bounding boxes via target vectors $y = [p_c, b_x, b_y, b_h, b_w, c_1, c_2, c_3]^T$ normalized relative to grid cells ($0 \le b_x, b_y \le 1$), and employing multiple overlapping Anchor Boxes per cell to detect overlapping objects of varying aspect ratios.

- **Intersection over Union (IoU) & Non-Max Suppression (NMS):** Computing spatial overlap ratios $\text{IoU} = \frac{\text{Area of Intersection}}{\text{Area of Union}}$ to evaluate box precision, and filtering redundant candidate bounding boxes by discarding predictions with confidence below threshold $p_c < \tau_{\text{score}}$ followed by iterative NMS suppression ($\text{IoU} \ge \tau_{\text{overlap}}$).

- **Fully Convolutional Sliding Windows & YOLO Pipeline:** Replacing slow sliding-window cropping with fully convolutional 1x1 layers for parallel spatial inference, dividing images into $S \times S$ grid cells, and predicting tensor volumes $S \times S \times (B \cdot (5 + C))$ in a single forward pass.

- **Transpose Convolutions & Spatial Upsampling:** Reversing spatial downsampling using Transpose Convolutions (`Conv2DTranspose`), expanding feature resolution from $(H, W)$ to $(s \cdot H, s \cdot W)$ while learning upsampling filter kernels.

- **U-Net Architecture & Dense Semantic Segmentation:** Constructing symmetrical contracting (encoder) and expansive (decoder) paths, leveraging long skip connections to concatenate high-resolution low-level spatial features with upsampled semantic features, and training pixel-wise classifiers using Sparse Categorical Cross-Entropy loss.

---

## 🧪 Interactive Laboratory & Visual Selection Matrix

This module's core video lectures, lecture notes, autonomous driving YOLO projects, and CARLA U-Net segmentation notebooks are mapped directly to their operational focus:

| Asset / Deliverable | Operational Focus |
| :--- | :--- |
| **[Lecture Slides](./Lectures%20Files/C4_W3.pdf)** | Formal visual reference covering grid cell bounding box encodings, anchor box assignment rules, IoU geometry, Non-Max Suppression algorithms, transpose convolutions, and U-Net skip paths. |
| **[Car Detection with YOLO Assignment](./04%20Convolutional%20Neural%20Networks/03_Week/01%20Assingment/Autonomous_driving_application_Car_detection.ipynb)** | End-to-end implementation of the YOLO v2 detection post-processing pipeline: filtering candidate boxes by score (`yolo_filter_boxes`), computing IoU (`iou`), executing NMS (`yolo_non_max_suppression`), and evaluating predictions on drive-scene camera images. |
| **[Image Segmentation with U-Net Assignment](./04%20Convolutional%20Neural%20Networks/03_Week/02%20Assingment/Image_segmentation_Unet_v2.ipynb)** | Building a functional U-Net model from scratch in TensorFlow Keras: implementing encoder contracting blocks, expansive upsampling blocks (`Conv2DTranspose`), concatenating skip connections (`tf.keras.layers.concatenate`), and segmenting roads/objects on the CARLA self-driving dataset. |

---

## 💡 Visual Pipeline Reference

The spatial geometry of Intersection over Union alongside the U-Net encoder-decoder skip connection architecture:

- **Intersection over Union (IoU) Formulation**:

  $$\text{IoU}(B_1, B_2) = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1) + \text{Area}(B_2) - \text{Area}(B_1 \cap B_2)}$$

- **U-Net Expansive Skip Connection Pipeline**:

  $$\begin{aligned}   \mathbf{X}_{\text{encoder}}^{[l]} \longrightarrow &\text{Conv2D} \longrightarrow \text{MaxPool2D} \longrightarrow \dots \text{Bottleneck} \\   \longrightarrow &\text{Conv2DTranspose}\left(\mathbf{X}_{\text{decoder}}^{[l+1]}\right) \longrightarrow \mathbf{X}_{\text{upsampled}}^{[l]} \\   \longrightarrow &\text{Concatenate}\left(\left[\mathbf{X}_{\text{upsampled}}^{[l]}, \mathbf{X}_{\text{skip}}^{[l]}\right]\right) \longrightarrow \text{Conv2D} \longrightarrow \hat{\mathbf{Y}}_{\text{mask}}   \end{aligned}$$

---

## 🎯 Technical Skills Architecture

### 📊 Object Detection & Dense Spatial Theory
- **Single-Shot Grid Detection:** Understanding spatial grid assignment mechanics where an object's center point uniquely determines its responsible grid cell and anchor box.

- **NMS Box Suppression:** Applying class-wise non-max suppression loops to eliminate duplicate bounding box detections while retaining distinct adjacent objects.

- **Semantic Mask Resolution:** Proving why low-level spatial skip connections in U-Net restore high-frequency boundary information lost during max-pooling downsampling.


### 🤖 Applied TensorFlow Engineering
- **Tensor Masking & Selection:** Leveraging `tf.boolean_mask` and `tf.image.non_max_suppression` to dynamically filter multi-dimensional anchor prediction tensors on the GPU.

- **Expansive Decoding Layers:** Engineering symmetrical decoder paths using `tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')`.

- **Pixel-Wise Segmentation Optimization:** Compiling dense prediction networks with `tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)` to process integer segmentation targets across self-driving scenes.


---

## 🛠️ Production Tech Stack & Ecosystem

| Deep Learning Framework | Computer Vision Engine | Interactive Environment |
| :---: | :---: | :---: |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x_Keras-FF6F00?style=flat&logo=tensorflow&logoColor=white) | ![YOLO & U-Net](https://img.shields.io/badge/Vision-YOLO_v2_&_UNet-0056D2?style=flat&logo=opencv&logoColor=white) | ![Jupyter](https://img.shields.io/badge/Jupyter-Interactive_Labs-FA0F00?style=flat&logo=jupyter&logoColor=white) |
