# 🏅 Custom CNN for 100 Sports Image Classification

<p align="center">
  <b>Computer Vision and Pattern Recognition Course Assignment</b><br>
  Custom Convolutional Neural Network for Multiclass Sports Image Classification
</p>

---

## 📌 Project Information

| Item | Details |
|---|---|
| **Course** | Computer Vision and Pattern Recognition |
| **Assignment Title** | CNN Development on a Custom Dataset |
| **Student Name** | `Niloy Paul` |
| **Student ID** | `23-50400-1` |
| **Dataset** | 100 Sports Image Classification |
| **Dataset Source** | Kaggle |
| **Framework** | PyTorch |
| **Model Type** | Custom Convolutional Neural Network |
| **Number of Classes** | 100 |
| **Input Image Size** | 224 × 224 × 3 |

---

## 📖 Project Overview

This project develops a **custom Convolutional Neural Network (CNN)** for classifying images into **100 different sports categories**.

The complete computer vision pipeline includes:

- downloading the dataset from Kaggle;
- inspecting and validating the dataset;
- applying image preprocessing and augmentation;
- loading separate training, validation, and test sets;
- designing a custom CNN architecture from scratch;
- training the model using cross-entropy loss;
- applying optimization and regularization techniques;
- selecting and saving the best-performing model;
- evaluating the model on an independent test set;
- reporting overall and per-class performance;
- generating training curves and confusion matrices;
- analyzing the best, worst, and most frequently confused classes.

The goal is not only to obtain high accuracy, but also to demonstrate a complete, organized, reproducible, and academically valid image-classification workflow.

---

## 🎯 Assignment Objectives

The main objectives are:

1. Build a custom CNN without using a pretrained architecture as the primary model.
2. Correctly load and preprocess a multiclass image dataset.
3. Create training, validation, and test DataLoaders.
4. Use an appropriate optimizer and multiclass loss function.
5. Apply regularization to reduce overfitting.
6. Monitor training and validation performance.
7. Save the best model weights in `.pth` format.
8. Evaluate the final model using comprehensive metrics.
9. Visualize the results using graphs and a confusion matrix.
10. Discuss model strengths, weaknesses, limitations, and future improvements.

---

## 🗂️ Dataset

### Dataset Name

**100 Sports Image Classification**

### Kaggle Link

```text
https://www.kaggle.com/datasets/gpiosenka/sports-classification
```

### Kaggle Dataset Handle

```text
gpiosenka/sports-classification
```

### Dataset Summary

| Split | Number of Images |
|---|---:|
| Training | 13,493 |
| Validation | 500 |
| Test | 500 |
| **Total** | **14,493** |

### Image Properties

- **Image size:** 224 × 224 pixels
- **Channels:** 3-channel RGB
- **Format:** JPG
- **Number of classes:** 100
- **Task:** Single-label multiclass image classification

> **Important limitation:** The validation and test sets contain approximately five images per class. Therefore, per-class metrics may change noticeably because of only one or two incorrect predictions.

---

## 🧠 Custom CNN Architecture

The project uses a custom CNN implemented from scratch in PyTorch.

### Architecture Flow

```text
Input Image: 3 × 224 × 224
        │
        ▼
Convolution Block 1: 3 → 32 channels
        │
        ▼
Convolution Block 2: 32 → 64 channels
        │
        ▼
Convolution Block 3: 64 → 128 channels
        │
        ▼
Convolution Block 4: 128 → 256 channels
        │
        ▼
Adaptive Average Pooling
        │
        ▼
Fully Connected Layer
        │
        ▼
Output Layer: 100 classes
```

### Components Used in Each Convolution Block

Each block contains:

```text
Conv2D
→ Batch Normalization
→ ReLU
→ Conv2D
→ Batch Normalization
→ ReLU
→ Max Pooling
→ Dropout2D
```

### Purpose of the Main Components

| Component | Purpose |
|---|---|
| **Convolution** | Learns edges, textures, shapes, and class-specific visual features |
| **Batch Normalization** | Stabilizes and accelerates training |
| **ReLU** | Adds nonlinearity |
| **Max Pooling** | Reduces spatial dimensions and computation |
| **Dropout** | Reduces overfitting |
| **Adaptive Average Pooling** | Produces a fixed-length feature vector |
| **Fully Connected Layer** | Performs final classification |
| **Output Layer** | Produces one score for each of the 100 classes |

No Softmax layer is placed inside the model because `CrossEntropyLoss` expects raw logits.

---

## ⚙️ Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Maximum Epochs | 25 |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss |
| Early-Stopping Patience | 5 epochs |
| Random Seed | 42 |
| Model-Selection Criterion | Lowest validation loss |

These values can be adjusted depending on hardware limitations and validation behavior.

---

## 🧪 Data Preprocessing and Augmentation

### Training Transformations

The training set uses augmentation to improve generalization:

- random resized crop;
- random horizontal flip;
- small rotation;
- brightness adjustment;
- contrast adjustment;
- saturation adjustment;
- conversion to tensor;
- RGB normalization.

### Validation and Test Transformations

Validation and test images use deterministic preprocessing:

- resize to 224 × 224;
- convert to tensor;
- normalize using RGB mean and standard deviation.

Random augmentation is not used on validation or test images because evaluation must remain consistent.

---

## 🏋️ Training Strategy

The model is trained using the following procedure:

1. Set the model to training mode.
2. Load a batch of augmented training images.
3. Perform a forward pass.
4. Calculate multiclass cross-entropy loss.
5. Backpropagate the gradients.
6. Update model parameters using AdamW.
7. Evaluate the model on the validation set.
8. Reduce the learning rate when validation loss stops improving.
9. Save the best model checkpoint.
10. Stop training early if validation loss does not improve for five consecutive epochs.

### Regularization Techniques

The following regularization methods are used:

- data augmentation;
- batch normalization;
- dropout;
- spatial dropout;
- weight decay;
- learning-rate scheduling;
- early stopping;
- best-checkpoint saving.

---

The complete checkpoint may contain:

- model weights;
- optimizer state;
- best epoch;
- validation loss;
- validation accuracy;
- class names;
- class-to-index mapping;
- image size;
- experiment hyperparameters.

### Model Loading Example

```python
model = CustomSportsCNN(num_classes=100)

state_dict = torch.load(
    "best_custom_cnn_weights.pth",
    map_location=device
)

model.load_state_dict(state_dict)
model.to(device)
model.eval()
```

> The assignment specifies that a model file larger than 20 MB should not be uploaded to GitHub.

---

## 📊 Evaluation Metrics

The final model is evaluated on the independent test set using:

- accuracy;
- macro precision;
- macro recall;
- macro F1-score;
- weighted precision;
- weighted recall;
- weighted F1-score;
- per-class precision;
- per-class recall;
- per-class F1-score;
- support;
- raw confusion matrix;
- normalized confusion matrix.

### Metric Interpretation

| Metric | Meaning |
|---|---|
| **Accuracy** | Percentage of all test images classified correctly |
| **Precision** | Of all images predicted as a class, how many were correct |
| **Recall** | Of all actual images in a class, how many were detected |
| **F1-score** | Harmonic balance between precision and recall |
| **Macro Average** | Treats every class equally |
| **Weighted Average** | Accounts for class frequency |
| **Support** | Number of actual test samples for a class |

---

## 📈 Visualizations

The notebook generates:

1. training class-distribution graph;
2. sample training images;
3. training and validation loss curves;
4. training and validation accuracy curves;
5. raw confusion matrix;
6. normalized confusion matrix;
7. misclassified-image visualization.

These visualizations support interpretation of model behavior and common errors.

---

## 🔎 Performance Analysis

The notebook performs:

- best-performing class analysis;
- worst-performing class analysis;
- most frequent confusion-pair analysis;
- misclassified-image inspection;
- prediction-confidence inspection;
- training-versus-validation behavior analysis.

### Important Interpretation Note

Training accuracy may be lower than validation accuracy because:

- training images are randomly augmented;
- dropout is active during training;
- batch normalization uses mini-batch statistics;
- validation images are cleaner and deterministic;
- the validation set is relatively small.

Therefore, online training accuracy and validation accuracy are not always directly comparable.

---

## 📚 Technologies Used

- Python
- PyTorch
- Torchvision
- TorchInfo
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- KaggleHub
- Google Colab
- GitHub

---

<p align="center">
  <b>Developed as a CVPR course assignment using PyTorch and a custom CNN.</b>
</p>
