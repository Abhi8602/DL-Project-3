# Deep Learning Project 3: Adversarial Robustness Evaluation

## Authors
- Abhishek Katke (ak11553)
- Harshini Vijaya Kumar (hv2201)

## Repository
🔗 [GitHub Repository](https://github.com/Abhi8602/DL-Project-3)

---

##  Abstract

This project investigates the **adversarial robustness** of deep convolutional neural networks by evaluating how susceptible they are to small, targeted perturbations. We explore three prominent white-box adversarial attack strategies—**Fast Gradient Sign Method (FGSM)**, **Projected Gradient Descent (PGD)**, and **Adversarial Patch Attacks**—on a **pre-trained ResNet-34** model using a subset of the ImageNet dataset. 

We also test the **transferability** of adversarial examples on a second model, **DenseNet-121**, and measure performance using Top-1 and Top-5 accuracy, PSNR, and L∞ norms. The results highlight critical vulnerabilities and reinforce the need for robust defenses like adversarial training.

---

##  Introduction

Deep Neural Networks (DNNs) achieve high performance in tasks like image classification but are vulnerable to adversarial examples—subtle, often imperceptible changes that can lead to incorrect predictions. This poses significant risks in real-world applications like autonomous driving or medical imaging. 

Our objective is to:
- Evaluate robustness of popular models.
- Measure degradation in classification accuracy under adversarial attacks.
- Examine cross-model **transferability** of adversarial examples.

---

##  Methodology

###  Model Architectures
- **ResNet-34**: Main model under attack.
- **DenseNet-121**: Used for transferability testing.

###  Dataset
- Subset of the **ImageNet validation dataset**.
- Standard preprocessing (resizing + normalization with ImageNet stats).

###  Attack Strategies
- **FGSM**: One-step gradient sign method with ε = 0.02.
- **PGD**: Iterative gradient attack with ε = 0.02, α = 0.005, and 10 iterations.
- **Patch Attack**: Localized noise in 32x32 patch, ε = 0.3, 20 steps.

###  Evaluation Metrics
- **Top-1 & Top-5 Accuracy**
- **Attack Success Rate**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **L∞ Norm Distance**

---

##  Implementation

- Framework: **PyTorch**
- Executed on: **CUDA-enabled GPU**
- Clean and adversarial examples generated dynamically.
- Used `torch.topk()` for classification accuracy.
- Visualized perturbations and computed PSNR differences.

---

##  Results Summary

| **Model**      | **Top-1 (Clean)** | **Top-5 (Clean)** |
|----------------|-------------------|--------------------|
| ResNet-34      | 78.01%            | 93.62%             |
| DenseNet-121   | 76.54%            | 92.87%             |

| **Attack Type** | **Top-1** | **Top-5** | **Accuracy Drop** | **PSNR (dB)** | **Success Rate** |
|-----------------|-----------|-----------|--------------------|----------------|------------------|
| FGSM            | 26.4%     | 50.6%     | 49.6%              | 28.4           | 71.5%            |
| PGD             | 0.4%      | 6.6%      | 75.6%              | 24.8           | 85.3%            |
| Patch           | 16.0%     | 39.0%     | 60.0%              | 26.7           | 78.6%            |

---

##  Transferability Results (on DenseNet-121)

| **Attack Type** | **Top-1 Accuracy** | **Top-5 Accuracy** | **Accuracy Drop** |
|-----------------|--------------------|---------------------|--------------------|
| FGSM            | 42.4%              | 66.4%               | 32.4%              |
| PGD             | 39.0%              | 64.4%               | 35.8%              |
| Patch           | 42.8%              | 67.0%               | 32.0%              |

---

##  Key Takeaways

- **PGD is the most powerful** attack, significantly reducing accuracy.
- **Adversarial examples transfer** between architectures, indicating shared vulnerabilities.
- **Patch attacks** are physically realizable and still degrade performance effectively.
- Despite **high PSNR (low visual distortion)**, model predictions fail—highlighting the brittleness of current models.

---

##  References

- I. J. Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015.  
- A. Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks*, ICLR 2018.  
- **ChatGPT** (OpenAI), **DeepSeek**, **Claude**, and peer discussions were used to help brainstorm, verify math formulations, and refine explanations for methodology and results documentation.

---

##  Future Work

- Incorporate **black-box and physical-world attacks**.
- Explore **defensive methods** (e.g., adversarial training, defensive distillation).
- Extend evaluation to **object detection and segmentation** models.

---

## 📁 Folder Structure

```bash
DL-Project-3/
├── attacks/
│   ├── fgsm.py
│   ├── pgd.py
│   └── patch.py
├── models/
│   ├── resnet34.py
│   └── densenet121.py
├── data/
├── utils/
│   ├── visualize.py
│   └── metrics.py
├── notebooks/
├── results/
└── README.md
