# Satellite Image Resolution Enhancement

## Comparative Assessment of Resolution Enhancement Models for Satellite Images

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://m24cse002-m24cse032-m24cse030-m24csa023.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QwP7Iwlwbxg8fQDnciDQ_w19lr9tbaVI?usp=sharing)

This project compares various resolution enhancement techniques for satellite imagery using the UC Merced Land Use dataset. It includes traditional interpolation methods, deep learning-based super-resolution models, and novel hybrid approaches.

---

## 🚀 Features

- Multiple enhancement techniques:
  - Traditional interpolation: Bicubic, Lanczos
  - Deep learning: CNN-based Super-Resolution
  - Wavelet-based sharpening
  - Wiener filtering
  - Hybrid methods: Wavelet-SR, Wiener-SR
- Quantitative evaluation metrics: **PSNR**, **SSIM**, **MSE**
- Interactive web app for real-time comparisons
---

## 📦 Dataset

We use the **UC Merced Land Use** dataset with 10 selected categories and 100 images per class:

- Agricultural
- Baseball diamond
- Beach
- Buildings
- Forest
- Airplane
- Freeway
- Golf course
- Harbor
- Mobile home park

---

## 🔧 Methodology

### 🔄 Data Preprocessing Pipeline

1. Convert color space from BGR to RGB
2. Generate LR-HR image pairs:
   - HR: 256×256
   - LR: 128×128 (2× downsampling + Gaussian blur)


### 🧠 Enhancement Approaches

#### Traditional Methods
- **Bicubic**: Cubic spline-based interpolation
- **Lanczos**: High-quality sinc function resampling

#### Deep Learning (Super-Resolution)
- CNN-based architecture:
  - LeakyReLU activations
  - Residual connections
  - Scale-specific upsampling
  - Sigmoid output layer

#### Traditional Enhancement Techniques
- **Wavelet Sharpening**: Haar wavelets with detail boosting
- **Wiener Filtering**: Denoising + deblurring filter

#### Hybrid Techniques
- **Wavelet-SR**: Wavelet enhancement on SR output
- **Wiener-SR**: SR combined with Wiener filtering

---

## 📊 Results

| Method        | SSIM   | PSNR (dB) | MSE       |
|---------------|--------|-----------|-----------|
| Bicubic       | 0.6756 | 26.49     | 0.069530  |
| Lanczos       | 0.6963 | 27.02     | 0.064853  |
| Super-Resolution | **0.8456** | 27.07     | **0.002539** |
| Wavelet       | 0.8126 | **28.49** | 0.002618  |
| Wiener        | 0.6949 | 26.43     | 0.004028  |
| Wavelet-SR    | 0.8221 | 26.39     | 0.002962  |
| Wiener-SR     | 0.7447 | 26.00     | 0.003429  |

- ✅ **Super-Resolution** had highest SSIM & lowest MSE
- ✅ **Wavelet** recorded highest PSNR
- 🔁 Hybrid methods showed notable improvements

---

## 🌐 Demo Application

Explore the enhancement techniques live:

👉 [Streamlit App](https://m24cse002-m24cse032-m24cse030-m24csa023.streamlit.app/)

---

## 🛠️ Implementation

Built with:

- **Python**
- **TensorFlow**, **OpenCV**, **scikit-image**, **PyWavelets**
- **Streamlit** (Web App)
- **Google Colab** (Development & Training)

---

## 👥 Team Members

| Name                          | ID         | Contributions |
|-------------------------------|------------|---------------|
| **Akaash Chatterjee**         | M24CSE002  | Team Lead, CNN SR, Wavelet-SR, Integration |
| **Alok Dutta**                | M24CSE032  | Interpolation methods, Dataset Prep |
| **Prathmesh Chintamani Gosavi** | M24CSA023  | Wavelet & Wiener-SR, Preprocessing |
| **Vishwanath Singh**          | M24CSE030  | Wiener Filter, Visualization & Metrics |

---

## 🔗 Links

- 🚀 [Streamlit Deployment](https://m24cse002-m24cse032-m24cse030-m24csa023.streamlit.app/)
- 📓 [Colab Notebook](https://colab.research.google.com/drive/1QwP7Iwlwbxg8fQDnciDQ_w19lr9tbaVI#scrollTo=UY_QBQOrIYq1)

---

## 📜 License

This project was developed as part of the **CSL7360 - Computer Vision** course at **IIT Jodhpur**.

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{chatterjee2025satellite,
  author = {Chatterjee, Akaash and Dutta, Alok and Singh, Vishwanath and Gosavi, Prathmesh Chintamani},
  title = {Comparative Assessment of Resolution Enhancement Models for Satellite Images},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/akaash897/Computer_Vision}}
}
