# 🐟 Fish Freshness Classifier

[![Live Demo](https://img.shields.io/badge/🚀%20Try%20Now-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://fish-freshness-classifier.streamlit.app/)
[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/farizsalmant/fish-freshness-model/tree/main)
[![License: MIT](https://img.shields.io/badge/📜%20License-MIT-blue?style=for-the-badge)](https://github.com/farizsalman/Fish-Freshness-Classifier/blob/main/LICENCE)

---

## 📌 Overview

The **Fish Freshness Classifier** is an intelligent web-based tool that predicts whether a fish is **Fresh 🟢** or **Not Fresh 🔴** using deep learning. Built using TensorFlow and Streamlit, this project enables real-time predictions through a user-friendly interface where you can either upload or capture a fish image.

---

## 🔴🟢 Live in Action

- 🎯 **App**: [Click here to try it on Streamlit »](https://fish-freshness-classifier.streamlit.app/)
- 🧠 **Model**: [See the model on Hugging Face »](https://huggingface.co/datasets/farizsalmant/fish-freshness-model/tree/main)

---

## ⚙️ Technologies Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>

---

## 🧠 How It Works

1. 🖼 Upload or capture a fish image using the app
2. 🪄 The image is resized to **224x224** and normalized
3. 🤖 A deep learning model (hosted on Hugging Face) processes the image
4. ✅ Returns a freshness prediction with confidence score

---

## 📁 Project Structure
```
Fish-Freshness-Classifier/
├── app.py # Streamlit frontend
├── train_model.py # Model training script
├── split_data.py # Dataset handling (optional)
├── requirements.txt # Python dependencies
├── runtime.txt # Python version for deployment
├── LICENCE # MIT license
└── README.md # Project documentation

```
---

## 💾 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/farizsalman/Fish-Freshness-Classifier.git
cd Fish-Freshness-Classifier

# 2. (Optional) Create virtual environment
python -m venv venv
On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

---
📦 Model Info  
**Format:** `.h5` (Keras)  
**Hosted on Hugging Face:**  
[![Model on Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/farizsalmant/fish-freshness-model/tree/main)  
- Automatically downloaded on first run if not found locally

---

 📝 License  
[![License: MIT](https://img.shields.io/badge/📜%20License-MIT-blue?style=for-the-badge)](https://github.com/farizsalman/Fish-Freshness-Classifier/blob/main/LICENCE)  
This project is licensed under the MIT License.  
Feel free to use, modify, and distribute it with proper attribution.

---

👨‍💻 Author  
Salman Faris T  
[![GitHub](https://img.shields.io/badge/🐙%20GitHub-farizsalman-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/farizsalman)

