# Week 7 Milestone: Real-Time AI Image Classification Deployment

## 📋 Project Description
This project marks the final deployment stage of my Deep Learning journey with AnalystLab Africa (Batch A). Over the past few weeks, I trained a Convolutional Neural Network (CNN) model using computer vision techniques to classify objects. This week, the focus shifted from isolated notebooks to MLOps (Machine Learning Operations)—packaging, optimizing, and deploying the model so anyone can interact with it in the real world.

### The Problem Solved
Raw machine learning models trapped inside Jupyter Notebooks (`.ipynb` files) are useless to regular users. This application bridges that gap by providing a clean, accessible public web interface where anyone can upload a real-world image and instantly receive an AI-generated prediction.

---

## 🛠️ Technologies Used
* **Python** (Core programming language)
* **Streamlit** (Frontend interface and interactive web framework)
* **ONNX Runtime** (Open Neural Network Exchange - used for high-performance model deployment)
* **Pillow (PIL) & NumPy** (Image preprocessing and matrix transformations)
* **GitHub** (Version control and code repository hosting)

---

## 🏗️ How the AI System Works Under the Hood

1. **Model Serialization & Optimization:** Heavy deep learning frameworks can easily overwhelm and crash free-tier cloud servers due to memory spikes. To prevent this, the trained CNN weights were serialized and converted into an optimized **ONNX container format**. This keeps the model incredibly lightweight and lightning-fast.
2. **The Preprocessing Data Pipeline:** When a user uploads an image, the backend processing script instantly takes over using Pillow and NumPy. It automatically resizes the image to the exact dimensions the network expects, normalizes the pixel color channel arrays, and reshapes it into a structured matrix.
3. **The Live Prediction:** The processed data array is pushed through the live ONNX runtime inference engine, which instantly calculates and displays the final object classification prediction on the user's screen.

---

## 🚀 Live Links & Project Assets

* **Live Deployed Application:** [PASTE YOUR STREAMLIT APP LINK HERE]
* **Video Walkthrough Demo:** [PASTE YOUR LOOM VIDEO LINK HERE]
* **Professional Social Update:** [PASTE YOUR LINKEDIN POST LINK HERE]

---

## 💻 How to Run This Project Locally

If you want to run this application on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME
