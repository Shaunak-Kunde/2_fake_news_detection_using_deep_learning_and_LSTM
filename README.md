# Fake News Detection Using Deep Learning

This project implements a deep learning approach to detect fake news articles using natural language processing techniques. The model is trained on labeled news data to classify whether a given news headline or article is *real* or *fake*.

## 📌 Project Overview

The goal of this project is to build a binary classification model using deep learning (LSTM-based neural network) to detect fake news. The project includes the following steps:
- Data loading and preprocessing
- Text tokenization and padding
- Model building using Keras (TensorFlow backend)
- Training and evaluation of the model
- Performance visualization and accuracy metrics

## 🧠 Model Architecture

The deep learning model used is a Sequential LSTM model that includes:
- Embedding Layer
- LSTM Layer
- Dense Layer with ReLU activation
- Output Layer with Sigmoid activation

## 🗂️ Dataset

The dataset used includes labeled news articles with binary labels:
- **1** for real news
- **0** for fake news

Ensure you have the dataset file (`news.csv` or similar) available in your working directory or modify the code to load the correct dataset path.

## ⚙️ Requirements

The notebook uses the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow` / `keras`
- `sklearn`

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow