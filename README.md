# 🧠 Deep Emotion

**Deep Emotion** is a deep learning-based emotion classification system that analyzes textual input and predicts human emotions such as **joy**, **sadness**, **anger**, **fear**, **love**, and **surprise**. It leverages a **Bidirectional LSTM** with an **Attention Mechanism** and uses **GloVe embeddings** for semantic understanding. The model is deployed through a **Streamlit** web application, allowing real-time emotion prediction from user input.

---

## 📌 Key Features

- Emotion classification from text using deep learning.
- BiLSTM with Attention mechanism for enhanced contextual understanding.
- Pretrained GloVe embeddings for rich semantic word representation.
- Text preprocessing using SpaCy (tokenization & lemmatization).
- Streamlit-based interactive web application.
- Evaluation using accuracy, F1-score, and confusion matrices.

---

## 🛠 Technologies Used

- Python
- PyTorch
- SpaCy
- Hugging Face Datasets
- Scikit-learn
- GloVe Embeddings
- Streamlit
- Matplotlib & Seaborn

---

## 📂 Project Structure


```bash
DeepEmotion/
│
├── data_loader.py          # Text preprocessing, data loading, GloVe integration
├── model.py                # BiLSTM + Attention model architecture
├── trainer.py              # Training loop, evaluation, visualization
├── main.py                 # Entry point for training and evaluation
├── config.py               # All hyperparameters and paths
├── gui.py                  # Streamlit-based web interface
├── requirements.txt        # Required packages
└── README.md               # Project documentation

