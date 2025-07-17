# Sentiment Comparison of Political Leaders Using NLP and DL

This project performs **multi-label emotion classification** on political leader comments using **Natural Language Processing (NLP)** and **Deep Learning** techniques. It compares the performance of CNN, Bi-LSTM, and CNN-BiLSTM models and includes a **Streamlit app** for interactive sentiment prediction.

---

##  Project Highlights

- ✅ **Text Cleaning** and Preprocessing
- ✅ **Tokenization** and Padding
- ✅ Multi-label emotion tagging with:
  - 🧠 CNN (Convolutional Neural Network)
  - 🔁 Bi-LSTM (Bidirectional LSTM)
  - 📊 CNN-BiLSTM hybrid model
- ✅ **Model Evaluation**: Hamming Loss, AUC Score, F1 Score
- ✅ **Streamlit Web App** to input text and view predicted emotions

---

##  Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas, Numpy**
- **Streamlit** for UI

---

## Model Performance

| Model        | Hamming Loss | AUC (micro) | Best F1 Score | Exact Match Accuracy |
|--------------|--------------|-------------|---------------|-----------------------|
| CNN          | 0.0012       | 0.9727      | 0.9136        | 99.08%                |
| Bi-LSTM      | 0.0036       | 0.9703      | —             | 97.55%                |
| CNN-BiLSTM   | 0.0071       | 0.9686      | —             | 94.76%                |

---

##  Streamlit App

You can run the app locally using:

```bash
streamlit run app.py
