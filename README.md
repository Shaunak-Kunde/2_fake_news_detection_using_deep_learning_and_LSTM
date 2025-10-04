Of course. Here is the updated README.md with a new section explaining the model choice, as you requested.

Markdown

# Fake News Detection using Deep Learning and LSTM 🧠

In this project, I implemented a deep learning model to detect and classify "fake" vs. "real" news articles. This was part of my Kwiklabs Google Cloud Platform hands-on lab experience. Using a dataset of news articles, this notebook walks through the entire machine learning workflow, from data cleaning and exploratory analysis to building, training, and evaluating a sophisticated NLP model.

The core of this project is a Bidirectional Long Short-Term Memory (LSTM) network, which is highly effective for sequence-based tasks like natural language processing, as it can understand the context from both past and future words in a sentence.

---

## 📋 Project Workflow

The project follows these key steps:

1.  **Data Loading & Initial Analysis**: Two separate datasets (`True.csv` and `Fake.csv`) are loaded, each containing thousands of news articles. An initial analysis is performed to understand the size, memory usage, and basic structure of the data.

2.  **Exploratory Data Analysis (EDA) & Preprocessing**:
    * A target column `isfake` is created, labeling true news as `1` and fake news as `0`.
    * The two datasets are concatenated into a single DataFrame.
    * The `title` and `text` of each article are combined into a single feature for comprehensive analysis.
    * Data visualization is performed using `seaborn` and `WordCloud` to understand the distribution of news subjects and to visualize the most frequent words in both fake and real news articles.

3.  **Data Cleaning (NLP Pipeline)**:
    * Text data is cleaned by removing stopwords (common words like "the", "a", "in") and short words using the `nltk` and `gensim` libraries.
    * This preprocessing step ensures that only meaningful words are kept, which improves model performance and reduces noise.

4.  **Tokenization and Padding**:
    * The cleaned text is tokenized, meaning each word is converted into a unique integer.
    * Since neural networks require inputs of a consistent length, the sequences of tokens are padded (or truncated) to a fixed length of 40 words.

5.  **Model Building and Training**:
    * A `Sequential` model is built using TensorFlow/Keras.
    * **Embedding Layer**: Converts the integer-encoded words into dense vector representations.
    * **Bidirectional LSTM Layer**: The core of the model, this layer processes the text sequence in both forward and backward directions to capture long-range dependencies and context.
    * **Dense Layers**: A fully connected layer with a `ReLU` activation function followed by a final output layer with a `sigmoid` activation for binary classification.
    * The model is compiled with an `adam` optimizer and `binary_crossentropy` loss function, then trained for 2 epochs.

6.  **Model Evaluation**:
    * The trained model is used to make predictions on the unseen test data.
    * The model's performance is evaluated using an **accuracy score** and a **confusion matrix**, which provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.

---

## 🤔 Model Selection: Why LSTM over Naive Bayes?

While classic algorithms like Naive Bayes are often used for text classification, a Bidirectional LSTM was chosen for this project due to its superior ability to understand context.

* **Why Naive Bayes Was Not Used**: Naive Bayes operates on a "bag-of-words" principle, treating words as independent features. This approach is fast but fails to capture the order of words and the contextual nuances within sentences. Detecting fake news often depends on understanding these subtle relationships, which is a significant limitation of the Naive Bayes model.

* **Why LSTM Was Chosen**: The Bidirectional LSTM is a type of recurrent neural network designed specifically for sequential data like text. It processes articles word-by-word, maintaining a "memory" of the context from both the beginning and the end of a sentence. This allows it to learn complex patterns and long-range dependencies, making it far more effective at discerning the subtle differences that often distinguish fake from real news.

---

## 📈 Results

The trained LSTM model achieved an impressive **accuracy of approximately 99.68%** on the test dataset. The confusion matrix further confirms the model's excellent performance, showing a very low number of misclassifications and demonstrating its effectiveness in distinguishing between real and fake news articles.

---
<img width="560" height="403" alt="image" src="https://github.com/user-attachments/assets/d8c6d26e-416a-4bef-adc2-50f1e766f828" />

## 🛠️ Tools & Libraries Used

-   **Python 3.x**
-   **TensorFlow & Keras** for building and training the deep learning model.
-   **Pandas & NumPy** for data manipulation and numerical operations.
-   **NLTK, spaCy, & Gensim** for advanced natural language processing and text cleaning.
-   **Scikit-learn** for splitting data and model evaluation.
-   **Matplotlib, Seaborn, & Plotly** for data visualization.
-   **WordCloud** for creating insightful word cloud visualizations.


You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
