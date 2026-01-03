## Bangla Fake News Detection using Deep learning
In this project, the dataset was preprocessed using **traditional NLP techniques tailored for the Bengali language**, including puncuation,stopword,extraspaces & special character cleaning, and tokenization. Language-specific preprocessing steps were applied to ensure better representation of Bengali text.

For feature extraction, different approaches were used based on the model requirements:
- **Stemming then TF-IDF vectorization** was applied to tokenized text and used as input for a **Random Forest classifier**, enabling effective capture of important terms and their relative importance.
- For semantic representation, **Word2Vec embeddings** were generated using tokenized text to learn contextual word relationships.

Finally, a **Convolutional Neural Network (CNN)** was trained using the Word2Vec-based representations to automatically learn hierarchical features from the text and perform fake news classification.

This hybrid approach combines **traditional machine learning models** with **deep learning techniques** to evaluate and compare their effectiveness on Bengali fake news detection.
## Dataset Description
Datast Link : https://www.kaggle.com/datasets/evilspirit05/bengali-fake-news-dataset

This dataset consists of a well-curated collection of **Bengali news articles** created to support research in **fake news detection**. The data has been carefully gathered and processed to help researchers and practitioners develop, train, and evaluate machine learning and deep learning models capable of distinguishing between **real and fake news** in the Bengali language.

### Key Features

- **Source**  
  News articles were scraped from multiple popular Bengali news websites and public APIs. Well-established and reputable news portals were prioritized to ensure diversity and authenticity of content.

- **Time Coverage**  
  The dataset spans news articles published between **January 2018 and November 2018**, providing a meaningful historical context for analyzing news trends during this period.

- **Language**  
  All content is written in **Bengali**, making the dataset especially valuable for NLP research in low-resource languages.
