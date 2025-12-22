# preprocessing.py

import re
import numpy as np
import nltk

# ADD THIS LINE:
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # Newer NLTK resource
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from pathlib import Path

# -----------------------------
# Load static resources ONCE
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Load Word2Vec model
W2V_PATH = BASE_DIR / "artifacts" / "word2vec.model"
w2v_model = Word2Vec.load(str(W2V_PATH))

VECTOR_SIZE = w2v_model.vector_size

# Load stopwords
STOPWORDS_PATH = BASE_DIR / "artifacts" / "stopwords.txt"

with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
    bengali_stopwords = set(line.strip() for line in f if line.strip())


# -----------------------------
# Text preprocessing functions
# -----------------------------

def remove_punctuation(text: str) -> str:
    """
    Keep Bangla unicode, English letters, numbers, spaces
    """
    return re.sub(r"[^\u0980-\u09FFa-zA-Z0-9\s]", "", text)


def remove_extra_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords_from_text(text: str) -> str:
    words = text.split()
    filtered_words = [w for w in words if w not in bengali_stopwords]
    return " ".join(filtered_words)


def tokenize_text(text: str):
    return word_tokenize(text)


# -----------------------------
# Vectorization
# -----------------------------

def get_news_vector(tokens: list[str]) -> np.ndarray:
    """
    Convert tokens to average Word2Vec embedding
    """
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if not vectors:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


# -----------------------------
# Full preprocessing pipeline
# -----------------------------

def preprocess_text(text: str) -> np.ndarray:
    """
    Full preprocessing pipeline:
    raw text -> numeric vector
    """

    if text is None:
        text = ""

    text = remove_punctuation(text)
    text = remove_extra_spaces(text)
    text = remove_stopwords_from_text(text)

    tokens = tokenize_text(text)

    vector = get_news_vector(tokens)

    return vector
