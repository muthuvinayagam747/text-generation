import streamlit as st
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Twitter Text Generation",
    page_icon="🐦"
)


# ============================================
# TITLE
# ============================================

st.title("🐦 Twitter Text Generation using LSTM")

st.write(
    "Generate new text using an LSTM model trained "
    "on Twitter-style text data."
)


# ============================================
# DATASET
# ============================================

tweets = [
    "I love learning artificial intelligence",
    "I love learning machine learning",
    "I love learning deep learning",
    "Machine learning is very interesting",
    "Artificial intelligence is changing the world",
    "Deep learning is useful for many applications",
    "Python is easy to learn",
    "Python is useful for machine learning",
    "I enjoy working with Python",
    "I enjoy learning new technologies",
    "Technology is changing our daily life",
    "Artificial intelligence makes technology better",
    "Machine learning helps solve real problems",
    "Deep learning is a powerful technology",
    "I love programming with Python",
    "Learning Python is very useful",
    "I like artificial intelligence projects",
    "I like machine learning projects",
    "I like deep learning projects",
    "Learning technology is fun"
]


# ============================================
# TOKENIZATION
# ============================================

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

total_words = len(tokenizer.word_index) + 1

input_sequences = []

for tweet in tweets:

    token_list = tokenizer.texts_to_sequences(
        [tweet]
    )[0]

    for i in range(1, len(token_list)):

        input_sequences.append(
            token_list[:i+1]
        )


max_sequence_length = max(
    len(x) for x in input_sequences
)

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_length,
        padding="pre"
    )
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]


# ============================================
# LSTM MODEL
# ============================================

model = Sequential([
    Embedding(
        total_words,
        64,
        input_length=X.shape[1]
    ),

    LSTM(100),

    Dense(
        total_words,
        activation="softmax"
    )
])


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam"
)


# ============================================
# TRAIN MODEL
# ============================================

@st.cache_resource
def train_model():

    model.fit(
        X,
        y,
        epochs=100,
        verbose=0
    )

    return model


model = train_model()


# ============================================
# TEXT GENERATION
# ============================================

def generate_text(seed_text, next_words):

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences(
            [seed_text]
        )[0]

        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_length - 1,
            padding="pre"
        )

        predicted = model.predict(
            token_list,
            verbose=0
        )

        predicted_word_index = np.argmax(
            predicted
        )

        output_word = ""

        for word, index in tokenizer.word_index.items():

            if index == predicted_word_index:

                output_word = word
                break

        if output_word == "":
            break

        seed_text += " " + output_word

    return seed_text


# ============================================
# USER INPUT
# ============================================

st.header("Generate Text")

seed_text = st.text_input(
    "Enter starting text:",
    "I love learning"
)

number_of_words = st.slider(
    "Number of words to generate:",
    1,
    10,
    5
)


# ============================================
# BUTTON
# ============================================

if st.button("Generate Text"):

    result = generate_text(
        seed_text,
        number_of_words
    )

    st.success("Generated Text")

    st.write(result)


# ============================================
# INFORMATION
# ============================================

st.write("---")

st.info(
    "This is an educational demonstration "
    "of text generation using an LSTM neural network."
)
