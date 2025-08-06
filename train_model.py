import numpy as np
import pandas as pd
import nltk
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------
# 1. DATA ACQUISITION AND PREPARATION
# ----------------------------------------------------------------
print("--- 1. Acquiring and Preparing Data ---")
# Download the Gutenberg corpus if not already present. Gutenberg is a great free resource for large book datasets that are comparable to text datasets you may find in the wild.
nltk.download('gutenberg')
from nltk.corpus import gutenberg

# --- CHANGE: Load the raw text of Jane Austen's Emma ---
raw_text = gutenberg.raw('austen-emma.txt')

# --- CHANGE: Save to a local file named 'emma.txt' ---
with open('emma.txt', 'w', encoding='utf-8') as f:
    f.write(raw_text)

# Load and preprocess the text. Convert to lowercase for uniformity. This helps reduce the vocabulary size.
with open('emma.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()

print("Text from 'austen-emma.txt' loaded and converted to lowercase.")

# ----------------------------------------------------------------
# 2. TOKENIZATION AND SEQUENCE GENERATION
# ----------------------------------------------------------------
print("\n--- 2. Tokenizing and Generating N-Gram Sequences ---")
tokenizer = Tokenizer()
tokenizer.fit_on_text([text])
total_words = len(tokenizer.word_index) + 1

# Create a reverse mapping from index to word for faster lookups
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

# Create n-gram sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure uniform length
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
print(f"Total sequences: {len(input_sequences)}, Max sequence length: {max_sequence_len}")

# ----------------------------------------------------------------
# 3. PREPARE DATA FOR MODEL TRAINING
# ----------------------------------------------------------------
print("\n--- 3. Preparing Data for Model ---")
# Features (X) are all words except the last one in each sequence
X = input_sequences[:, :-1]
# Labels (y) are the last word of each sequence
labels = input_sequences[:, -1]
# One-hot encode the labels
y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# ----------------------------------------------------------------
# 4. BUILD AND TRAIN THE LSTM MODEL
# ----------------------------------------------------------------
print("\n--- 4. Building and Training the LSTM Model ---")
model = Sequential([
    Embedding(total_words, 100, input_length=max_sequence_len - 1),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Use EarlyStopping to prevent overfitting and save time
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# ----------------------------------------------------------------
# 5. TEST PREDICTION AND SAVE ARTIFACTS
# ----------------------------------------------------------------
print("\n--- 5. Saving Model and Tokenizer ---")

def predict_next_word(model, tokenizer, index_to_word_map, text, max_len):
    """Predicts the next word in a sequence."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=1)[0]
    return index_to_word_map.get(predicted_index, "Unknown")

# --- Test the function with a line from Emma ---
input_text = "If I loved you less, I might be"
predicted_word = predict_next_word(model, tokenizer, index_to_word, input_text, max_sequence_len)
print(f"Input: '{input_text}'")
print(f"Predicted next word: '{predicted_word}'")

# --- Save the model with a new name ---
model.save("emma_next_word_lstm.h5")
print("Model saved to emma_next_word_lstm.h5")

# --- Save the tokenizer with a new name ---
with open('emma_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved to emma_tokenizer.pickle")