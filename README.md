# Next-Word Prediction using an LSTM Network and Streamlit
This project demonstrates how to build, train, and deploy a deep learning model for next-word prediction. The model uses a Long Short-Term Memory (LSTM) network, a type of Recurrent Neural Network (RNN), trained on the text of Jane Austen's Emma. An interactive web application is provided using Streamlit to showcase the model's predictions.

<br>

## Domains of Expertise at Play
Data Acquisition: Fetches text directly from the NLTK Gutenberg corpus.

Text Preprocessing: Utilizes the Keras Tokenizer for efficient word embedding and sequence generation.

N-Gram Modeling: Creates training data by generating n-gram sequences from the source text.

Deep Learning Model: Implements a sequential LSTM network in Keras with Embedding, Dropout, and Dense layers.

Optimized Training: Uses EarlyStopping to prevent overfitting and reduce training time.

Interactive UI: A user-friendly web interface built with Streamlit to input text and get real-time predictions.

Model Persistence: The trained model and tokenizer are saved to disk, allowing the web app to run independently of the training script.


## How It Works
### Data Preparation (train_model.py):

The script downloads the text of Emma and saves it locally. The text is tokenized, mapping each unique word to an integer. The entire text is converted into overlapping n-gram sequences. For example, the line "to be or not to be" becomes ['to', 'be'], ['to', 'be', 'or'], ['to', 'be', 'or', 'not'], and so on. These sequences are padded with leading zeros to ensure they are all of a uniform length. The sequences are split into features (X) and labels (y). For each sequence, X is all words except the last, and y is the last word.

Zero padding and n-gram smoothing or backoff mitigations could be used in future examples as Emma is, while beautiful on its own, categorized as a classical literature piece. For this use case, and this training data, the prediction use case should be other similar titles released around the same time period.

This example can be further extrapolated into training data sets with a focus in a specific domain area. For example, you wouldn't expect a training data set based on Dr. Suessi libraries to be able to predict next word probability for Biomedical companies. To achieve these goals, the next step may be utilizing larger training datasets and/or LLM integration with MCP architecture.

The subtext here is that it's important to define the use case and be clear about the training datasets necessary to produce a result that your stakeholders will expect, especially in scenarios where LLM tokens are out of scope or budget like a prototype or MVP.

### Model Training (train_model.py):

An LSTM network is defined. The Embedding layer creates dense vector representations of the words. The LSTM layers learn the sequential patterns in the text.

The model is compiled with categorical_crossentropy loss, suitable for multi-class classification (predicting one word out of all possible words).

The model is trained on the prepared data, with EarlyStopping monitoring the validation loss to find the optimal number of epochs.

Finally, the trained model (next_word_lstm.h5) and the tokenizer (tokenizer.pickle) are saved.

### Interactive Prediction (app.py):

Streamlit is a simple, free UI that provides a frontend for backend applications like this one. The Streamlit application loads the saved model and tokenizer. The user enters a sequence of text. The input text is preprocessed using the same tokenization and padding logic as the training data. The model predicts the most likely next word, which is then displayed to the user.

The local URL provided by Streamlit is usually http://localhost:8501

