# -*- coding: utf-8 -*-


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import string
from string import digits
import matplotlib.pyplot as plt
# %matplotlib inline
import re
import pickle

import seaborn as sns
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model

sentences=pd.read_csv("/content/Hindi_English_Truncated_Corpus.csv")

print("The total number of NaN values in the DataFrame is:\n", pd.isnull(sentences).sum())

sentences=sentences[~pd.isnull(sentences['english_sentence'])]

sentences.drop_duplicates(inplace=True)

sentences=sentences.sample(n=25000,random_state=42)
sentences.shape

# Lowercase all characters
# Convert the English and Hindi sentences to lowercase
sentences['english_sentence'] = sentences['english_sentence'].str.lower()
sentences['hindi_sentence'] = sentences['hindi_sentence'].str.lower()
sentences.shape

# Remove quotes from 'english_sentence' and 'hindi_sentence' columns
sentences['english_sentence'] = sentences['english_sentence'].str.replace("'", "")
sentences['hindi_sentence'] = sentences['hindi_sentence'].str.replace("'", "")
sentences.shape

# Define a function to remove special characters from a string
def remove_special_chars(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

# Apply the function to the 'english_sentence' and 'hindi_sentence' columns of the DataFrame
sentences['english_sentence'] = sentences['english_sentence'].apply(remove_special_chars)
sentences['hindi_sentence'] = sentences['hindi_sentence'].apply(remove_special_chars)
sentences.shape

# Define a function to remove all digits from a string
def remove_digits(text):
    return text.translate(str.maketrans('', '', string.digits))

# Apply the function to the 'english_sentence' and 'hindi_sentence' columns
sentences['english_sentence'] = sentences['english_sentence'].apply(remove_digits)
sentences['hindi_sentence'] = sentences['hindi_sentence'].apply(remove_digits)


# Remove numeric characters from the 'hindi_sentence' column
sentences['hindi_sentence'] = sentences['hindi_sentence'].str.replace("[२३०८१५७९४६]", "", regex=True)

# Define a function to clean up the text
def clean_text(text):
    # Remove leading and trailing spaces
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(" +", " ", text)
    return text

# Apply the clean_text function to the 'english_sentence' and 'hindi_sentence' columns
sentences['english_sentence'] = sentences['english_sentence'].apply(clean_text)
sentences['hindi_sentence'] = sentences['hindi_sentence'].apply(clean_text)

# Add start and end tokens to target sequences
# Add start and end tokens to target sequences
sentences['hindi_sentence'] = 'START_ ' + sentences['hindi_sentence'] + ' _END'
sentences['hindi_sentence']

### Get English and Hindi Vocabulary
# Split the 'english_sentence' column into words and flatten the resulting list
eng_words = [word for eng in sentences['english_sentence'] for word in str(eng).split()]

# Create a set of unique English words in the DataFrame
eng_words_collection = set(eng_words)

# Split the 'english_sentence' column into words and flatten the resulting list
hindi_words = [word for hindi in sentences['hindi_sentence'] for word in str(hindi).split()]

# Create a set of unique English words in the DataFrame
hindi_words_collection = set(hindi_words)

len(hindi_words_collection)

# Compute the length of the English sentences and store the result in a new column
sentences['length_eng_sentence'] = sentences['english_sentence'].str.split().str.len()

# Compute the length of the Hindi sentences and store the result in a new column
sentences['length_hin_sentence'] = sentences['hindi_sentence'].str.split().str.len()

sentences.head()

sentences[sentences['length_eng_sentence']>30].shape

sentences=sentences[sentences['length_eng_sentence']<=20]
sentences=sentences[sentences['length_hin_sentence']<=20]

sentences.shape

input_words = sorted(list(eng_words_collection))
target_words = sorted(list(hindi_words_collection))
num_encoder_tokens = len(eng_words_collection)
num_decoder_tokens = len(hindi_words_collection)
num_encoder_tokens, num_decoder_tokens

num_decoder_tokens += 1 #for zero padding

input_token_index = {word: i+1 for i, word in enumerate(input_words)}
target_token_index = {word: i+1 for i, word in enumerate(target_words)}

# Create dictionaries to map indices to input and target vocabulary words
reverse_input_char_index = {idx: word for word, idx in input_token_index.items()}
reverse_target_char_index = {idx: word for word, idx in target_token_index.items()}

sentences = shuffle(sentences)
sentences.head(10)

x, y = sentences['english_sentence'], sentences['hindi_sentence']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)
x_train.shape, x_test.shape

x_train.to_pickle('X_train.pkl')
x_test.to_pickle('X_test.pkl')

def generate_batch(X = x_train, y = y_train, batch_size = 128):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                for t, word in enumerate(target_text.split()):
                    if t<len(target_text.split())-1:
                        decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data], decoder_target_data)

latent_dimension=300

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens, latent_dimension, mask_zero = True)(encoder_inputs)
encoder_LSTM = LSTM(latent_dimension, return_state=True)
encoder_outputs, state_h, state_c = encoder_LSTM(enc_emb)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# Decoder
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dimension, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_LSTM = LSTM(latent_dimension, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_LSTM(dec_emb,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.summary()

train_samples = len(x_train)
val_samples = len(x_test)
batch_size = 8
epochs = 100

max_length_src=max(sentences['length_hin_sentence'])
max_length_tar=max(sentences['length_eng_sentence'])

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=40)

# Train the model with early stopping
model.fit_generator(generator=generate_batch(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=train_samples//batch_size,
                    epochs=epochs,
                    validation_data=generate_batch(x_test, y_test, batch_size=batch_size),
                    validation_steps=val_samples//batch_size, callbacks=[early_stopping])

model.save_weights('nmt_weights.h5')

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dimension,))
decoder_state_input_c = Input(shape=(latent_dimension,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_LSTM(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 50):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

train_gen = generate_batch(x_train, y_train, batch_size = 1)
k=-1

k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', x_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', x_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', x_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', x_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])

k+=1
(input_seq, actual_output), _ = next(train_gen)
decoded_sentence = decode_sequence(input_seq)
print('Input English sentence:', x_train[k:k+1].values[0])
print('Actual Hindi Translation:', y_train[k:k+1].values[0][6:-4])
print('Predicted Hindi Translation:', decoded_sentence[:-4])
