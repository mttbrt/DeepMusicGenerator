from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from random import shuffle

import numpy as np
import os, sys, argparse, pickle, random
import matplotlib.pyplot as plt

CHORD_CLASSES = 24+3 # 12 major chords, 12 minor chords, "no chord", song start and song end tags

SONG_START = 25
SONG_END = 26


def one_hot_encode_chord(chord_id):
    return np.array([1 if i == chord_id else 0 for i in range(CHORD_CLASSES)])


def init_model(chord_block_size):
    global model
    model = Sequential(name = 'lstm-chord-generator')
    model.add(LSTM(256, stateful=True, dropout=0.1, batch_input_shape = (1, 1, CHORD_CLASSES), name='LSTM1'))
    model.add(Dense(CHORD_CLASSES, activation='softmax'))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', 'mae'])

    return model


def preprocess(dataset, chord_block_size):
    #one-hot encode everything
    one_hot_encoded_dataset = []
    for song in dataset:
        one_hot_encoded_song = [one_hot_encode_chord(SONG_START)]
        for chord in song[1:]:
            one_hot_encoded_song.append(one_hot_encode_chord(chord))
        one_hot_encoded_song.append(one_hot_encode_chord(SONG_END))
        one_hot_encoded_dataset.append(one_hot_encoded_song)

    #Create feature-target pairs
    sequences = []
    for sequence in one_hot_encoded_dataset:
        X = list(map(lambda x: [x], sequence[:-1]))
        Y = sequence[1:]
        X = np.array(X)
        Y= np.array(Y)
        sequences.append((X,Y))

    return sequences

def train_model(model, sequences, epochs):
    for i in range(epochs):
        print(f'Epoch {i+1}/{epochs}:')
        for j, sequence in enumerate(sequences):
            print(f'Sequence {j+1}/{len(sequences)}:')
            X, Y = sequence
            model.fit(X, Y, batch_size=1, epochs=1, shuffle=False)
            model.reset_states()
    return model

def make_song(model, seed, max_length=None):
    song = [SONG_START] + seed

    #priming
    for chord in song[:-1]:
        sample = np.array([[one_hot_encode_chord(chord)]])
        model.predict_classes(sample, batch_size=1)

    #generation
    chord_index = len(song)-1
    while True:
        sample = np.array([[one_hot_encode_chord(song[chord_index])]])
        new_chord = model.predict_classes(sample, batch_size=1)[0]
        
        song.append(new_chord)
        if new_chord == SONG_END:
            break

        chord_index += 1
        if max_length != None and chord_index >= max_length:
            break

    model.reset_states()
    return song


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Train chord generation model.')
    parser.add_argument('--data', metavar = 'D', type = str, default = 'output_sequences.p', help = 'Specify the pickle file with chords sequences for each song. [default: output_sequences.p]')
    parser.add_argument('--block', metavar = 'B', type = int, default = 4, help = 'Chord sequence size. [default: 4]')
    parser.add_argument('--epochs', metavar = 'e', type = int, default = 256, help = 'Epoch on which the network will be trained. [default: 256]')
    args = parser.parse_args()

    try:
    	#input is expected to be a pickling of a 2-dimensional python list so that raw_data[i][j] is the j-th chord of the i-th song
        dataset = pickle.load(open(args.data, 'rb'))
    except:
        print('Specified file does not exists.')
        print('Type \'python chord-generation.py -h\' to get more information.')
        exit(0)

    if args.block < 0:
        print('Number of chords in a block must be greater than 0.')
        print('Type \'python chord-generation.py -h\' to get more information.')
        exit(0)

    if args.epochs < 0:
        print('Number of epochs must be greater than 0.')
        print('Type \'python chord-generation.py -h\' to get more information.')
        exit(0)

    #Build and train model
    model = init_model(args.block)
    print(model.summary())

    print(f'Preprocessing {len(dataset)} chord sequences...')
    sequences = preprocess(dataset, args.block)
    
    model = train_model(model, sequences, args.epochs)

    song = make_song(model, [21,7,0,5], 100)
    print(song)
