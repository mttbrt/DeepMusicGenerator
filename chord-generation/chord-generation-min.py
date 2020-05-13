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

CHORD_CLASSES = 24+1 # 12 major chords, 12 minor chords, and "no chord"


def one_hot_encode_chord(chord_id): #TODO: move this logic to chord extraction
    if chord_id >= CHORD_CLASSES:
        raise ValueError(f'Chord id must be between 0 and {CHORD_CLASSES}. Found {chord_id}')
    return [1 if i == chord_id else 0 for i in range(CHORD_CLASSES)]


def init_model(chord_block_size):
    global model
    model = Sequential(name = 'lstm-chord-generator')
    model.add(LSTM(256, dropout=0.1, input_shape = (chord_block_size, CHORD_CLASSES+1), name='LSTM1'))
    model.add(Dense(CHORD_CLASSES, activation='softmax'))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', 'mae'])

    return model


def preprocess(dataset, chord_block_size):
    #one-hot encode everything
    one_hot_encoded_dataset = []
    for song in dataset:
        one_hot_encoded_song = []
        song_length = len(song)
        for i, chord in enumerate(song):
            one_hot_encoded_song.append([i/song_length] + one_hot_encode_chord(chord))
        one_hot_encoded_dataset.append(one_hot_encoded_song)

    #Create feature-target pairs
    X = []
    Y = []
    for sequence in one_hot_encoded_dataset:
        for i in range(1, len(sequence) - (chord_block_size + 1)):
            X.append(sequence[i:i+chord_block_size])
            Y.append(sequence[i+chord_block_size][1:])
    X = np.array(X)
    Y= np.array(Y)

    return X, Y

def split_validation(X, Y, split):
    samples = len(X)
    validation_indices = random.choices(range(samples), k=int(split*samples))
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    
    for i in range(samples):
        if i in validation_indices:
            X_val.append(X[i])
            Y_val.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_val = np.array(X_val)
    Y_val = np.array(Y_val)
    return X_train, Y_train, X_val, Y_val


def make_song(model, seed, length, location=False):
    song = seed
    block_length = len(seed)
    index_window = list(range(block_length))

    for i in range(length-block_length):
        if location:
            sample = np.array([[[i/length]+one_hot_encode_chord(song[i]) for i in index_window]])
        else:
            sample = np.array([[one_hot_encode_chord(song[i]) for i in index_window]])

        new_chord = model.predict_classes(sample)[0]
        
        song.append(new_chord)
        index_window = index_window[1:]+[index_window[-1]+1]

    print(len(song))
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
    X, Y = preprocess(dataset, args.block)
    X_train, Y_train, X_val, Y_val = split_validation(X, Y, 0.2)
    print(f'{len(X)} points available')

    history = model.fit(X_train, Y_train, batch_size=32, epochs=args.epochs, validation_data=(X_val, Y_val))

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    song = make_song(model, [21, 21, 7, 7, 0, 0, 5, 5], 180, location=True)
    print(song)
