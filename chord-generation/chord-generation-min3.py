import tensorflow as tf

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

SEQ_LEN = 8
BATCH_SIZE = 8



def init_model(chord_block_size):
    global model
    model = Sequential(name = 'lstm-chord-generator')
    model.add(Embedding(CHORD_CLASSES, 32, batch_input_shape=(BATCH_SIZE, None)))
    model.add(LSTM(256, return_sequences=True, name='LSTM1'))
    model.add(Dense(CHORD_CLASSES, activation='softmax'))

    return model


def preprocess(dataset, chord_block_size):
    all_data = []
    for song in dataset:
        sequences_in_song = len(song) // (SEQ_LEN+1)
        all_data += song[:(SEQ_LEN+1)*sequences_in_song]

    dataset = tf.data.Dataset.from_tensor_slices(all_data)
    sequences = dataset.batch(SEQ_LEN+1, drop_remainder=True)
    
    def split_input_target(chunk):
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)

    return dataset


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
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics=['acc'])

    print(f'Preprocessing {len(dataset)} chord sequences...')
    dataset = preprocess(dataset, args.block)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    history = model.fit(dataset, epochs=args.epochs, callbacks=[checkpoint_callback])
