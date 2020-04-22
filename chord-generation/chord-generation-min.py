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

CHORD_CLASSES = 24+1 # 12 major chords, 12 minor chords, and "no chord"


def one_hot_encode_chord(chord_id):
    return np.array([1 if i == chord_id else 0 for i in range(CHORD_CLASSES)])


def init_model(chord_block_size):
    global model
    model = Sequential(name = 'lstm-chord-generator')
    model.add(LSTM(64, input_shape = (chord_block_size, CHORD_CLASSES), name='LSTM1'))
    model.add(Dense(CHORD_CLASSES, activation='softmax'))

    optimizer = RMSprop(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', 'mae'])

    return model


def preprocess(dataset, chord_block_size):
    #one-hot encode everything
    one_hot_encoded_dataset = []
    for song in dataset:
        one_hot_encoded_song = []
        for chord in song:
            one_hot_encoded_song.append(one_hot_encode_chord(chord))
        one_hot_encoded_dataset.append(one_hot_encoded_song)

    #Create feature-target pairs
    X = []
    Y = []
    for sequence in one_hot_encoded_dataset:
        for i in range(len(sequence) - (chord_block_size + 1)):
            X.append(sequence[i:i+chord_block_size])
            Y.append(sequence[i+chord_block_size])
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


def train_model():
    print('Training...')

    train_set, test_set = split_dataset(0.8)
    for e in range(1, EPOCHS + 1):
        print(f'\nEpoch {e}/{EPOCHS}')
        tot_epoch_loss = 0
        tot_epoch_acc = 0
        tot_epoch_mae = 0

        shuffle(train_set)
        prog_bar = progressbar.ProgressBar(maxval=len(train_set))
        for j, song in enumerate(train_set):
            # preprocessing: song is split in multiple sliding windows of {chord_block_size} elements
            xdata = []
            ydata = []
            for i in range(len(song) - chord_block_size - 1):
                xdata.append(song[i:i+chord_block_size])
                ydata.append(song[i+chord_block_size])
            xdata = np.array(xdata)

            X = xdata if EMBEDDING else np.reshape(xdata, (xdata.shape[0], chord_block_size, 1))
            Y = utils.to_categorical(ydata, num_classes = CHORD_CLASSES) # one-hot encode chords
            stats = model.fit(X, Y, batch_size = BATCH_SIZE, shuffle = False, verbose = False)
            model.reset_states()
            prog_bar.update(j+1)

            tot_epoch_loss += stats.history['loss'][0]
            tot_epoch_acc += stats.history['acc'][0]
            tot_epoch_mae += stats.history['mae'][0]

        print(f'\nACCURACY: {tot_epoch_acc/len(train_set)} | LOSS: {tot_epoch_loss/len(train_set)} | MAE: {tot_epoch_mae/len(train_set)}')

        # Test and backup model every epoch
        test_model(test_set)

        if not os.path.exists('model_backups'):
            os.makedirs('model_backups')
        model.save('model_backups/epoch_' + str(e) + '.pickle')


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
