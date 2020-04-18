#from keras import sequential
from keras.models import Sequential, load_model
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from random import shuffle
import numpy as np

import os, sys, argparse, pickle, progressbar

CHORD_CLASSES = 24+1 # 12 major chords, 12 minor chords, and "no chord"

# Model constants
EMBEDDING = False
EMBEDDING_SIZE = 6 if EMBEDDING else 1 # size of embedding vector
CHORDS_BLOCK = 4 # how many chords pass to the network every time
BATCH_SIZE = 1 # how many sequences to process in parallel (> 1 for polyphonic generation)
HIDDEN_UNITS = 128 # number of LSTM cells
LEARNING_RATE = 0.001
ACTIVATION = 'softmax'
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adadelta'

# Training constants
EPOCHS = 256

dataset = None
model = None
transition_matrix = {}

def init_model():
    global model
    model = Sequential(name = 'LSTM Chord Generator')
    if EMBEDDING:
        model.add(Embedding(input_dim = CHORD_CLASSES, output_dim = EMBEDDING_SIZE, input_length = CHORDS_BLOCK, batch_input_shape = (BATCH_SIZE, CHORDS_BLOCK), name='Embedding'))
    model.add(LSTM(HIDDEN_UNITS, batch_input_shape = (BATCH_SIZE, CHORDS_BLOCK, EMBEDDING_SIZE), return_sequences = False, stateful = True, name='LSTM'))
    model.add(Dense(CHORD_CLASSES, name=f'Dense_{ACTIVATION}', activation=ACTIVATION))
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['acc', 'mae'])

    print(model.summary())

# Splits the dataset in two subsets: the training set contains the {percentage}
# of files, the testing set contains the 1 - {percentage} of files
def split_dataset(percentage):
    global dataset
    shuffle(dataset)
    dataset_len = len(dataset)
    return dataset[:int(dataset_len*percentage)], dataset[int(dataset_len*percentage):]

def train_model(train_set, test_set):
    print('Training...')

    for e in range(1, EPOCHS + 1):
        print(f'\nEpoch {e}/{EPOCHS}')
        tot_epoch_loss = 0
        tot_epoch_acc = 0
        tot_epoch_mae = 0

        shuffle(train_set)
        prog_bar = progressbar.ProgressBar(maxval=len(train_set))
        for j, song in enumerate(train_set):
            # preprocessing: song is split in multiple sliding windows of {CHORDS_BLOCK} elements
            xdata = []
            ydata = []
            for i in range(len(song) - CHORDS_BLOCK - 1):
                xdata.append(song[i:i+CHORDS_BLOCK])
                ydata.append(song[i+CHORDS_BLOCK])
            xdata = np.array(xdata)

            X = xdata if EMBEDDING else np.reshape(xdata, (xdata.shape[0], CHORDS_BLOCK, 1))
            Y = np_utils.to_categorical(ydata, num_classes = CHORD_CLASSES) # one-hot encode chords
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

def test_model(test_set):
    print('Testing...')

    tot_test_loss = 0
    tot_test_acc = 0
    tot_test_mae = 0

    shuffle(test_set)
    prog_bar = progressbar.ProgressBar(maxval=len(test_set))
    for j, song in enumerate(test_set):
        # preprocessing: song is split in multiple sliding windows of {CHORDS_BLOCK} elements
        xdata = []
        ydata = []
        for i in range(len(song) - CHORDS_BLOCK - 1):
            xdata.append(song[i:i+CHORDS_BLOCK])
            ydata.append(song[i+CHORDS_BLOCK])
        xdata = np.array(xdata)

        X = xdata if EMBEDDING else np.reshape(xdata, (xdata.shape[0], CHORDS_BLOCK, 1))
        Y = np_utils.to_categorical(ydata, num_classes = CHORD_CLASSES) # one-hot encode chords
        stats = model.evaluate(X, Y, batch_size = BATCH_SIZE, verbose = False)
        model.reset_states()
        prog_bar.update(j+1)

        tot_test_loss += stats[0]
        tot_test_acc += stats[1]
        tot_test_mae += stats[2]

    print(f'\nACCURACY: {tot_test_acc/len(test_set)} | LOSS: {tot_test_loss/len(test_set)} | MAE: {tot_test_mae/len(test_set)}')

def performance_eval(test_sequences):
    compute_transition_matrix()
    model = load_model(f'model_backups/epoch_{EPOCHS}.pickle')

    for chord_sequence in test_sequences:
        sequence = np.array([chord_sequence])
        data = sequence if EMBEDDING else np.reshape(sequence, (sequence.shape[0], CHORDS_BLOCK, 1))
        prediction = model.predict(data)
        predicted_chord = [list(np.where(pr == np.amax(pr))[0]) for pr in prediction][0][0]
        print(f'Given {chord_sequence} predicted {predicted_chord}')

        str_pred = ''
        for i in range(CHORDS_BLOCK):
            str_pred += str(chord_sequence[i])
        print(f'Expected predictions: {transition_matrix[str_pred]}\n')

def compute_transition_matrix():
    # group all possible sequences of 4 elements and save their prediction
    global transition_matrix
    for song in dataset:
        for i in range(len(song) - CHORDS_BLOCK - 1):
            block = ''
            for j in range(CHORDS_BLOCK):
                block += str(song[i+j])
            prediction = str(song[i+CHORDS_BLOCK])

            # add or update chord prediction given {CHORDS_BLOCK} previous chords
            if block in transition_matrix:
                if prediction in transition_matrix[block]:
                    transition_matrix[block][prediction] += 1
                else:
                    transition_matrix[block][prediction] = 1
            else:
                transition_matrix[block] = {prediction : 1}

    # transform number of occurrences for each prediction in percentages
    for block in transition_matrix:
        tot_pred = sum(transition_matrix[block].values())
        for prediction in transition_matrix[block]:
            transition_matrix[block][prediction] /= tot_pred

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Train chord generation model.')
    parser.add_argument('--data', metavar = 'D', type = str, default = 'output_sequences.p', help = 'Specify the pickle file with chords sequences for each song. [default: output_sequences.p]')
    parser.add_argument('--embedding', metavar = 'E', type = int, default = 0, help = 'Adds an embedding layer with the specified dimension. [default: 0 (no embedding)]')
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

    if args.embedding == 0:
        EMBEDDING = False
    elif args.embedding > 0:
        EMBEDDING = True
        EMBEDDING_SIZE = args.embedding
    else:
        print('Embedding size must be greater than 0.')
        print('Type \'python chord-generation.py -h\' to get more information.')
        exit(0)

    if args.block > 0:
        CHORDS_BLOCK = args.block
    else:
        print('Number of chords in a block must be greater than 0.')
        print('Type \'python chord-generation.py -h\' to get more information.')
        exit(0)

    if args.epochs > 0:
        EPOCHS = args.epochs
    else:
        print('Number of epochs must be greater than 0.')
        print('Type \'python chord-generation.py -h\' to get more information.')
        exit(0)

    # Build and train model
    init_model()
    train_set, test_set = split_dataset(0.8)
    train_model(train_set, test_set)

    # Model evaluation
    # performance_eval([[21, 5, 0, 16], [0, 16, 0, 16], [5, 7, 0, 21], [7, 2, 4, 0], [16, 5, 21, 7]])
