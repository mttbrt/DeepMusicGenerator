#from keras import sequential
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from random import shuffle
import numpy as np

import os, sys, pickle

CHORD_CLASSES = 24+1 #12 major chords, 12 minor chords, and "no chord"

# Model constants
CHORD_EMBEDDING = 3
STEP_SIZE = 1
BATCH_SIZE = 1
HIDDEN_UNITS = 512
LEARNING_RATE = 0.01
ACTIVATION = 'softmax'
LOSS = 'categorical_crossentropy'

# Training constants
EPOCHS = 2

dataset = None
model = None

#Returns a one-hot encoding of a chord id
def id_to_vec(chord_id):
    if chord_id == None:
        return np.array([1 if i == CHORD_CLASSES-1 else 0 for i in range(CHORD_CLASSES)])
    return np.array([1 if i == chord_id else 0 for i in range(CHORD_CLASSES)])

def init_model():
    global model
    model = Sequential()
    model.add(Embedding(CHORD_CLASSES, CHORD_EMBEDDING, input_length = STEP_SIZE, batch_input_shape=(BATCH_SIZE, STEP_SIZE)))
    model.add(LSTM(HIDDEN_UNITS, stateful = False))
    model.add(Dense(CHORD_CLASSES))
    model.add(Activation(ACTIVATION))
    model.compile(RMSprop(learning_rate=LEARNING_RATE), LOSS)

# Splits the dataset in two subsets: the training set contains the {percentage} of
# files, the testing set contains the 1 - {percentage} of files
def split_dataset(percentage):
    global dataset
    shuffle(dataset)
    dataset_len = len(dataset)
    return dataset[:int(dataset_len*percentage)], dataset[int(dataset_len*percentage):]

# TODO: solve 'UserWarning: Converting sparse IndexedSlices to a dense Tensor'
def train_model(train_set, test_set):
    print('Training...')
    for e in range(1, EPOCHS + 1):
        print(f'Epoch {e}/{EPOCHS}')
        tot_epoch_loss = 0
        for i, song in enumerate(train_set):
            # [improvement] TODO: use CHORD_CLASSES-1 instead of None for "no chord" during preprocessing
            #for j, chord in enumerate(song):
            #    if not chord:
            #        song[j] = CHORD_CLASSES-1

            #TODO: this does not seem right for X and Y
            X = song
            Y = np_utils.to_categorical(X, num_classes = CHORD_CLASSES) # one-hot encode chords
            h = model.fit(X, Y, batch_size = BATCH_SIZE, shuffle = False, verbose = False)
            loss = h.history['loss'][0]
            model.reset_states()

            print(f'\tSong {i}: train loss {loss}')
            tot_epoch_loss += loss

        print(f'\tTRAIN MEAN LOSS: {tot_epoch_loss/len(train_set)}')

        # Test and backup model every epoch (could be changed)
        test_model(test_set)

        if not os.path.exists('model_backups'):
            os.makedirs('model_backups')
        model.save('model_backups/epoch_' + str(e) + '.pickle')

def test_model(test_set):
    print('Testing...')
    tot_test_loss = 0
    for i, song in enumerate(test_set):
        # [improvement] TODO: use CHORD_CLASSES-1 instead of None for "no chord" during preprocessing
        #for j, chord in enumerate(song):
        #    if not chord:
        #        song[j] = CHORD_CLASSES - 1

        X = song
        Y = np_utils.to_categorical(X, num_classes = CHORD_CLASSES) # one-hot encode chords
        loss = model.evaluate(X, Y, batch_size = BATCH_SIZE, verbose = False)
        model.reset_states()

        tot_test_loss += loss
        print(f'\tSong {i}: test loss {loss}')

    print(f'\tTEST MEAN LOSS: {tot_test_loss/len(test_set)}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python chord-generation.py <training data>')
        exit(0)

    input_file = sys.argv[1]
	#input is expected to be a pickling of a 2-dimensional python list so that raw_data[i][j] is the j-th chord of the i-th song
    dataset = pickle.load(open(input_file, 'rb'))

    # print(id_to_vec(dataset[0][0])) #debug

    ## TODO: everything lmao ##
    init_model()
    train_set, test_set = split_dataset(0.8)
    train_model(train_set, test_set)
