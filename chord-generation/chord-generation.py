#from keras import sequential
from keras.models import Sequential, load_model
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from random import shuffle
import numpy as np

import os, sys, pickle

CHORD_CLASSES = 24+1 # 12 major chords, 12 minor chords, and "no chord"

# Model constants
EMBEDDING_SIZE = 6 # size of embedding vector
CHORDS_BLOCK = 4 # how many chords pass to the network every time
BATCH_SIZE = 1 # how many sequences to process in parallel (> 1 for polyphonic generation)
HIDDEN_UNITS = 128 # number of LSTM cells
LEARNING_RATE = 0.0001
ACTIVATION = 'softmax'
LOSS = 'categorical_crossentropy'

# Training constants
EPOCHS = 32

dataset = None
model = None

# # Returns a one-hot encoding of a chord id
# def id_to_vec(chord_id):
#     if chord_id == None:
#         return np.array([1 if i == CHORD_CLASSES-1 else 0 for i in range(CHORD_CLASSES)])
#     return np.array([1 if i == chord_id else 0 for i in range(CHORD_CLASSES)])

def init_model():
    global model
    model = Sequential()
    model.add(Embedding(input_dim = CHORD_CLASSES, output_dim = EMBEDDING_SIZE, input_length = CHORDS_BLOCK, batch_input_shape = (BATCH_SIZE, CHORDS_BLOCK)))
    model.add(LSTM(HIDDEN_UNITS, batch_input_shape = (BATCH_SIZE, CHORDS_BLOCK, EMBEDDING_SIZE), return_sequences = False, stateful = True))
    model.add(Dense(CHORD_CLASSES))
    model.add(Activation(ACTIVATION))
    model.compile(RMSprop(learning_rate = LEARNING_RATE), LOSS)

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
        print(f'Epoch {e}/{EPOCHS}')
        tot_epoch_loss = 0
        shuffle(train_set)
        for j, song in enumerate(train_set):
            # preprocessing: song is split in multiple sliding windows of CHORDS_BLOCK elements
            xdata = []
            ydata = []
            for i in range(len(song) - CHORDS_BLOCK - 1):
                xdata.append(song[i:i+CHORDS_BLOCK])
                ydata.append(song[i+CHORDS_BLOCK])
            xdata = pad_sequences(xdata, maxlen=CHORDS_BLOCK)

            X = np.reshape(xdata, (xdata.shape[0], CHORDS_BLOCK))
            Y = np_utils.to_categorical(ydata, num_classes = CHORD_CLASSES) # one-hot encode chords
            stats = model.fit(X, Y, batch_size = BATCH_SIZE, shuffle = False, verbose = False)
            song_loss = stats.history['loss'][0]
            model.reset_states()

            print(f'\tSong {j}/{len(train_set)}: train loss {song_loss}')
            tot_epoch_loss += song_loss

        print(f'\tTRAIN MEAN LOSS: {tot_epoch_loss/len(train_set)}')

        # Test and backup model every epoch
        test_model(test_set)

        if not os.path.exists('model_backups'):
            os.makedirs('model_backups')
        model.save('model_backups/epoch_' + str(e) + '.pickle')

def test_model(test_set):
    print('Testing...')
    tot_test_loss = 0
    shuffle(test_set)
    for j, song in enumerate(test_set):
        # preprocessing: song is split in multiple sliding windows of CHORDS_BLOCK elements
        xdata = []
        ydata = []
        for i in range(len(song) - CHORDS_BLOCK - 1):
            xdata.append(song[i:i+CHORDS_BLOCK])
            ydata.append(song[i+CHORDS_BLOCK])
        xdata = pad_sequences(xdata, maxlen=CHORDS_BLOCK)

        X = np.reshape(xdata, (xdata.shape[0], CHORDS_BLOCK))
        Y = np_utils.to_categorical(ydata, num_classes = CHORD_CLASSES) # one-hot encode chords
        song_loss = model.evaluate(X, Y, batch_size = BATCH_SIZE, verbose = False)
        model.reset_states()

        print(f'\tSong {j}/{len(test_set)}: train loss {song_loss}')
        tot_test_loss += song_loss

    print(f'\tTEST MEAN LOSS: {tot_test_loss/len(test_set)}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python chord-generation.py <training data>')
        exit(0)

    input_file = sys.argv[1]
	#input is expected to be a pickling of a 2-dimensional python list so that raw_data[i][j] is the j-th chord of the i-th song
    dataset = pickle.load(open(input_file, 'rb'))

    # Build and train model
    init_model()
    train_set, test_set = split_dataset(0.8)
    train_model(train_set, test_set)

    # Debugging
    # model = load_model(f'model_backups/epoch_{EPOCHS}.pickle')
    # CHORDS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B', 'Cm', 'C#m', 'Dm', 'Ebm', 'Em', 'Fm', 'F#m', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm', 'NO CHORD']
    # for chord_sequence in [[3, 5, 6, 8], [7, 9, 11, 12]]: # C# (C#, Eb, F, F#, Ab, Bb, C) and Fm (F, G, Ab, Bb, C, C#, Eb)
    #     input_seq = pad_sequences([chord_sequence], maxlen=CHORDS_BLOCK)
    #     input_data = np.reshape(input_seq, (input_seq.shape[0], CHORDS_BLOCK))
    #     prediction = model.predict(input_data)
    #     predicted_chord = [list(np.where(pr == np.amax(pr))[0]) for pr in prediction][0][0]
    #     print(f'Given {chord_sequence} predicted {predicted_chord}({CHORDS[predicted_chord]})')
