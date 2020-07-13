from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense

from natsort import natsorted, ns
from random import shuffle
import numpy as np
import argparse, pickle, re, progressbar, os

REST = 0
MIDI_LOWEST = 21 # Lowest midi note taken into consideration: A0
MIDI_HIGHEST = 108 # Highest midi note taken into consideration: C8
MIDI_NOTES = MIDI_HIGHEST - MIDI_LOWEST + 1
NOTE_CLASSES = MIDI_NOTES + 1 # 88 notes of the piano and rest

EPOCHS = 0
START_EPOCH = 0
HIDDEN_UNITS = 12 # number of LSTM cells - formula: N_h = N_s / (alpha * (N_i + N_o))
BATCH_SIZE = 1 # how many sequences to process in parallel
NOTES_BLOCK = 4 # how many notes pass to the network every time
EMBEDDING_SIZE = NOTE_CLASSES # size of embedding vector
ACTIVATION = 'softmax'
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'

dataset = None
model = None

def init_model(load_last=False):
    global model
    if load_last:
        last_backup = natsorted(os.listdir('model_backups', alg=ns.IGNORECASE))[-1]
        model = load_model(os.path.join('model_backups', last_backup))

        global START_EPOCH
        START_EPOCH = int(re.search('_(.+?)\.', last_backup).group(1))
    else:
        model = Sequential(name = 'lstm-melody-generator')
        model.add(LSTM(HIDDEN_UNITS,
                        activation='tanh',
                        recurrent_activation='sigmoid',
                        recurrent_dropout=0,
                        unroll=False,
                        use_bias=True,
                        batch_input_shape = (BATCH_SIZE, NOTES_BLOCK, EMBEDDING_SIZE),
                        return_sequences = False,
                        stateful = True,
                        name='LSTM'))
        model.add(Dropout(0.3))
        model.add(Dense(NOTE_CLASSES, name=f'Dense_{ACTIVATION}', activation=ACTIVATION))
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['acc', 'mae'])

    print(model.summary())

def train_model():
    print('Training...')

    shuffle(dataset)
    train_set, test_set = dataset[:int(len(dataset)*0.7)], dataset[int(len(dataset)*0.7):]
    for e in range(START_EPOCH + 1, EPOCHS + 1):
        print(f'\nEpoch {e}/{EPOCHS}')
        tot_epoch_loss = 0
        tot_epoch_acc = 0
        tot_epoch_mae = 0

        shuffle(train_set)
        prog_bar = progressbar.ProgressBar(maxval=len(train_set))
        for j, song in enumerate(train_set):
            # preprocessing: song is split in multiple sliding windows of {NOTES_BLOCK} elements
            xdata = []
            ydata = []
            for i in range(len(song) - NOTES_BLOCK - 1):
                xdata.append(song[i:i+NOTES_BLOCK])
                ydata.append(song[i+NOTES_BLOCK])

            X = np.array(xdata)
            Y = np.array(ydata)
            stats = model.fit(X, Y, batch_size = BATCH_SIZE, shuffle = False, verbose = False)
            model.reset_states()
            prog_bar.update(j+1)

            tot_epoch_loss += stats.history['loss'][0]
            tot_epoch_acc += stats.history['acc'][0]
            tot_epoch_mae += stats.history['mae'][0]

        print(f'\nACCURACY: {tot_epoch_acc/len(train_set)} | LOSS: {tot_epoch_loss/len(train_set)} | MAE: {tot_epoch_mae/len(train_set)}')

        if not os.path.exists('model_backups'):
            os.makedirs('model_backups')
        model.save(os.path.join('model_backups', 'epoch_' + str(e) + '.h5'))

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Train melody generation model.')
    parser.add_argument('--data', metavar = 'D', type = str, default = 'melody_sequences.p', help = 'Specify the pickle file with melody sequences for each song. [default: melody_sequences.p]')
    parser.add_argument('--epochs', metavar = 'e', type = int, default = 100, help = 'Epoch on which the network will be trained. [default: 100]')
    args = parser.parse_args()

    try:
    	#input is expected to be a pickling of a 2-dimensional python list so that raw_data[i][j] is the j-th eighth of the i-th song
        dataset = pickle.load(open(args.data, 'rb'))
    except:
        print('Specified file does not exists.')
        print('Type \'python melody-generation.py -h\' to get more information.')
        exit(0)

    if args.epochs > 0:
        EPOCHS = args.epochs
    else:
        print('Number of epochs must be greater than 0.')
        print('Type \'python melody-generation.py -h\' to get more information.')
        exit(0)

    for i, song in enumerate(dataset):
        for j, eighth in enumerate(song):
            # translate notes in each eighth in range [0, 88]
            translated = [(note - (MIDI_LOWEST - 1) if note != REST else note) for note in eighth]
            # one-hot encoded piano roll
            piano_roll = [(1 if v in translated else 0) for v in range(NOTE_CLASSES)]
            dataset[i][j] = piano_roll

    # TODO controllare se nel pickle degli accordi ci sono anche le pause, in caso contrario metterle.
    # legere anche pickle degli accordi e formare il vettore finale come svizzeri

    #Build and train model
    init_model()
    train_model()
