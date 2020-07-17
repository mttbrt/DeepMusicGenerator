from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np

import re, os, sys, argparse, pickle, progressbar

CHORD_CLASSES = 24+1 # 12 major chords, 12 minor chords, and "no chord"

# Model constants
EMBEDDING = False
EMBEDDING_SIZE = 6 if EMBEDDING else 1 # size of embedding vector
CHORDS_BLOCK = 4 # how many chords pass to the network every time
BATCH_SIZE = 1 # how many sequences to process in parallel (> 1 for polyphonic generation)
HIDDEN_UNITS = 12 # number of LSTM cells - formula: N_h = N_s / (alpha * (N_i + N_o))
LEARNING_RATE = 0.001
ACTIVATION = 'softmax'
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'

# Training constants
START_EPOCH = 0
EPOCHS = 100

dataset = None
model = None
transition_matrix = {}

# for human sorting
def natural_keys(text):
    return [ (int(c) if c.isdigit() else c) for c in re.split(r'(\d+)', text) ]

# If load_last is True than load the last model trained in 'model_backups'
def init_model(load_last=False):
    global model
    if load_last:
        last_backup = sorted(os.listdir('model_backups'), key=natural_keys)[-1]
        model = load_model(os.path.join('model_backups', last_backup))

        global START_EPOCH
        START_EPOCH = int(re.search('_(.+?)\.', last_backup).group(1))
    else:
        model = Sequential(name = 'lstm-chord-generator')
        if EMBEDDING:
            model.add(Embedding(input_dim = CHORD_CLASSES, output_dim = EMBEDDING_SIZE, input_length = CHORDS_BLOCK, batch_input_shape = (BATCH_SIZE, CHORDS_BLOCK), name='Embedding'))
        #model.add(LSTM(HIDDEN_UNITS, batch_input_shape = (BATCH_SIZE, CHORDS_BLOCK, EMBEDDING_SIZE), return_sequences = False, stateful = True, name='LSTM'))
        model.add(LSTM(HIDDEN_UNITS,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True,
            batch_input_shape = (BATCH_SIZE, CHORDS_BLOCK, EMBEDDING_SIZE),
            return_sequences = False,
            stateful = True,
            name='LSTM'))
        model.add(Dropout(0.3))
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

# Training with one step ahead cross validation
def train_model_cv():
    print('Training...')

    total_chord_blocks_in_dataset = sum(len(s) - CHORDS_BLOCK - 1 for s in dataset)

    # training_set_size: number of chord blocks to train the NN on
    for training_set_size in range(START_EPOCH + 1, total_chord_blocks_in_dataset):
        print(f'\n{training_set_size}/{total_chord_blocks_in_dataset}')
        # Preprocessing: get chord blocks in a sequential way keeping the dataset matrix form
        xdata = []
        ydata = []
        current_training_set_size = len(xdata) # number of current chord blocks in the training set
        song_iter = 0 # iterate through songs in dataset

        # I get 1 more chord block to test the model
        while current_training_set_size < training_set_size + 1:
            song = dataset[song_iter]
            max_chord_blocks_in_song = len(song) - CHORDS_BLOCK - 1

            for i in range(min(max_chord_blocks_in_song, (training_set_size + 1) - current_training_set_size)):
                xdata.append(song[i:i+CHORDS_BLOCK])
                ydata.append(song[i+CHORDS_BLOCK])

            current_training_set_size += len(xdata)
            song_iter += 1

        xdata = np.array(xdata[:-1]) # remove last chord block for training (it is for testing)
        ydata = np.array(ydata[:-1]) # remove last chord block for training (it is for testing)

        X = xdata if EMBEDDING else np.reshape(xdata, (xdata.shape[0], CHORDS_BLOCK, 1))
        Y = utils.to_categorical(ydata, num_classes = CHORD_CLASSES) # one-hot encode chords
        stats = model.fit(X, Y, batch_size = BATCH_SIZE, shuffle = False, verbose = False)
        model.reset_states()

        acc = stats.history['acc'][0]
        loss = stats.history['loss'][0]
        mae = stats.history['mae'][0]
        print(f'TRAINING - ACCURACY: {acc} | LOSS: {loss} | MAE: {mae}')

        # Test and backup model every 10 iterations
        txdata = np.array([xdata[-1]]) # the last chord block was for testing
        tydata = np.array([ydata[-1]]) # the last chord block was for testing
        X = txdata if EMBEDDING else np.reshape(txdata, (txdata.shape[0], CHORDS_BLOCK, 1))
        Y = utils.to_categorical(tydata, num_classes = CHORD_CLASSES) # one-hot encode chords
        tstats = model.evaluate(X, Y, batch_size = BATCH_SIZE, verbose = False)
        print(f'TESTING - ACCURACY: {tstats[0]} | LOSS: {tstats[1]} | MAE: {tstats[2]}')

        if training_set_size % 10 == 0:
            if not os.path.exists('model_backups'):
                os.makedirs('model_backups')
            model.save(os.path.join('model_backups', 'iter_' + str(training_set_size) + '.h5'))

    # save last configuration
    model.save(os.path.join('model_backups', 'iter_' + str(training_set_size) + '.h5'))

def train_model():
    print('Training...')

    training_loss = []
    training_acc = []
    testing_loss = []
    testing_acc = []

    # init plots
    plt.ion()
    fig = plt.figure()
    fig.suptitle(f'Metrics (epochs: {EPOCHS}, block size: {CHORDS_BLOCK})')
    fig.canvas.set_window_title('Metrics')
    # loss plot
    ax_train_loss = fig.add_subplot(2, 1, 1)
    ax_test_loss = fig.add_subplot(2, 1, 1)
    plt.ylabel('loss')
    train_labels_set = False
    # accuracy plot
    ax_train_acc = fig.add_subplot(2, 1, 2)
    ax_test_acc = fig.add_subplot(2, 1, 2)
    plt.ylabel('accuracy')
    test_labels_set = False

    train_set, test_set = split_dataset(0.7)
    for e in range(START_EPOCH + 1, EPOCHS + 1):
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
            Y = utils.to_categorical(ydata, num_classes = CHORD_CLASSES) # one-hot encode chords
            stats = model.fit(X, Y, batch_size = BATCH_SIZE, shuffle = False, verbose = False)
            model.reset_states()
            prog_bar.update(j+1)

            tot_epoch_loss += stats.history['loss'][0]
            tot_epoch_acc += stats.history['acc'][0]
            tot_epoch_mae += stats.history['mae'][0]

        training_loss.append(tot_epoch_loss/len(train_set))
        training_acc.append(tot_epoch_acc/len(train_set))

        print(f'\nACCURACY: {tot_epoch_acc/len(train_set)} | LOSS: {tot_epoch_loss/len(train_set)} | MAE: {tot_epoch_mae/len(train_set)}')

        # dinamically update plots
        if train_labels_set:
            ax_train_loss.plot(range(0, len(training_loss)), training_loss, color='#eb3434', marker='o')
            ax_train_acc.plot(range(0, len(training_acc)), training_acc, color='#3434eb', marker='o')
        else:
            ax_train_loss.plot(range(0, len(training_loss)), training_loss, color='#eb3434', marker='o', label='train')
            ax_train_acc.plot(range(0, len(training_acc)), training_acc, color='#3434eb', marker='o', label='train')
            train_labels_set = True
        ax_train_loss.legend()
        ax_train_acc.legend()

        # Test and backup model every epoch
        t_loss, t_acc = test_model(test_set)
        testing_loss.append(t_loss)
        testing_acc.append(t_acc)

        if test_labels_set:
            ax_test_loss.plot(range(0, len(testing_loss)), testing_loss, color='#ff7070', marker='o')
            ax_test_acc.plot(range(0, len(testing_acc)), testing_acc, color='#7070ff', marker='o')
        else:
            ax_test_loss.plot(range(0, len(testing_loss)), testing_loss, color='#ffb0b0', marker='o', label='test')
            ax_test_acc.plot(range(0, len(testing_acc)), testing_acc, color='#b0b0ff', marker='o', label='test')
            test_labels_set = True
        ax_test_loss.legend()
        ax_test_acc.legend()

        plt.pause(2)
        plt.draw()

        if not os.path.exists('model_backups'):
            os.makedirs('model_backups')
        model.save(os.path.join('model_backups', 'epoch_' + str(e) + '.h5'))

    plt.show()
    plt.savefig(f'metrics_{EPOCHS}_epochs_{CHORDS_BLOCK}_blocksize.png')

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
        Y = utils.to_categorical(ydata, num_classes = CHORD_CLASSES) # one-hot encode chords
        stats = model.evaluate(X, Y, batch_size = BATCH_SIZE, verbose = False)
        model.reset_states()
        prog_bar.update(j+1)

        tot_test_loss += stats[0]
        tot_test_acc += stats[1]
        tot_test_mae += stats[2]

    t_loss = tot_test_loss/len(test_set)
    t_acc = tot_test_acc/len(test_set)
    print(f'\nACCURACY: {t_acc} | LOSS: {t_loss} | MAE: {tot_test_mae/len(test_set)}')

    return t_loss, t_acc

def performance_eval(test_sequences):
    compute_transition_matrix()
    last_backup = sorted(os.listdir('model_backups'), key=natural_keys)[-1]
    model = load_model(os.path.join('model_backups', last_backup))

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

def compose_song(first_chords, num_predictions):
    last_backup = sorted(os.listdir('model_backups'), key=natural_keys)[-1]
    model = load_model(os.path.join('model_backups', last_backup))

    tot_chords = first_chords
    for i in range(num_predictions):
        last_chords = np.array([tot_chords[-CHORDS_BLOCK:]])
        data = last_chords if EMBEDDING else np.reshape(last_chords, (last_chords.shape[0], CHORDS_BLOCK, 1))
        prediction = model.predict(data)
        predicted_chord = [list(np.where(pr == np.amax(pr))[0]) for pr in prediction][0][0]
        tot_chords.append(predicted_chord)

    pickle.dump( tot_chords, open( "predicted_chords.p", "wb" ) )

    CHORDS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B', 'Cm', 'C#m', 'Dm', 'Ebm', 'Em', 'Fm', 'F#m', 'Gm', 'Abm', 'Am', 'Bbm', 'Bm', 'No Chord']
    for chord in tot_chords:
        print(CHORDS[chord], end = ' ')
    print()

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
    parser.add_argument('--data', metavar = 'D', type = str, default = 'chord_sequences.p', help = 'Specify the pickle file with chords sequences for each song. [default: chord_sequences.p]')
    parser.add_argument('--embedding', metavar = 'E', type = int, default = 0, help = 'Adds an embedding layer with the specified dimension. [default: 0 (no embedding)]')
    parser.add_argument('--block', metavar = 'B', type = int, default = 4, help = 'Chord sequence size. [default: 4]')
    parser.add_argument('--epochs', metavar = 'e', type = int, default = 100, help = 'Epoch on which the network will be trained. [default: 100]')
    parser.add_argument('--optimizer', metavar = 'o', type = str, default = 'adam', help = 'Keras compiler\'s optimizer. [default: adam]')
    args = parser.parse_args()

    try:
    	# input is expected to be a pickling of a 2-dimensional python list so that raw_data[i][j] is the j-th chord of the i-th song
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

    OPTIMIZER = args.optimizer

    # Build and train model
    init_model()
    # train_model_cv()
    train_model()

    # Model evaluation
    # performance_eval([[21, 5, 0, 16], [0, 16, 0, 16], [5, 7, 0, 21], [7, 2, 4, 0], [16, 5, 21, 7]])
    compose_song([7, 2, 21, 0, 7, 2, 21, 0], 16)
