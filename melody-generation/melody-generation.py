from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
# from keras.optimizers import RMSprop, Adam

from music21 import converter, note, chord
from music21.duration import Duration
from music21.stream import Stream

from natsort import natsorted, ns
from random import shuffle
import numpy as np
import argparse, pickle, re, progressbar, os

REST = 0
MIDI_LOWEST = 21 # Lowest midi note taken into consideration: A0
MIDI_HIGHEST = 108 # Highest midi note taken into consideration: C8
MIDI_NOTES = MIDI_HIGHEST - MIDI_LOWEST + 1
NOTE_CLASSES = MIDI_NOTES + 1 # 88 notes of the piano and rest
CHORD_CLASSES = 24+1 # 12 major chords, 12 minor chords, and "no chord"
INPUT_SIZE = NOTE_CLASSES + CHORD_CLASSES + CHORD_CLASSES

EPOCHS = 0
START_EPOCH = 0
HIDDEN_UNITS = 12 # number of LSTM cells - formula: N_h = N_s / (alpha * (N_i + N_o))
BATCH_SIZE = 1 # how many sequences to process in parallel
NOTES_BLOCK = 1 # how many notes pass to the network every time
EMBEDDING_SIZE = INPUT_SIZE # size of embedding vector
ACTIVATION = 'sigmoid'
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'

chords_dataset = None
melody_dataset = None
input_vectors = []

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
        # model.add(Dropout(0.3))
        # model.add(Dense(MIDI_NOTES, name=f'Dense_{ACTIVATION}', activation=ACTIVATION))
        model.add(Dense(EMBEDDING_SIZE, name=f'Dense_{ACTIVATION}', activation=ACTIVATION))
        model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['acc', 'mae'])

    print(model.summary())

def train_model():
    print('Training...')

    shuffle(input_vectors)
    train_set, test_set = input_vectors[:int(len(input_vectors)*0.7)], input_vectors[int(len(input_vectors)*0.7):]
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
            xdata = np.array(xdata)

            X = np.array(xdata) #np.reshape(xdata, (xdata.shape[0], 1, INPUT_SIZE))
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

def compose_song(first_notes, num_predictions):
    last_backup = natsorted(os.listdir('model_backups'), alg=ns.IGNORECASE)[-1]
    model = load_model(os.path.join('model_backups', last_backup))

    melody = []
    tot_notes = first_notes
    for i in range(num_predictions):
        last_notes = np.array([tot_notes[-NOTES_BLOCK:]])
        prediction = model.predict(last_notes)

        notes = [list(np.where(pr == np.amax(pr))[0]) for pr in prediction[0][:89]][0][0]
        current_chord = 89 + [list(np.where(pr == np.amax(pr))[0]) for pr in prediction[0][89:114]][0][0]
        next_chord = 114 + [list(np.where(pr == np.amax(pr))[0]) for pr in prediction[0][114:]][0][0]

        predicted_note = [0] * INPUT_SIZE
        predicted_note[notes] = 1
        predicted_note[current_chord] = 1
        predicted_note[next_chord] = 1

        tot_notes.append(predicted_note)
        melody.append([notes, current_chord])

    pickle.dump( tot_notes, open( "predicted_melody.p", "wb" ) )

    build_midi(melody)

def build_midi(melody):
    melody[0] = (62, 90)

    # create and save midi
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # TODO: ora c'è una sola nota, creare accordi minori e maggiori
    CHORDS = [['C3'], ['C#3'], ['D3'], ['D#3'], ['E3'], ['F3'], ['F#3'], ['G3'], ['G#3'], ['A3'], ['A#3'], ['B3'], ['C3'], ['C#3'], ['D3'], ['D#3'], ['E3'], ['F3'], ['F#3'], ['G3'], ['G#3'], ['A3'], ['A#3'], ['B3']]
    s = Stream()
    for i, (n, c) in enumerate(melody):
        rest = False
        chrd = []

        n += MIDI_LOWEST-1
        if n == MIDI_LOWEST-1:
            rest = True
        else:
            note_ = NOTES[n%12]
            note_octave = int(n/12)
            chrd.append(note.Note(nameWithOctave=note_+str(note_octave)))

        c -= MIDI_NOTES
        if c == CHORD_CLASSES-1:
            rest = True
        else:
            chord_ = CHORDS[c%12]
            for nt in chord_:
                chrd.append(note.Note(nameWithOctave=nt))

        if len(chrd) < 1:
            if rest:
                s.append(note.Rest(duration=Duration(0.5)))
            else:
                print('Error: no chord and no rest')
                sys.exit(0)
        else:
            s.append(chord.Chord(chrd, duration=Duration(0.5)))

    s.write('midi', fp='melodies/generated.mid')

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Train melody generation model.')
    parser.add_argument('--data', metavar = 'D', type = str, default = 'melody_sequences.p', help = 'Specify the pickle file with melody sequences for each song. [default: melody_sequences.p]')
    parser.add_argument('--epochs', metavar = 'e', type = int, default = 100, help = 'Epoch on which the network will be trained. [default: 100]')
    args = parser.parse_args()

    try:
    	#input is expected to be a pickling of a 2-dimensional python list so that raw_data[i][j] is the j-th eighth of the i-th song
        melody_dataset = pickle.load(open(args.data, 'rb'))
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

    for i, song in enumerate(melody_dataset):
        for j, eighth in enumerate(song):
            # translate notes in each eighth in range [0, 88]
            translated = [(note - (MIDI_LOWEST - 1) if note != REST else note) for note in eighth]
            # one-hot encode melody eighth ones
            melody_dataset[i][j] = [(1 if v in translated else 0) for v in range(NOTE_CLASSES)]

    # one-hot encode predicted chords
    chords_dataset = pickle.load(open('chord_sequences.p', 'rb'))
    for i, song in enumerate(chords_dataset):
        for j, ch in enumerate(song):
            chords_dataset[i][j] = [(1 if v == ch else 0) for v in range(CHORD_CLASSES)]

    # create vectors with melody, current chord and next chord.
    for i, song in enumerate(melody_dataset):
        song_vectors = []
        for j, eighth in enumerate(song):
            measure = int(j/8)
            if len(chords_dataset[i]) > measure + 1:
                eighth.extend(chords_dataset[i][measure])
                eighth.extend(chords_dataset[i][measure + 1])
                song_vectors.append(eighth)
        input_vectors.append(song_vectors)

    # Build and train model
    init_model()
    train_model()

    # 0-88 piano roll, 89-113 current chord, 113-25 next chord
    # intro_1, intro_2, intro_3, intro_4, intro_5, intro_6, intro_7, intro_8 = [0] * INPUT_SIZE, [0] * INPUT_SIZE, [0] * INPUT_SIZE, [0] * INPUT_SIZE, [0] * INPUT_SIZE, [0] * INPUT_SIZE, [0] * INPUT_SIZE, [0] * INPUT_SIZE
    # intro_1[62] = 1; intro_1[89] = 1; intro_1[118] = 1
    # intro_2[64] = 1; intro_2[89] = 1; intro_2[118] = 1
    # intro_3[66] = 1; intro_3[89] = 1; intro_3[118] = 1
    # intro_4[62] = 1; intro_4[89] = 1; intro_4[118] = 1
    # intro_5[62] = 1; intro_5[89] = 1; intro_5[118] = 1
    # intro_6[64] = 1; intro_6[89] = 1; intro_6[118] = 1
    # intro_7[66] = 1; intro_7[89] = 1; intro_7[118] = 1
    # intro_8[62] = 1; intro_8[89] = 1; intro_8[118] = 1
    # compose_song([intro_1, intro_2, intro_3, intro_4, intro_5, intro_6, intro_7, intro_8], 16)
