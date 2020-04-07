#from keras import sequential
import numpy as np

import sys, pickle

CHORD_CLASSES = 24+1 #12 major chords, 12 minor chords, and "no chord"

#Returns a one-hot encoding of a chord id
def id_to_vec(chord_id):
    if chord_id == None:
        return np.array([1 if i == CHORD_CLASSES-1 else 0 for i in range(CHORD_CLASSES)])
    return np.array([1 if i == chord_id else 0 for i in range(CHORD_CLASSES)])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python chord-generation.py <training data>')
        exit(0)

    input_file = sys.argv[1]
	#input is expected to be a pickling of a 2-dimensional python list so that raw_data[i][j] is the j-th chord of the i-th song
    raw_data = pickle.load(open(input_file, 'rb'))

    print(id_to_vec(raw_data[0][0])) #debug

    ## TODO: everything lmao ##