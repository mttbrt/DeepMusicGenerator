from music21 import converter, note, chord
from music21.stream import Stream
from music21.instrument import UnpitchedPercussion
from math import sqrt, log, inf
from itertools import accumulate 

import sys, glob

PITCH_CLASSES = 12 #Using pitch classes that go from 0 (C) to 11 (B)
FUNDAMENTALS = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
MODES = ['major', 'minor']

#Utility, given a chord id returns its common name
def id_to_chord_name(chord_id):
    if chord_id > 23:
        return 'Unknown chord'
    m, f = divmod(chord_id, PITCH_CLASSES)
    return f'{FUNDAMENTALS[f]} {MODES[m]}'

#Returns a normalized version of the input histogram
def normalize_histogram(histogram):
    total = 0
    for entry in histogram:
        total += entry
    return [histogram[i]/total for i in range(len(histogram))]

#Creates and returns the reference histograms for the 24 basic major and minor chords
def make_chord_histograms():
    chord_histograms = []
    for chord_id in range(24): #12 major and 12 minor chords, no seventh chords yet
        mode, fundamental = divmod(chord_id, PITCH_CLASSES)
        if mode == 0: #major chord
            chord_pitches = [fundamental, (fundamental + 4) % PITCH_CLASSES, (fundamental + 7) % PITCH_CLASSES] #major third
        else: #minor chord
            chord_pitches = [fundamental, (fundamental + 3) % PITCH_CLASSES, (fundamental + 7) % PITCH_CLASSES] #minor third
        histogram = [1 if n in chord_pitches else 0 for n in range(PITCH_CLASSES)]
        chord_histograms.append(normalize_histogram(histogram))
    return chord_histograms

#Returns the bhattacharyya distance between two histograms or 1000 if the two histograms do not overlap
#The two histograms involved must have the same number of bins
def bhattacharyya_distance(h1, h2):
    if len(h1) != len(h2):
        raise ValueError('Histograms h1 and h2 have different domain sizes')
    domain = len(h1)
    bc = 0
    for i in range(domain):
        bc += sqrt(h1[i]*h2[i])
    return -log(bc) if bc != 0 else inf

#Given a collection of reference histograms and a target histogram, returns whichever reference histogram is closest to the target one (according to bhattacharyya distance).
#All the histograms involved must have the same number of bins
def find_closest_histogram(chord_histograms, target_histogram):
    #simply an argmax
    best = 0
    min_distance = bhattacharyya_distance(chord_histograms[0],target_histogram)
    for i, histogram in enumerate(chord_histograms):
        distance = bhattacharyya_distance(histogram, target_histogram)
        if distance < min_distance:
            best = i
            min_distance = distance
    return best

def get_chords_progression_from_key(key, scale_type = MODES[0]):
    # Major scale
    if scale_type == MODES[0]:
        pattern = [2, 2, 1, 2, 2, 2, 1]
        progression = [MODES[0], MODES[1], MODES[1], MODES[0], MODES[0], MODES[1], 'dim']
    # Minor scale
    elif scale_type == MODES[1]:
        pattern = [2, 1, 2, 2, 1, 2, 2]
        progression = [MODES[1], 'dim', MODES[0], MODES[1], MODES[1], MODES[0], MODES[0]]
    else:
        print('Scale type not supported.')
        exit(0)

    indices = [(key + i) % 12 for i in accumulate(pattern)]
    indices.insert(0, indices.pop()) # rotate: last element is moved in front of the list
    chords = []
    for i, fundamental in enumerate(indices):
        # Major chord
        if progression[i] == MODES[0]:
            chords.append([fundamental, (fundamental + 4) % PITCH_CLASSES, (fundamental + 7) % PITCH_CLASSES])
        # Minor chord
        elif progression[i] == MODES[1]:
            chords.append([fundamental, (fundamental + 3) % PITCH_CLASSES, (fundamental + 7) % PITCH_CLASSES])
        # Diminished chord
        else:
            chords.append([fundamental, (fundamental + 3) % PITCH_CLASSES, (fundamental + 6) % PITCH_CLASSES])

    print(f'Pitch of notes in {FUNDAMENTALS[key]} {scale_type} scale: {indices}')
    print(f'\nChords in {FUNDAMENTALS[key]} {scale_type} scale:')
    for i, chord in enumerate(chords):
        print(f'Chord {FUNDAMENTALS[chord[0]]} {progression[i]}: [{FUNDAMENTALS[chord[0]]}, {FUNDAMENTALS[chord[1]]}, {FUNDAMENTALS[chord[2]]}]')

    return chords

if __name__ == "__main__":
    inputs = sys.argv[1:]
    if len(inputs) == 0:
        print('\nUsage: python3 chord-extraction.py <file1> [<file2> ...]\n')
        exit(0)

    chord_histograms = make_chord_histograms()

    #input_dir = join('raw_midi')
    #inputs = glob.glob(join(input_dir,'Lady Gaga - Edge of Glory.mid'))

    for f in inputs:
        print(f'\n=={f}==')
        print('Opening file...')
        mid = converter.parse(f)
        print('Chordifying and splitting song... (this may take a while)')
        measures = mid.chordify(addTies=False).measures(0,-1,indicesNotNumbers=True)
        print('Done.\n')

        measures.write(fp='debug.mid',fmt='mid') #debug

        for i, measure in enumerate(measures):
            pch = [.0 for i in range(PITCH_CLASSES)]
            if len(measure.notes) > 0: # if a measure has notes
                for c in measure.notes:
                    for p in c.pitches:
                        pch[p.pitchClass] += c.duration.quarterLength

                pch = normalize_histogram(pch)

                h = find_closest_histogram(chord_histograms, pch)
                chord = id_to_chord_name(h)
            else: # if a measure is empty
                chord = 'No chord'

            print(f'Measure {i+1}: {chord}')

#        i = 0
#        measure = mid.measure(i).flat
#        while len(measure) == 0:    #skip leading blank measures
#            i += 1
#            measure = mid.measure(i).flat
#
#        while len(measure) != 0:
#            pch = [.0 for i in range(PITCH_CLASSES)]
#            if len(measure.notes) > 0: # skip measures with all rests
#                for n in measure.notes:
#                    if isinstance(n, note.Note): #if we encounter a single note
#                        pch[n.pitch.pitchClass] += n.duration.quarterLength
#                    elif isinstance(n, chord.Chord): #if we encounter a chord
#                        for p in n.pitches:
#                            pch[p.pitchClass] += n.duration.quarterLength
#
#                pch = normalize_histogram(pch)
#
#                h = find_closest_histogram(chord_histograms, pch)
#                print(f'Measure {i+1}: {id_to_chord_name(h)}')
#
#            i += 1
#            measure = mid.measure(i).flat
