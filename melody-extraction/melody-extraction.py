from music21 import converter, note, chord
from music21.duration import Duration
from music21.stream import Stream
from music21.instrument import UnpitchedPercussion
from math import sqrt, log, inf
from itertools import accumulate

import os, sys, glob, pickle

NO_NOTE = 88
NOTE_CLASSES = NO_NOTE+1 # 88 notes of the piano and "no note"
PITCH_CLASSES = 12 #Using pitch classes that go from 0 (C) to 11 (B)
MIDI_LOWEST = 21 # Lowest midi note taken into consideration: A0
MIDI_HIGHEST = 108 # Highest midi note taken into consideration: C8

FUNDAMENTALS = ['C', 'C#', 'D', 'E-', 'E', 'F', 'F#', 'G', 'A-', 'A', 'B-', 'B']
MAJOR_PRIORITY = [7,0,21,5,14,16,12,19,23,17,1,9,4,11,2,10,22,8,3,6,13,20,15,18]
MINOR_PRIORITY = [21,7,0,5,14,4,16,12,19,23,17,1,9,11,2,10,22,8,3,6,13,20,15,18]

def one_hot_encode_note(note_midi_id):
    if note_midi_id == None:
        id = NO_NOTE
    else:
        id = note_midi_id - MIDI_LOWEST
    if id >= NOTE_CLASSES:
        raise ValueError(f'Note id must be None or between {MIDI_LOWEST} and {MIDI_HIGHEST}. Found {note_midi_id}')
    return [1 if i == id else 0 for i in range(NOTE_CLASSES)]

#Returns whether a histogram is empty (a.k.a. uninitialized)
def histogram_is_empty(histogram):
    for bin in histogram:
        if bin > 0:
            return False
    return True

#Returns a normalized version of the input histogram
def normalize_histogram(histogram):
    total = 0
    for entry in histogram:
        total += entry
    return [histogram[i]/total for i in range(len(histogram))]

def permute_list(list, permutation_map):
    if len(list) != len(permutation_map):
        return list
    return [list[i] for i in permutation_map]

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
    return list(enumerate(chord_histograms))

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
    best = chord_histograms[0][0]
    min_distance = bhattacharyya_distance(chord_histograms[0][1],target_histogram) #chord_histograms contains (id,histogram) tuples
    for id, histogram in chord_histograms:
        distance = bhattacharyya_distance(histogram, target_histogram)
        if distance < min_distance:
            best = id
            min_distance = distance
    return best

if __name__ == "__main__":
    inputs = sys.argv[1:]
    if len(inputs) == 0:
        print('\nUsage: python3 chord-extraction.py <file1> [<file2> ...]\n')
        exit(0)

    output_sequences = []

    # make a list with all midi files in directories (if any) passed as input
    files = []
    for input in inputs:
        if os.path.isfile(input) and (input.lower().endswith('.mid') or input.lower().endswith('.midi')):
            files.append(input)
        elif os.path.isdir(input):
            for (dirpath, dirnames, filenames) in os.walk(input):
                for file in filenames:
                    if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                        files.append(os.path.join(dirpath, file))
        else:
            print(f'Warning: {input} is neither a midi file nor a directory.')

    num_of_files = len(files)

    if num_of_files == 0:
        print('No midi file found.')
        sys.exit(0)

    errors = 0
    for i,f in enumerate(files):
        print(f'\n({i+1}/{num_of_files}):{f}')
        print('Opening file...')
        try:
            mid = converter.parse(f)
        except:
            errors += 1
            continue
        key = mid.analyze('key')
        print(f'Song is in {key}')
        print('Preprocessing song... (this may take a while)')
        indesiderata = [element for element in mid.recurse(classFilter=('Instrument','MetronomeMark'))]
        #instruments = [instrument for instrument in mid.getInstruments(recurse=True)]
        mid.remove(indesiderata, recurse=True)

        measures = mid.chordify(addTies=False).measures(0,-1,indicesNotNumbers=True)
        print('Done.\n')

        #measures.write(fp='debug.mid',fmt='mid') #debug

        chord_histograms = make_chord_histograms()
        output_sequence = []

        for i, measure in enumerate(measures):
            print(f'Measure {i+1}')

            for c in measure.notes:
                eighths = int(c.duration.quarterLength / 0.5)
                # repeat the occurrence of the chord as many times as eighths it plays
                for eighth in range(eighths):
                    # Version with 3 highest pitches
                    # # take three highest pitches
                    # highest_pitches = c.pitches[-3:] if len(c.pitches) > 3 else c.pitches
                    # # compute closest chord given the three highest pitches
                    # pch = [.0 for _ in range(PITCH_CLASSES)]
                    # for p in highest_pitches:
                    #     pch[p.pitchClass] = 0.5
                    #
                    # if histogram_is_empty(pch):
                    #     chord = NO_CHORD
                    # else:
                    #     pch = normalize_histogram(pch)
                    #     chord_priority = MAJOR_PRIORITY if key.mode == 'major' else MINOR_PRIORITY
                    #     chord = find_closest_histogram(permute_list(chord_histograms, chord_priority), pch)
                    #
                    # mjr_chord = chord - PITCH_CLASSES if chord >= PITCH_CLASSES else chord

                    # Version with highest pitch
                    mjr_chord = c.pitches[-1].nameWithOctave if len(c.pitches) > 0 else NO_NOTE
                    output_sequence.append(mjr_chord)

            print('|', end=' ')
            print(' - '.join(x for x in output_sequence[-8:]), end=' |\n\n')

        output_sequences.append(output_sequence)

        # DEBUG: create and save midi
        s = Stream()
        for n in output_sequence:
            s.append(note.Note(n, duration=Duration(0.5)))
        s.write('midi', fp=f.split('/')[-1])

    print(f'Pickling chords of {num_of_files-errors} out of {num_of_files} files...')
    pickle.dump( output_sequences, open( "melody_sequences.p", "wb" ) )
