from music21 import converter, note, chord
from music21.stream import Stream
from music21.instrument import UnpitchedPercussion
from math import sqrt, log, inf
from itertools import accumulate

import os, sys, glob, pickle

NOTE_CLASSES = 88+1 # 88 notes of the piano and "no note"
MIDI_LOWEST = 21 # Lowest midi note taken into consideration: A0
MIDI_HIGHEST = 108 # Highest midi note taken into consideration: C8
NO_NOTE = 88

def one_hot_encode_note(note_midi_id):
    if note_midi_id == None:
        id = NO_NOTE
    else:
        id = note_midi_id - MIDI_LOWEST
    if id >= NOTE_CLASSES:
        raise ValueError(f'Note id must be None or between {MIDI_LOWEST} and {MIDI_HIGHEST}. Found {note_midi_id}')
    return [1 if i == id else 0 for i in range(NOTE_CLASSES)]

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


        output_sequence = []

        for i, measure in enumerate(measures):
            pch = [.0 for _ in range(PITCH_CLASSES)]
            for c in measure.notes:
                for p in c.pitches:
                    pch[p.pitchClass] += c.duration.quarterLength

            if histogram_is_empty(pch):
                chord = NO_CHORD
            else:
                pch = normalize_histogram(pch)
                chord_priority = MAJOR_PRIORITY if key.mode == 'major' else MINOR_PRIORITY
                #chord = find_closest_histogram(chord_histograms, pch)
                chord = find_closest_histogram(permute_list(chord_histograms, chord_priority), pch)

            output_sequence.append(chord)
            print(f'Measure {i+1}: {id_to_chord_name(chord)}')

        output_sequences.append(output_sequence)

    print(f'Pickling chords of {num_of_files-errors} out of {num_of_files} files...')
    pickle.dump( output_sequences, open( "melody_sequences.p", "wb" ) )
