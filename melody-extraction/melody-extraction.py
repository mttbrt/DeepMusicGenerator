# lower note 24 - higher note 127

from music21 import converter, note, chord
from music21.duration import Duration
from music21.stream import Stream
from music21.instrument import UnpitchedPercussion
from math import sqrt, log, inf
from itertools import accumulate

import os, sys, glob, pickle, copy

REST = 0
NO_NOTE = 88
NOTE_CLASSES = NO_NOTE+1 # 88 notes of the piano and "no note"
PITCH_CLASSES = 12 #Using pitch classes that go from 0 (C) to 11 (B)
MIDI_LOWEST = 21 # Lowest midi note taken into consideration: A0
MIDI_HIGHEST = 108 # Highest midi note taken into consideration: C8

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
                filenames.sort()
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
        mid.remove(indesiderata, recurse=True)

        measures = mid.chordify(addTies=False).measures(0,-1,indicesNotNumbers=True)
        print('Done.\n')

        output_seq = []
        eighth_current_size = 0.0

        def append_note(nt):
            global eighth_current_size
            empty_space = 0.5 - eighth_current_size

            if nt.duration.quarterLength > empty_space:
                # creo una nota (o rest) da 0.5 e la aggiungo alla sequenza
                val = copy.deepcopy(nt)
                val.duration.quarterLength = empty_space
                output_seq.append(val)

                eighth_current_size = 0.0

                # la porzione rimanente di nota (o rest) viene messa nel buffer
                buffer = copy.deepcopy(nt)
                buffer.duration.quarterLength -= empty_space
            else:
                # aggiungo la nota (o rest) alla sequenza
                output_seq.append(nt)
                buffer = None

                eighth_current_size += nt.duration.quarterLength
                if eighth_current_size == 0.5:
                    eighth_current_size = 0.0

            return buffer

        # preprocessing
        for i, measure in enumerate(measures):
            for c in measure.notesAndRests:
                buff = append_note(c)
                while buff != None:
                    buff = append_note(buff)

        # ottengo blocchi di 0.5 uniformi (non pezzi di lunghezze diverse < 0.5)
        output_sequence = []
        eighth_durations = []
        eighth_notes = []

        for nt in output_seq:
            eighth_durations.append(nt.duration.quarterLength)
            # eighth_notes.append([REST] if nt.isRest else [n.pitch.midi for n in nt.notes if (n.pitch.midi >= MIDI_LOWEST and n.pitch.midi <= MIDI_HIGHEST)])
            eighth_notes.append([REST] if nt.isRest else [n.pitch.nameWithOctave for n in nt.notes])

            if sum(eighth_durations) == 0.5:
                # mappo ogni nota con il totale del tempo che suona all'interno dell'ottava
                notes_and_durations = {}
                for i, a in enumerate(eighth_notes):
                    for e in a:
                        if e in notes_and_durations:
                            notes_and_durations[e] += eighth_durations[i]
                        else:
                            notes_and_durations[e] = eighth_durations[i]

                # reset
                eighth_durations = []
                eighth_notes = []

                # costruisco l'ottava con le note che suonano almeno mezza ottava
                actual_eighth = [nd for nd in notes_and_durations if notes_and_durations[nd] >= 0.25]

                # rimuovo le pause se ci sono già altre note che suonano nell'ottava
                if len(actual_eighth) > 1:
                    actual_eighth = [x for x in actual_eighth if x != REST]

                output_sequence.append(actual_eighth)
                # print(output_sequence[-5:])

        output_sequences.append(output_sequence)

        # DEBUG: create and save midi
        s = Stream()
        for i, n in enumerate(output_sequence):
            if len(n) == 1:
                if n[0] == 0:
                    s.append(note.Rest(duration=Duration(0.5)))
                else:
                    s.append(note.Note(n[0], duration=Duration(0.5)))
            else:
                s.append(chord.Chord(n, duration=Duration(0.5)))
            # s.append(note.Note(n, duration=output_duration[i]))
        s.write('midi', fp='melodies/' + f.split('/')[-1])

    print(f'Pickling chords of {num_of_files-errors} out of {num_of_files} files...')
    pickle.dump( output_sequences, open("melody_sequences.p", "wb"))
