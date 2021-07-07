import json
from glob import glob
import os

import result.rttm

PUNCTUATIONS = ' 。，？！、'

def find_label(start, stop, rttms):
    intersections = []
    for rttm in rttms:
        parts = result.rttm.intersection(start, stop, 'l', rttm.start, rttm.stop, 'r')
        if parts is None:
            continue
        else:
            for (istart, istop, tag) in parts:
                if tag == 'intersection':
                    intersections.append((istop - istart, rttm.speaker))
    intersections.sort(key=lambda x: x[0], reverse=True)
    if intersections:
        return intersections[0][1]
    else:
        return None


for path in glob('result/vad_xf/*.transcript.json'):
    print('processing', path)
    with open(path) as f:
        sentences = json.load(f)

    # break sentences into smaller chunks
    new_sentences = []
    buf = []
    for sent in sentences:
        for word in sent['words']:
            w_start = word['start_ms_audio_time']
            w_stop = word['stop_ms_audio_time']
            if not buf:
                buf.append(word)
            else:
                last_stop = buf[-1]['stop_ms_audio_time']
                if word['text'].strip() in PUNCTUATIONS:
                    # start anew
                    buf.append(word)
                    new_sentences.append(buf)
                    buf = []
                elif last_stop + 3000 < w_start:
                    # start anew
                    new_sentences.append(buf)
                    buf = [word]
                else:
                    buf.append(word)
    if buf:
        new_sentences.append(buf)


    # lookup speaker label for each sentence
    # load truth
    basename = os.path.basename(path)
    file_id = basename.split('.')[0]
    rttm_path = os.path.join('result/label_rttm_raw', f'{file_id}.rttm')
    rttm = result.rttm.load_rttm(rttm_path)

    # tag each word with label
    new_sentences_words_tagged = []
    for sentence in new_sentences:
        new_words = []
        for word in sentence:
            if word['text'] in PUNCTUATIONS:
                # punctuation has the same start and stop time, ignore it
                continue
            start = word['start_ms_audio_time'] / 1000
            stop = word['stop_ms_audio_time'] / 1000
            label = find_label(start, stop, rttm)
            # note label could be None
            new_words.append((word, label))
        if new_words:
            new_sentences_words_tagged.append(new_words)

    # break words at speaker turn
    new_sentences_sentence_tagged = []

    buf = []
    buf_speaker = None

    def set_buf_speaker(spk):
        global buf_speaker
        if spk is not None:
            buf_speaker = spk

    for sent in new_sentences_words_tagged:
        for (word, this_speaker) in sent:
            if not buf:
                buf.append(word)
                set_buf_speaker(this_speaker)
            else:
                if this_speaker is None:
                    buf.append(word)
                    set_buf_speaker(this_speaker)
                elif buf_speaker is None:
                    buf.append(word)
                    set_buf_speaker(this_speaker)
                elif this_speaker == buf_speaker:
                    buf.append(word)
                    set_buf_speaker(this_speaker)
                elif this_speaker != buf_speaker:
                    assert buf_speaker is not None
                    new_sentences_sentence_tagged.append((buf, buf_speaker))
                    # start a new sentence
                    buf = [word]
                    buf_speaker = None
                    set_buf_speaker(this_speaker)
                else:
                    assert False

        # at the end of the sentence, we still clear the buffer
        if buf:
            if buf_speaker is None:
                ends = [
                    (w['start_ms_audio_time'], w['stop_ms_audio_time'])
                    for w in buf
                ]
                start = ends[0][0] / 1000
                stop = ends[-1][1] / 1000
                text = ''.join(w['text'] for w in buf)
                print(f'buf dropped due to no speaker: {stop - start:.4f} {start:.4f} {stop:.4f} {text}')
            else:
                new_sentences_sentence_tagged.append((buf, buf_speaker))

        buf = []
        buf_speaker = None

    # write out rttm and vad
    rttms = []
    vads = []
    for (sent, speaker) in new_sentences_sentence_tagged:
        start = sent[0]['start_ms_audio_time'] / 1000
        stop = sent[-1]['stop_ms_audio_time'] / 1000
        rttms.append(result.rttm.Rttm(file_id, start, stop-start, speaker))
        vads.append((start, stop))
    rttm_out_path = os.path.join('result/label_rttm_xf', f'{file_id}.rttm')
    result.rttm.write_rttm(rttms, rttm_out_path)
    adjacent_merged = os.path.join('result/label_rttm_xf_adjacent_merged', f'{file_id}.rttm')
    result.rttm.merge_rttm_by_adjacency(rttm_out_path, adjacent_merged)
    vad_out_path = os.path.join('result/vad_xf', f'{file_id}.lab')
    result.rttm.write_lab(vads, vad_out_path)