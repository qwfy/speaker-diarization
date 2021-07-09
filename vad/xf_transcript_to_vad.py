import json
from glob import glob
import os
import numpy as np
import copy
import itertools

import result.rttm

PUNCTUATIONS = ' 。，？！、'


def find_labels(start, stop, rttms):
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
        return intersections
    else:
        return None


class Visitor:
    def __init__(self, sentences_with_words_tagged):
        self._xs = sentences_with_words_tagged
        self._visiting = 0
        self.words_dropped = set()

    def nexts(self):
        if self._visiting + 1 > len(self._xs):
            return None
        else:
            visiting = self._visiting
            self._visiting += 1
            sent = self._xs[visiting]
            all_tags = set()
            for word, tags in sent:
                if tags is None:
                    self.words_dropped.add(word['id'])
                else:
                    for tag in tags:
                        all_tags.add(tag[1])

            # no need to consider gap between the words,
            # since the input arg
            sents_tagged = []
            for tag in all_tags:
                words = []
                for w, tags in sent:
                    if tags:
                        for _, tag1 in tags:
                            if tag1 == tag:
                                words.append(w)
                                break
                if words:
                    sents_tagged.append((words, tag))
            return sents_tagged



for path in glob('result/vad_xf/*.transcript.json'):
    print('processing', path)
    with open(path) as f:
        sentences = json.load(f)

    # break sentences into smaller chunks
    # i.e. split on punctuations and large gaps
    word_id = 0
    new_sentences = []
    buf = []
    for sent in sentences:
        for word in sent['words']:
            word['id'] = word_id
            word_id += 1
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
                elif last_stop + 200 < w_start:
                    # start anew
                    new_sentences.append(buf)
                    buf = [word]
                else:
                    buf.append(word)
    if buf:
        new_sentences.append(buf)

    # sanity check
    old_len = np.sum([len(x['words']) for x in sentences])
    new_len = np.sum([len(x) for x in new_sentences])
    print(old_len, new_len)
    assert old_len == new_len

    # make a copy for vad labelling
    # (for at prod time, the only thing we know are the ends of the sentence,
    # the label is unknown)
    vad_sentences = copy.deepcopy(new_sentences)


    # lookup speaker label for each sentence
    # load truth
    basename = os.path.basename(path)
    file_id = basename.split('.')[0]
    rttm_path = os.path.join('result/label_rttm_raw', f'{file_id}.rttm')
    rttm = result.rttm.load_rttm(rttm_path)

    # tag each word with one or more labels
    new_sentences_words_tagged = []
    for sentence in new_sentences:
        new_words = []
        for word in sentence:
            if word['text'] in PUNCTUATIONS:
                # punctuation has the same start and stop time, ignore it
                continue
            start = word['start_ms_audio_time'] / 1000
            stop = word['stop_ms_audio_time'] / 1000
            labels = find_labels(start, stop, rttm)
            # note label could be None
            new_words.append((word, labels))
        if new_words:
            new_sentences_words_tagged.append(new_words)

    # label sentences with labels
    visitor = Visitor(new_sentences_words_tagged)
    new_sentences_sentence_tagged = []
    while True:
        nexts = visitor.nexts()
        if nexts is None:
            break
        else:
            new_sentences_sentence_tagged.extend(nexts)

    # write out rttm
    rttms = []
    # vads = []
    for (sent, speaker) in new_sentences_sentence_tagged:
        start = sent[0]['start_ms_audio_time'] / 1000
        stop = sent[-1]['stop_ms_audio_time'] / 1000
        rttms.append(result.rttm.Rttm(file_id, start, stop-start, speaker))
        # vads.append((start, stop))
    rttm_out_path = os.path.join('result/label_rttm_xf', f'{file_id}.rttm')
    result.rttm.write_rttm(rttms, rttm_out_path)
    result.rttm.rttm_to_tsv(rttm_out_path, rttm_out_path + '.txt')
    adjacent_merged = os.path.join('result/label_rttm_xf_adjacent_merged', f'{file_id}.rttm')
    result.rttm.merge_rttm_by_adjacency(rttm_out_path, adjacent_merged, threshold_ms=200)
    result.rttm.rttm_to_tsv(adjacent_merged, adjacent_merged + '.txt')

    # vad using rttm
    # vad_out_path = os.path.join('result/vad_xf', f'{file_id}.lab')
    # result.rttm.write_lab(vads, vad_out_path)

    # write out vad
    # note that the vad is not obtained from the rttms,
    # as the rttms are cleaned,
    # which cannot be done in production.
    # however, to amend the missing human labels,
    # we drop those un-labeled segments from vad.

    new_vad_sentences = []
    for sent in vad_sentences:
        words = []
        for word in sent:
            if word['id'] not in visitor.words_dropped:
                words.append(word)
        if words:
            new_vad_sentences.append(words)
    vads = []
    for sent in new_vad_sentences:
        start = sent[0]['start_ms_audio_time'] / 1000
        stop = sent[-1]['stop_ms_audio_time'] / 1000
        vads.append((start, stop))
    vad_out_path = os.path.join('result/vad_xf', f'{file_id}.lab')
    result.rttm.write_lab(vads, vad_out_path)
    result.rttm.write_tsv(vads, vad_out_path + '.txt')

#%%

for path in glob('result/vad_xf/*.transcript.json'):
    print('processing', path)
    with open(path) as f:
        sentences = json.load(f)
    rows = [
        (x['words'][0]['start_ms_audio_time'] / 1000,
        x['words'][-1]['stop_ms_audio_time'] / 1000,
        x['text'])
        for x in sentences
    ]
    basename = os.path.basename(path)
    file_id = basename.split('.')[0]
    tsv_path = os.path.join('result/vad_xf', f'{file_id}.transcript.txt')
    result.rttm.write_tsv(rows, tsv_path)

#%%
import matplotlib.pyplot as plt
#%%
import result.rttm
import importlib
importlib.reload(result.rttm)
#%%
rows = []
deltas = []
for path in glob('result/vad_xf/*.lab'):
    file_rows = result.rttm.load_lab(path)
    new_file_rows = []
    result.rttm.merge_interval([(a, b, 'a') for a, b in file_rows], new_file_rows, 0.2)
    file_rows = [
        (a, b)
        for a, b, _ in new_file_rows
    ]
    rows.extend(file_rows)
    file_rows.sort()
    for (_, a), (b, _) in zip(file_rows[:-1], file_rows[1:]):
        deltas.append(1000 * (b - a))
lengths = [1000 * (b - a) for a, b in rows]
# lengths = [x for x in lengths if x <= 2000]
# deltas = [x for x in deltas if x <= ]
#%%
plt.hist([x for x in lengths if x < np.median(lengths)], bins=100)
plt.show()
#%%
plt.hist([x for x in lengths if x > np.median(lengths)], bins=100)
plt.show()
#%%
plt.hist([x for x in lengths if x < 20000], bins=100)
plt.show()
#%%
plt.hist(lengths, bins=100)
plt.show()
#%%
plt.hist(deltas, bins=100)
plt.show()
#%%
plt.hist([x for x in deltas if x < np.median(deltas)], bins=100)
plt.show()
#%%
plt.hist([x for x in deltas if x > np.median(deltas) and x < 15000], bins=100)
plt.show()
#%%
plt.hist([x for x in deltas if x < 15000], bins=100)
plt.show()
#%%
print('count lengths or short sentence', len(lengths))
print('count lengths or gaps', len(deltas))
#%%
np.median(lengths), np.median(deltas)
#%%

# print(len([1 for x in lengths if x < np.median(lengths)]))