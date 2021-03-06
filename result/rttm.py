"""
RTTM format:

Type -- segment type; should always by SPEAKER
File ID -- file name; basename of the recording minus extension (e.g., rec1_a)
Channel ID -- channel (1-indexed) that turn is on; should always be 1
Turn Onset -- onset of turn in seconds from beginning of recording
Turn Duration -- duration of turn in seconds
Orthography Field -- should always by <NA>
Speaker Type -- should always be <NA>
Speaker Name -- name of speaker of turn; should be unique within scope of each file
Confidence Score -- system confidence (probability) that information is correct; should always be <NA>
Signal Lookahead Time -- should always be <NA>
"""
import copy
import csv
import os.path
import re
import sys
from collections import defaultdict

import numpy as np
from dataclasses import dataclass

import result.show_st

from recursive import recursive, TailCall

@dataclass
class Rttm:
    file_id: str
    start: float
    duration: float
    speaker: str

    @property
    def stop(self):
        return self.start + self.duration


def convert_tsv_to_rttm(in_file_path, out_file_path):
    """
    Convert TSV of (start_second, stop_second, speaker) to RTTM format.
    """
    lines = []
    with open(in_file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                start, stop, speaker = line.split('\t')
                start = float(start)
                stop = float(stop)
                assert speaker == 's' or speaker == 't'
                lines.append((start, stop, speaker))

    basename = os.path.basename(in_file_path)
    basename, _ext = os.path.splitext(basename)
    rows = [Rttm(basename, start, stop - start, speaker) for
        start, stop, speaker in lines]
    write_rttm(rows, out_file_path)


def load_rttm(in_file_path):
    lines = []
    with open(in_file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                _type, file_id, _channel, start, duration, _na, _na, speaker, _na, _na = line.split(
                    ' '
                    )
                lines.append(
                    Rttm(file_id, float(start), float(duration), speaker)
                    )
    return lines


def write_rttm(rows, out_file_path):
    with open(out_file_path, 'w') as f:
        for rttm in rows:
            line = f'SPEAKER {rttm.file_id} 1 {rttm.start} {rttm.duration} <NA> <NA> {rttm.speaker} <NA> <NA>\n'
            f.write(line)


def write_tsv(rows, out_file_path):
    with open(out_file_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(rows)


def write_lab(rows, out_file_path):
    with open(out_file_path, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(rows)


def load_lab(in_file_path):
    rows = []
    with open(in_file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\s+', line)
            rows.append((float(parts[0]), float(parts[1])))
    return rows


def group_by(xs, get_key):
    groups = defaultdict(lambda: [])
    for x in xs:
        key = get_key(x)
        groups[key].append(x)
    return groups


def merge_rttm_by_adjacency(in_file_path, out_file_path, threshold_ms):
    """
    Merge neighbours of distance less than 3 seconds.
    """
    in_rows = load_rttm(in_file_path)
    # add index to keep order
    in_rows = list(enumerate(in_rows))
    # group by file_id
    groups = group_by(in_rows, lambda row: row[1].file_id)

    new_rows = []
    for same_file in groups.values():
        rows = merge_neighbour(same_file, threshold_ms)
        new_rows.extend(rows)
    # sort to restore the original order
    new_rows.sort()

    # drop index
    merged = [row for _index, row in new_rows]
    write_rttm(merged, out_file_path)


def rttm_to_tsv(in_file_path, out_file_path):
    rttms = load_rttm(in_file_path)
    rows = [(rttm.start, rttm.stop, rttm.speaker) for rttm in rttms]
    write_tsv(rows, out_file_path)


def merge_neighbour(rows_with_index, threshold_ms):
    threshold = threshold_ms / 1000
    while True:
        merge = False
        breaked = False
        a = None
        b = None
        r = None
        for i, row in rows_with_index:
            if breaked:
                break
            for i2, row2 in rows_with_index:
                # not the same segment, and is the same speaker
                if i2 != i and row2.speaker == row.speaker:
                    # check for threshold
                    (start1, stop1), (start2, stop2) = sorted(
                        [(row.start, row.stop), (row2.start, row2.stop)]
                    )
                    if stop1 + 100 / 1000 >= start2:
                        merge = True
                    elif stop1 + threshold >= start2:
                        # these two are subject to merge, considering the other speaker
                        if (not is_on_the_other_speaker(
                            stop1, rows_with_index, i, i2, row.speaker
                            ) and not is_on_the_other_speaker(
                            start2, rows_with_index, i, i2, row.speaker
                            ) and not is_on_the_other_speaker(
                            (min(stop1, start2), max(stop1, start2)),
                            rows_with_index, i, i2, row.speaker
                            )):
                            merge = True
                    else:
                        pass

                    # found one subject to merge, break to do the merge
                    if merge:
                        breaked = True
                        a = i
                        b = i2
                        r = row
                        break

        del i, i2, row, row2

        if merge:
            dropped = [(j, r) for j, r in rows_with_index if j != a and j != b]
            assert len(dropped) == len(rows_with_index) - 2
            new_row = Rttm(
                r.file_id, start1, max(stop1, stop2) - start1, r.speaker
            )
            dropped.append((min(a, b), new_row))
            # continue merging
            # return merge_neighbour(dropped)
            rows_with_index = dropped
        else:
            return rows_with_index


def is_on_the_other_speaker(end, rows, r1, r2, speaker):
    new_rows = [row for i, row in rows if
        i != r1 and i != r2 and row.speaker != speaker]
    for row in new_rows:
        if isinstance(end, float):
            if row.start <= end <= row.stop:
                return True
        elif isinstance(end, tuple):
            (start1, stop1), (start2, stop2) = sorted(
                [end, (row.start, row.stop)]
            )
            if stop1 > start2:
                return True
    return False


def rttm_to_oracle_vad(in_file_path, out_file_path):
    rows = load_rttm(in_file_path)

    # there is only one file in this rttm
    all_file_ids = set([row.file_id for row in rows])
    assert len(all_file_ids) == 1

    lines = []
    for row in rows:
        lines.append((row.start, row.stop))
    lines.sort()

    processed = []
    rest = lines
    while True:
        if len(rest) < 2:
            break
        else:
            a = (start1, stop1) = rest[0]
            b = (start2, stop2) = rest[1]
            rest = rest[2:]
            if stop1 >= start2:
                stop = max(stop1, stop2)
                start = start1
                rest.insert(0, (start, stop))
            else:
                rest.insert(0, b)
                processed.append(a)
    assert len(rest) == 1
    processed.append(rest[0])

    with open(out_file_path, 'w') as f:
        for start, stop in processed:
            f.write(f'{start} {stop} sp\n')


def align_sys_rttm_majority(in_file_path, out_file_path):
    rows = load_rttm(in_file_path)
    counts = defaultdict(lambda: 0)
    for row in rows:
        counts[row.speaker] += 1
    counts = [(v, k) for k, v in counts.items()]
    counts.sort()
    teacher_id = counts[-1][1]

    new_rows = []
    for row in rows:
        if row.speaker == teacher_id:
            row.speaker = 't'
        else:
            row.speaker = 's'
        new_rows.append(row)

    write_rttm(new_rows, out_file_path)


def intersection(start1, stop1, tag1, start2, stop2, tag2):
    (start1, stop1, tag1), (start2, stop2, tag2) = sorted(
        [(start1, stop1, tag1), (start2, stop2, tag2)]
        )
    if stop1 > start2:
        a = (start1, start2, tag1)
        b = (start2, min(stop1, stop2), 'intersection')
        if stop1 <= stop2:
            c = (stop1, stop2, tag2)
        else:
            c = (stop2, stop1, tag1)
        assert a[0] <= a[1] <= b[0] <= b[1] <= c[0] <= c[1]
        returns = []
        for x in (a, b, c):
            start, stop, speaker = x
            if np.abs(stop - start) <= 1e-6:
                pass
            else:
                returns.append(x)
        return returns

    else:
        return None


def st_intersection(start1, stop1, tag1, start2, stop2, tag2):
    (start1, stop1, tag1), (start2, stop2, tag2) = sorted(
        [(start1, stop1, tag1), (start2, stop2, tag2)]
        )
    if stop1 > start2:
        a = (start1, start2, tag1)
        b = (start2, min(stop1, stop2), 't')
        if stop1 <= stop2:
            c = (stop1, stop2, tag2)
        else:
            c = (stop2, stop1, tag1)
        assert a[0] <= a[1] <= b[0] <= b[1] <= c[0] <= c[1]
        returns = []
        for x in (a, b, c):
            start, stop, speaker = x
            if np.abs(stop - start) <= 1e-3:
                pass
            else:
                returns.append(x)
        return returns

    else:
        return None


def merge_rttm_by_st(in_file_path, out_file_path):
    """
    Tun st into t for st plotting
    """
    rows = load_rttm(in_file_path)
    file_id = rows[0].file_id

    ts = list(filter(lambda r: r.speaker == 't', rows))
    ts = [(r.start, r.stop, r.speaker) for r in ts]
    ss = list(filter(lambda r: r.speaker == 's', rows))
    ss = [(r.start, r.stop, r.speaker) for r in ss]
    assert len(ts) + len(ss) == len(rows)

    # total_teacher = np.sum([stop - start for start, stop, _ in ts])
    # total_student = np.sum([stop - start for start, stop, _ in ss])

    while True:
        found_intersection = False

        breaked = False
        t_len = len(ts)
        t_popped = 0
        while True:
            if breaked or t_popped >= t_len:
                break
            l1 = len(ts)
            t = ts.pop(0)
            assert l1 == len(ts) + 1
            t_popped += 1

            s_len = len(ss)
            s_popped = 0
            while s_popped < s_len:
                l1 = len(ss)
                s = ss.pop(0)
                assert l1 == len(ss) + 1
                s_popped += 1
                intersection = st_intersection(*s, *t)
                if intersection is None:
                    ss.append(s)
                else:
                    found_intersection = True
                    # print(t, s, intersection)
                    for (start, stop, speaker) in intersection:
                        if speaker == 's':
                            ss.append((start, stop, speaker))
                        elif speaker == 't':
                            ts.append((start, stop, speaker))
                        else:
                            assert False
                    breaked = True
                    break
            if not breaked:
                ts.append(t)

        if not found_intersection:
            break

    ts.sort()
    ss.sort()

    # sanity check
    # total_teacher_after = np.sum([stop - start for start, stop, _ in ts])
    # total_student_after = np.sum([stop - start for start, stop, _ in ss])
    # print(total_teacher, total_teacher_after)
    # print(total_student, total_student_after)
    # delta_teacher = total_teacher - total_teacher_after
    # assert delta_teacher >= 0
    # delta_student = total_student_after - total_student
    # assert delta_student >= 0
    # delta = np.abs(delta_teacher - delta_student)
    # assert delta <= 0.1

    all_rows = []
    all_rows.extend(merge_adjacent_of_same_speaker(ts, []))
    all_rows.extend(merge_adjacent_of_same_speaker(ss, []))
    all_rows.sort()
    all_rows = [Rttm(file_id, start, stop - start, speaker) for
        start, stop, speaker in all_rows]
    write_rttm(all_rows, out_file_path)


def merge_adjacent_of_same_speaker(xs, merged):
    xs.sort()
    if not xs:
        return merged

    h = start1, stop1, speaker = xs[0]
    t = xs[1:]

    # try to find a merge-able neighbour
    for i, (start2, stop2, _) in enumerate(t):
        if stop1 >= start2:
            # found a neighbour, merge them
            new = (start1, max(stop1, stop2), speaker)
            # two becomes one: head is dropped, i is dropped, new is added
            t.pop(i)
            # new is still subject to merge
            t.append(new)
            return merge_adjacent_of_same_speaker(t, merged)

    merged.append(h)
    return merge_adjacent_of_same_speaker(t, merged)


# def merge_vad(xs, merged):
#   xs.sort()
#   if not xs:
#     return merged
#
#   h = start1, stop1 = xs[0]
#   t = xs[1:]
#
#   # try to find a merge-able neighbour
#   for i, (start2, stop2) in enumerate(t):
#     if stop1 + 1 >= start2:
#       # found a neighbour, merge them
#       new = (start1, max(stop1, stop2))
#       # two becomes one: head is dropped, i is dropped, new is added
#       t.pop(i)
#       # new is still subject to merge
#       t.append(new)
#       return merge_vad(t, merged)
#
#   merged.append(h)
#   return merge_vad(t, merged)


# def merge_vad(xs, merged):
#     while xs:
#         xs.sort()
#
#         h = start1, stop1 = xs[0]
#         t = xs[1:]
#
#         breaked = False
#
#         # try to find a merge-able neighbour
#         for i, (start2, stop2) in enumerate(t):
#             if stop1 + 1.2 >= start2:
#                 # found a neighbour, merge them
#                 new = (start1, max(stop1, stop2))
#                 # two becomes one: head is dropped, i is dropped, new is added
#                 t.pop(i)
#                 # new is still subject to merge
#                 t.append(new)
#                 xs = t
#                 breaked = True
#                 break
#         if breaked:
#             continue
#         merged.append(h)
#         xs = t
#
#     return merged
#
#
# def merge_vad_file(in_file_path, out_file_path):
#     rows = load_lab(in_file_path)
#     rows.sort()
#     rows = merge_vad(rows, [])
#     rows = [(a, b, 'sp') for a, b in rows]
#     write_lab(rows, out_file_path)


def plot_st(in_rttm_paths, save_path, hints=[]):
    rowss = [
        [[row.start, row.stop, row.speaker] for row in load_rttm(in_rttm_path)]

        for in_rttm_path in in_rttm_paths]

    return result.show_st.plot(rowss, in_rttm_paths, save_path, hints)


def plot_st_cli(in_paths, save_path, hints):
    in_paths = in_paths.split(':')
    hints = hints.split(':')
    return plot_st(in_paths, save_path, hints)


def lab_to_txt(in_file_path, out_file_path):
    lab = load_lab(in_file_path)
    rows = []
    for start, stop in lab:
        rows.append((start, stop, 'sp'))
    write_tsv(rows, out_file_path)

@recursive
def find_false_alarms(truths, preds, false_alarms):
    if not preds:
        # all predictions are consumed
        return None
    else:
        h = start, stop, _ = preds[0]
        t = preds[1:]

        # try to find an intersection for h in truths
        # if one is found, then the length of h is reduced by the intersection,
        # in this case, we continue to process the rest of h
        for start2, stop2, _ in truths:
            intersections = intersection(start, stop, 'a', start2, stop2, 'b')
            if intersections is None:
                continue
            else:
                all_as = [x for x in intersections if x[2] == 'a']
                if len(all_as) == 0:
                    # the h is fully consumed, thus it is not a false alarm
                    return TailCall(truths, t, false_alarms)
                else:
                    # the h is partially consumed, process the rest of h
                    t.extend(all_as)
                    return TailCall(truths, t, false_alarms)

        # if no intersections are found for h, then h must be false alarm
        false_alarms.append(h)
        return TailCall(truths, t, false_alarms)

def find_misses(truths, preds, misses):
    return find_false_alarms(preds, truths, misses)

@recursive
def find_wrong_labels(truths, preds, wrongs):
    if not preds:
        return None
    else:
        h = (start, stop, speaker) = preds[0]
        t = preds[1:]

        for start2, stop2, speaker2 in truths:
            intersections = intersection(start, stop, 'a', start2, stop2, 'b')
            if intersections is None:
                continue
            else:
                all_is = [x for x in intersections if x[2] == 'intersection']
                for i_start, i_stop, _ in all_is:
                    if speaker != speaker2:
                        wrongs.append(
                            (i_start, i_stop, speaker2 + '->' + speaker)
                        )

                all_as = [x for x in intersections if x[2] == 'a']
                for i_start, i_stop, _ in all_as:
                    if has_intersection(i_start, i_stop, truths):
                        t.insert(0, (i_start, i_stop, speaker))

                return TailCall(truths, t, wrongs)

        # if we reached here without return,
        # then there is no intersection with h,
        # which means h is a false alarm,
        # so drop it
        return TailCall(truths, t, wrongs)


def has_intersection(start, stop, xs):
    for start2, stop2, _ in xs:
        intersections = intersection(start, stop, 'a', start2, stop2, 'b')
        if intersections is None:
            continue
        else:
            for _, _, tag in intersections:
                if tag == 'intersection':
                    return True
    return False


@recursive
def merge_interval(xs, merged, tol):
    # xs: list of triple
    xs.sort()
    if len(xs) == 0:
        merged.sort()
        return
    else:
        h = (start1, stop1, tag1) = xs[0]
        # tail with the same tag as the head
        ht = [x for x in xs[1:] if x[2] == tag1]
        # tail with the different tag as the head
        tt = [x for x in xs[1:] if x[2] != tag1]
        if not ht:
            merged.append(h)
            return TailCall(tt, merged, tol)
        else:
            h2 = (start2, stop2, _) = ht[0]
            ht = ht[1:]
            (start1, stop1), (start2, stop2) = sorted([(start1, stop1), (start2, stop2)])
            if stop1 + tol >= start2:
                # merge h and h2, consumes both
                new_h = (start1, max(stop1, stop2), tag1)
                ht.insert(0, new_h)
                return TailCall(tt + ht, merged, tol)
            else:
                # no need to merge head
                merged.append(h)
                # return back h2
                ht.insert(0, h2)
                return TailCall(tt + ht, merged, tol)

def drop_short(xs):
    return [
        x for x in xs
        if x[1] - x[0] > 200 / 1000
    ]

def diff_rttm(rttm_path_true, rttm_path_pred, out_csv_template_name):
    trues = load_rttm(rttm_path_true)
    preds = load_rttm(rttm_path_pred)
    trues = [(rttm.start, rttm.stop, rttm.speaker) for rttm in trues]
    preds = [(rttm.start, rttm.stop, rttm.speaker) for rttm in preds]

    trues_fa = copy.deepcopy(trues)
    preds_fa = copy.deepcopy(preds)
    _false_alarms = []
    find_false_alarms(trues_fa, preds_fa, _false_alarms)
    false_alarms = []
    merge_interval(_false_alarms, false_alarms)
    false_alarms = [(a, b, 'fa') for (a, b, _) in false_alarms]
    false_alarms = drop_short(false_alarms)
    total_fas = np.sum([b-a for a, b, _ in false_alarms])

    trues_miss = copy.deepcopy(trues)
    preds_miss = copy.deepcopy(preds)
    _misses = []
    find_misses(trues_miss, preds_miss, _misses)
    misses = []
    merge_interval(_misses, misses)
    misses = [(a, b, 'miss') for (a, b, _) in misses]
    misses = drop_short(misses)
    total_misses = np.sum([b-a for a, b, _ in misses])


    trues_wrong = copy.deepcopy(trues)
    preds_wrong = copy.deepcopy(preds)
    _wrongs = []
    find_wrong_labels(trues_wrong, preds_wrong, _wrongs)
    wrongs = []
    merge_interval(_wrongs, wrongs)
    wrongs = drop_short(wrongs)
    total_wrongs = np.sum([b-a for a, b, _ in wrongs])

    total_trues = np.sum([b-a for a, b, _ in trues])
    total_preds = np.sum([b-a for a, b, _ in preds])
    der = (total_fas + total_misses + total_wrongs) / (total_trues + total_preds)
    print(f'FA    {total_fas:.4f}')
    print(f'MISS  {total_misses:.4f}')
    print(f'ERROR {total_wrongs:.4f}')
    print(f'TRUE  {total_trues:.4f}')
    print(f'PRED  {total_preds:.4f}')
    print(f'RATIO {der:.4f}')

    write_tsv(trues, out_csv_template_name + '.true.txt')
    write_tsv(preds, out_csv_template_name + '.pred.txt')
    write_tsv(false_alarms, out_csv_template_name + '.fa.txt')
    write_tsv(misses, out_csv_template_name + '.miss.txt')
    write_tsv(wrongs, out_csv_template_name + '.wrong.txt')


# plot_st(
#   [
#     'VBx/example/rttm/M_clean_chenrushen_yuanzhutiji.rttm',
#     'VBx/exp/M_clean_chenrushen_yuanzhutiji_majority_merged.rttm'
#   ],
#   'VBx/exp/oracle_vad.png',
#   ['oracle vad']
#  )

# convert_rttm_for_st(
#   'result/label_rttm_merged/M_clean_chenrushen_yuanzhutiji.rttm',
#   'result/label_rttm_merged/M_clean_chenrushen_yuanzhutiji_stmerged.rttm'
# )
# rttm_to_tsv(
#   'result/label_rttm_merged/M_clean_chenrushen_yuanzhutiji_stmerged.rttm',
#   'result/label_rttm_merged/M_clean_chenrushen_yuanzhutiji_stmerged.rttm.txt',
# )
# %%


if __name__ == '__main__':
    fun_name = sys.argv[1]
    args = sys.argv[2:]
    fun = globals()[fun_name]
    fun(*args)
