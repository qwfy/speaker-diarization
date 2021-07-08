import numpy as np
from tabulate import tabulate

import result.rttm


def evaluate_from_file(truth_label_file, prediction_label_file):
    truths = result.rttm.load_lab(truth_label_file)
    predictions = result.rttm.load_lab(prediction_label_file)
    return evaluate_from_tuples(truths, predictions)


def evaluate_from_tuples(truths, predictions):
    # truths = result.rttm.load_lab('result/vad_silero/M_echo_sjzm.oracle.lab')
    # predictions = result.rttm.load_lab('result/vad_silero/M_echo_sjzm.lab')
    # sanity check
    # predictions = [x for x in truths]

    assert_no_intersection(truths)
    assert_no_intersection(predictions)

    tp, fp = partition_predictions(truths, predictions)
    truth_silent = complement(truths)
    prediction_silent = complement(predictions)
    tn, fn = partition_predictions(truth_silent, prediction_silent)

    sum_tp = sum_ends(tp)
    sum_fp = sum_ends(fp)
    sum_tn = sum_ends(tn)
    sum_fn = sum_ends(fn)

    p = sum_tp / (sum_tp + sum_fp)
    r = sum_tp / sum_ends(truths)
    f1 = 2 * p * r / (p + r)
    n_p = sum_tn / (sum_fn + sum_tn)
    n_r = sum_tn / sum_ends(truth_silent)
    n_f1 = 2 * n_p * n_r / (n_p + n_r)

    result = [# 0:4
        sum_tp, sum_fn, sum_tp + sum_fn, sum_ends(truths), # 4:8
        sum_fp, sum_tn, sum_fp + sum_tn, sum_ends(truth_silent), # 8:11,
        p, r, f1, # 11:14
        n_p, n_r, n_f1]

    return result


def show_evaluation(result, confusion=True):
    y = ['Y']
    y.extend(result[0:4])
    n = ['N']
    n.extend(result[4:8])
    table = [['A\P', 'Y', 'N', 'P _', 'A _'], y, n]
    if confusion:
        print(tabulate(table, headers='firstrow'))

    print('---')
    p, r, f1 = result[8:11]
    print('precision  ', p)
    print('recall     ', r)
    print('F1         ', f1)

    print('---')
    n_p, n_r, n_f1 = result[11:14]
    print('precision n', n_p)
    print('recall    n', n_r)
    print('F1        n', n_f1)


def assert_no_intersection(xs):
    for i, (start1, stop1) in enumerate(xs):
        ys = [y for y in xs]
        ys.pop(i)
        for start2, stop2 in ys:
            assert result.rttm.intersection(
                start1, stop1, 'a', start2, stop2, 'b'
                ) is None


def get_first_intersection(start1, stop1, xs):
    xs = sorted(xs)
    disjunction = None
    intersection = None
    unused = None

    for start2, stop2 in xs:
        intersections = result.rttm.intersection(
            start1, stop1, 'a', start2, stop2, 'b'
            )
        if intersections is None:
            # if there is not a single intersection
            # then the entire segment is a disjunction (the final return)
            pass
        else:
            for start, stop, tag in intersections:
                if tag == 'intersection':
                    assert intersection is None
                    intersection = (start, stop)
                else:
                    pass
            for start, stop, tag in intersections:
                if tag == 'intersection':
                    pass
                elif tag == 'a':
                    if intersection is None:
                        assert False  # (_, _stop, _), (_start, _, _) = sorted(intersections)  # if abs(_start - _stop) <= 1e-3:  #   continue  # else:  #   assert False
                    else:
                        start_i, stop_i = intersection
                        if stop <= start_i:
                            disjunction = (start, stop)
                        elif stop_i <= start:
                            unused = (start, stop)
                        else:
                            assert False
                elif tag == 'b':
                    pass
                else:
                    assert False
            return disjunction, intersection, unused

    return (start1, stop1), None, None


def partition_predictions(truths, predictions):
    """
    Split predictions into two parts:
    - those intersect with truths
    - and those don't
    """
    predictions = [x for x in predictions]
    no_match = []
    match = []

    while predictions:
        start, stop = predictions.pop(0)
        disjunction, intersection, unused = get_first_intersection(
            start, stop, truths
            )
        if disjunction is not None:
            no_match.append(disjunction)
        if intersection is not None:
            match.append(intersection)
        if unused is not None:
            predictions.append(unused)

    return match, no_match


def complement(xs):
    c = []
    xs = sorted(xs)
    current_start = 0
    for start, stop in xs:
        assert current_start <= start
        c.append((current_start, start))
        current_start = stop
    #
    # if current_start < upper_bound:
    #   c.append((current_start, upper_bound))

    return c


def sum_ends(xs):
    return np.sum([stop - start for start, stop in xs])
