import numpy as np


def beam_search(iterations, candidates, get_prob_by_seq, seq, beam_size=3):
    cur_beam = []
    for _ in range(iterations):
        if not cur_beam:
            probs = get_prob_by_seq(seq)
            weight_matrix = list(zip(list(map(lambda x: seq + [x], candidates)), np.log(probs)))
            cur_beam = sorted(weight_matrix, key=lambda x: x[1])[-beam_size:]
            continue
        weight_matrix = []
        for (seqbeam, weightbeam) in cur_beam:
            probs = get_prob_by_seq(seqbeam)
            weight_matrix += list(zip(list(map(lambda x: seqbeam + [x], candidates)),
                                      np.add(np.log(probs), np.array([weightbeam] * len(probs)))))
        cur_beam = sorted(weight_matrix, key=lambda x: x[1])[-beam_size:]
    return list(map(lambda x: x[0], cur_beam))

