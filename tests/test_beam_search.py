import unittest
from rhythm.rhythm_handler import RhythmHandler
import numpy as np
import logging
import models.beam_search as bm


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

    # Unit test for depricated function beam_search
    """
    def test_lol_kek_cheburek(self):
        seq = ['лол', 'кек', 'чебурек', 'кот']

        def probs(lst):
            if not lst:
                return [0.9, 0.0, 0.0, 0.1]
            if lst == ['лол']:
                return [.0, .7, .2, .1]
            if lst == ['лол', 'кек']:
                return [.1, .2, .5, 0.2]
            return [.25, .25, .25, .25]

        assert(bm.beam_search(iterations=3, candidates=seq, get_prob_by_seq=probs, seq=[])[-1]
            == ['лол', 'кек', 'чебурек'])
    """

if __name__ == '__main__':
    unittest.main()
