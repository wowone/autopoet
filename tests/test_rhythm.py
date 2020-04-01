import unittest
from rhythm.rhythm_handler import RhythmHandler
import numpy as np
import logging


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)

   # def __init__(self):
    #    self.yamb_handler = RhythmHandler(get_stress=lambda x: 1, dict_size=10)

    def test_get_masks_yamb(self):
        yamb_handler = RhythmHandler(get_stress=lambda x: 1, rhythm=(1, 2))
        yamb_handler.generate_masks()

        self.assertListEqual([1, 0] * (yamb_handler.max_syllables // 2),
                             yamb_handler.masks_list[0].tolist())
        self.assertListEqual([0, 1] * (yamb_handler.max_syllables // 2),
                             yamb_handler.masks_list[1].tolist())

    def test_get_masks_horey(self):
        horey_handler = RhythmHandler(get_stress=lambda x: 1, rhythm=(0, 2))
        horey_handler.generate_masks()

        self.assertListEqual([0, 1] * (horey_handler.max_syllables // 2),
                             horey_handler.masks_list[0].tolist())
        self.assertListEqual([1, 0] * (horey_handler.max_syllables // 2),
                             horey_handler.masks_list[1].tolist())

    def test_get_masks_daktil(self):
        daktil_handler = RhythmHandler(get_stress=lambda x: 1, rhythm=(0, 3))
        daktil_handler.generate_masks()

        self.assertListEqual([0, 1, 1] * (daktil_handler.max_syllables // 3),
                             daktil_handler.masks_list[0].tolist())
        self.assertListEqual([1, 0, 1] * (daktil_handler.max_syllables // 3),
                             daktil_handler.masks_list[1].tolist())
        self.assertListEqual([1, 1, 0] * (daktil_handler.max_syllables // 3),
                             daktil_handler.masks_list[2].tolist())

    def test_get_word_vowels(self):
        handler = RhythmHandler(get_stress=lambda x: 1)
        self.assertEqual(handler.get_word_vowels('арбуз'), [0, 3])
        self.assertEqual(handler.get_word_vowels('мотоцикл'), [1, 3, 5])
        self.assertEqual(handler.get_word_vowels('восемь'), [1, 3])

    def test_get_syllables_coutn(self):
        handler = RhythmHandler(get_stress=lambda x: 1)
        self.assertEqual(handler.get_syllables_count('кошка'), 2)
        self.assertEqual(handler.get_syllables_count('собака'), 3)
        self.assertEqual(handler.get_syllables_count('сороковые'), 5)

    def test_horey_get_masks_for_stresses(self):
                    # вИхри:0 снЕжные:0 крутЯ:1 собАка:1 велосипЕд: 3 абрикОс:2
        stress_list = [0, 0, 1, 1, 3, 2]
        handler = RhythmHandler(get_stress=lambda x: 1, rhythm=(0, 2))
        masks = handler.get_masks_for_stresses(stress_list)
        self.assertEqual(masks[0].tolist(), [1, 1, 0, 0, 0, 1])
        self.assertEqual(masks[1].tolist(), [0, 0, 1, 1, 1, 0])


    def test_daktil_get_masks_for_stresses(self):
        stress_list = [0, 0, 1, 1, 3, 2, 4, 1, 3]
        handler = RhythmHandler(get_stress=lambda x: 1, rhythm=(0, 3))
        masks = handler.get_masks_for_stresses(stress_list)
        self.assertEqual(masks[0].tolist(), [1, 1, 0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(masks[1].tolist(), [0, 0, 1, 1, 0, 0, 1, 1, 0])
        self.assertEqual(masks[2].tolist(), [0, 0, 0, 0, 0, 1, 0, 0, 0])

    def test_get_stress_syllable(self):
        words = ['собака', 'кошка', 'спорт', 'география', 'будапешт', 'в']
        stresses = {words[0]: 3, words[1]: 1, words[2]: 2, words[3]: 5, words[4]: 5, words[5]: -1}
        get_stress = lambda x: stresses[x]
        handler = RhythmHandler(get_stress)
        self.assertEqual(handler.get_stress_syllable(words[0]), 1)
        self.assertEqual(handler.get_stress_syllable(words[1]), 0)
        self.assertEqual(handler.get_stress_syllable(words[2]), 0)
        self.assertEqual(handler.get_stress_syllable(words[3]), 2)
        self.assertEqual(handler.get_stress_syllable(words[4]), 2)
        self.assertEqual(handler.get_stress_syllable(words[5]), -1)

    def test_get_masks_for_words(self):
        words = ['собака', 'кошка', 'спорт', 'география', 'будапешт', 'в']
        stresses = {words[0]: 3, words[1]: 1, words[2]: 2, words[3]: 5, words[4]: 5, words[5]: -1}
        get_stress = lambda x: stresses[x]
        handler = RhythmHandler(get_stress, rhythm=(0, 2))
        masks = handler.get_masks_for_words(words)
        #print(handler.list_to_stressed(words))
        self.assertEqual(masks[0].tolist(), [0, 1, 1, 1, 1, 0])
        self.assertEqual(masks[1].tolist(), [1, 0, 0, 0, 0, 0])

    def test_horey_get_next_shift(self):
        handler = RhythmHandler(get_stress=lambda x: 1, rhythm=(0, 2))
        word = 'вихри'
        shift = handler.get_next_shift(word, prev_shift=0)
        self.assertEqual(shift, 0)
        word = 'снежные'
        shift = handler.get_next_shift(word, prev_shift=shift)
        self.assertEqual(shift, 1)

    def test_daktil_get_next_shift(self):
        handler = RhythmHandler(get_stress=lambda x: 1, rhythm=(0, 3))
        word = 'звенящими'
        shift = handler.get_next_shift(word, prev_shift=0)
        self.assertEqual(shift, 2)
        word = 'косами'
        shift = handler.get_next_shift(word, prev_shift=shift)
        self.assertEqual(shift, 2)
        word = 'было'
        shift = handler.get_next_shift(word, prev_shift=shift)
        self.assertEqual(shift, 0)


if __name__ == '__main__':
    unittest.main()
