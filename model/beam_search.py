import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.tokenize import word_tokenize
import re
from rupo.api import Engine
import numpy as np
import model.language_model as lang
from tokenizer import MyTokenizer
import rhythm.rhythm_handler as rh


def generate_probas(preds, temperature):
    # TODO: rewrite this
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # probas = np.random.multinomial(1, preds, 1)  # Why multinomial?
    return preds



# Return the longest common suffix in a list of strings
def longest_common_suffix(a, b):
    b = list(reversed(b))
    a = list(reversed(a))
    ans = 0
    for (ca, cb) in zip(a, b):
        if ca == cb:
            ans += 1
        else:
            break
    return ans


# Will be replaced with more complicated
def rhyme_mask(word, words):
    if word == "":
        return np.array([1] * len(words))
    mask = np.array([0] * len(words))
    for i, w in enumerate(words):
        if longest_common_suffix(w, word) >= 2:
            mask[i] = 1
    return mask

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


# Kinda Beam Father
class SeqGenerator:
    def __init__(self, model, tokenizer, seq_len, stress_masks, rhythm_handler, words, rhyme_dict, beam_size):
        """model, seq_len, seed, lines_len, stress_masks, rhythm_handler, words, footness,
                      rhyme_module, rhyme_dict,  beam_size=100"""
        self.model = model
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.rhythm_handler = rhythm_handler
        self.words = words
        self.dict_len = len(words)
        self.seed = ""
        self.stress_masks = stress_masks
        self.rhyme_dict = rhyme_dict
        self.rhyme_module = len(rhyme_dict)
        self.beams = [Beam(self)] * beam_size

    def fit_seed(self, seed):
        self.seed = seed

    def generate_poem(self, lines_len, footness):
        """
        for lines:
            while !end_of_line:
                for every_beam:
                    beam_step
                weights = get_all_beam_weights()
                beams = get_top_beams(weights)
                beams.update_if_survive(how many beams my suns alive)
            beam_end_of_line
        """
        """
        beam_step:
            mask = mask_by_shift
            mask = mask_by_rhyme
            seq = get_seq_for_beam
            probas = model.get_probas(seq)
            weigths.update()
            beam_update()
        """
        for beam in self.beams:
            beam.fit(self.seed)
        for line_index in range(lines_len):
            stress_count = 0
            for beam in self.beams:
                beam.update_shift()
            # TODO: What if beam choose word withoud stress? Lol. I've no solution right now, so.
            while stress_count != footness:
                weigths = []
                is_final = footness - 1 == stress_count
                line = line_index % self.rhyme_module
                for i, beam in enumerate(self.beams):
                    beam.step(line, is_final)
                    beam_weights = beam.weights
                    weigths += list(map(lambda x: (i, x), beam_weights))
                weigths = sorted(weigths, key=lambda x: x[1][1])[-len(self.beams):]
                new_beams = []
                for (i, beam_inf) in weigths:
                    old_beam = self.beams[i]
                    new_beam = Beam(self)
                    new_beam.beam_weight = beam_inf[1]
                    new_beam.poem = beam_inf[0]
                    new_beam.shift = old_beam.shift
                    new_beam.str_poem = old_beam.str_poem
                    new_beam.last_words = old_beam.last_words
                    new_beam.update(line, is_final)

                    del old_beam  # kill the beam?
                    new_beams.append(new_beam)
                self.beams = new_beams
                # TODO: is_stressed?
                stress_count += 1
        for i, beam in enumerate(self.beams):
            print(f"Beam number {i} once wrote:")
            print(beam.str_poem)


class Beam:
    def __init__(self, handler):
        self.handler = handler
        self.shift = 0
        self.last_words = {-1: ""}
        for i in range(handler.rhyme_module):
            self.last_words[i] = ""
        self.poem = ""
        self.str_poem = ""
        self.neutral_mask = np.array([1] * handler.dict_len)
        self.weights = []
        self.beam_weight = 0

    def fit(self, seed):
        self.poem = seed

    def step(self, line, is_final):
        mask_stress = self.handler.stress_masks[self.shift]
        if is_final:
            mask_rhyme = rhyme_mask(self.last_words[self.handler.rhyme_dict[line]], self.handler.words)
        else:
            mask_rhyme = self.neutral_mask
        mask = np.multiply(mask_rhyme, mask_stress)
        encoded = self.handler.tokenizer.text_to_seq([self.poem])[0]
        # TODO: Padded with zeros! Horrible solution! REDO!!!
        encoded = pad_sequences([encoded], maxlen=self.handler.seq_len, truncating='pre')

        prediction = self.handler.model.predict(encoded)[0]
        probas = generate_probas(prediction, temperature=0.7)
        self.update_weights(np.multiply(probas, mask))

    def update_shift(self):
        self.shift = 0

    def update_weights(self, probas):
        next_seqs = list(map(lambda x: self.poem + ' ' + x, self.handler.words))
        next_weights = np.add(np.log(probas), np.array([self.beam_weight] * len(probas)))
        self.weights = list(zip(next_seqs, next_weights))

    def update(self, line, is_final):
        predicted_word = self.poem.split(' ')[-1]
        self.str_poem += predicted_word + ' '
        self.shift = self.handler.rhythm_handler.get_next_shift(predicted_word, self.shift)
        if is_final:
            self.last_words[line] = predicted_word
            self.str_poem += '\n'
