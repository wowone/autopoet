from keras.preprocessing.sequence import pad_sequences
import numpy as np

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


# Kinda Beam Father
class SeqGenerator:
    def __init__(self, model, tokenizer, seq_len, stress_masks, rhythm_handler, words, rhyme_dict, beam_size,
                 temperature=1.0):
        """
        :param model: model with predict function
        :param tokenizer: tokenizer with text_to_seq function
        :param seq_len: length of model input layer
        :param stress_masks: binary masks for words with different rhythm shifts
        :param rhythm_handler: rhythm_handler with get_next_shift function
        :param words: dictionary of all words sorted in the same way as in model and tokenizer!
        :param rhyme_dict: dictionary for rhyme in form [line_number] : [rhymed_line_number]
                           example for ABAB : {0: -1, 1: -1, 2: 0, 3: 1}
                           example for AABB : {0: -1, 1: 0, 2: -1, 3: 2}
        :param beam_size: size of beam bucket (bigger ---> better, but slower)
        :param temperature: some constant for prediction calculation. ML technique to regulate model confidence
        """
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
        self.beams = []
        self.beam_size = beam_size
        self.temperature = temperature
        self.neutral_mask = np.array([1] * self.dict_len)

    def generate_probas(self, preds):
        # exp^(log(prob) / T) / (sum (exp))
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / self.temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        return preds

    def fit_seed(self, seed):
        self.seed = seed

    def generate_poem(self, lines_len, footness):
        for line_index in range(lines_len):
            stress_count = 0
            for beam in self.beams:
                # Set shift to zero in all beams
                # New line, new rhythm shifting
                beam.update_shift()
            # TODO: What if beam choose word withoud stress?
            while stress_count != footness:
                weights = []
                # Really important for all beams, is it final word of line and rhyme needed
                is_final = footness - 1 == stress_count
                # line is index of current line in rhyme_dict
                line = line_index % self.rhyme_module
                if not self.beams:
                    # First iteration of beam settings, just initialisation
                    # Runs once in generate_poem function
                    start_beam = Beam(self)
                    start_beam.fit(self.seed)
                    start_beam.step(line, is_final)
                    weights += list(map(lambda x: (0, x), start_beam.weights))
                    self.beams = [start_beam]
                else:
                    # weights - list of all beams predictions
                    # len(weights) == beam_size * len(words)

                    for i, beam in enumerate(self.beams):
                        beam.step(line, is_final)
                        beam_weights = beam.weights
                        weights += list(map(lambda x: (i, x), beam_weights))
                # Gen only first beam_size poems, sorted by probability
                weights = sorted(weights, key=lambda x: x[1][1])[-self.beam_size:]
                # Now we should create list of new, most probable beams
                new_beams = []
                for (i, beam_inf) in weights:
                    old_beam = self.beams[i]
                    new_beam = Beam(self)
                    # Save old_beam state
                    new_beam.beam_weight = beam_inf[1]
                    new_beam.poem = beam_inf[0]
                    new_beam.shift = old_beam.shift
                    new_beam.str_poem = old_beam.str_poem
                    # Very important! Copy value, not reference
                    new_beam.last_words = old_beam.last_words.copy()

                    new_beam.update(line, is_final)
                    new_beams.append(new_beam)

                # Should garbage collector do that?
                #for beam in self.beams:
                #    del beam

                # Lose references on old_beams, they must be deleted by GC
                self.beams = new_beams
                # TODO: is_stressed?
                stress_count += 1
        for i, beam in enumerate(self.beams):
            print(f"Beam number {i} once wrote:")
            print(f"poem weight: {beam.beam_weight}")
            print(beam.str_poem)


# Should be nested class
class Beam:
    def __init__(self, handler):
        self.handler = handler
        self.shift = 0
        self.last_words = {-1: ""}
        for i in range(handler.rhyme_module):
            self.last_words[i] = ""
        self.poem = ""
        self.str_poem = ""
        self.weights = []
        self.beam_weight = 0

    def fit(self, seed):
        self.poem = seed

    def step(self, line, is_final):
        mask_stress = self.handler.stress_masks[self.shift]
        if is_final:
            mask_rhyme = rhyme_mask(self.last_words[self.handler.rhyme_dict[line]], self.handler.words)
        else:
            mask_rhyme = self.handler.neutral_mask
        mask = np.multiply(mask_rhyme, mask_stress)
        encoded = self.handler.tokenizer.text_to_seq([self.poem])[0]
        # TODO: Padded with zeros! Horrible solution! REDO!!!
        encoded = pad_sequences([encoded], maxlen=self.handler.seq_len, truncating='pre')

        prediction = self.handler.model.predict(encoded)[0]
        probas = self.handler.generate_probas(prediction, temperature=0.4)
        self.update_weights(np.multiply(probas, mask))

    def update_shift(self):
        self.shift = 0

    def update_weights(self, probas):
        next_seqs = list(map(lambda x: self.poem + ' ' + x, self.handler.words))
        next_weights = np.add(np.log(probas), np.array([self.beam_weight] * len(probas)))
        self.weights = list(zip(next_seqs, next_weights))
        self.prob_max = np.max(np.array(list(map(lambda x: x[1], self.weights))))

    def update(self, line, is_final):
        predicted_word = self.poem.split(' ')[-1]
        self.str_poem += predicted_word + ' '
        self.shift = self.handler.rhythm_handler.get_next_shift(predicted_word, self.shift)
        if is_final:
            self.last_words[line] = predicted_word
            self.str_poem += '\n'
