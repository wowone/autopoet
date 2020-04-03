# This is example of generation poems using our model

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


def get_word(prediction, mask, temperature=1):
    p = generate_probas(prediction, temperature)
    mmask = np.multiply(p, mask)
    # print(np.max(mmask))
    return tokenizer.id_to_word[np.argmax(mmask)]
    # return tokenizer.id_to_word[np.argmax(p)]


def generate_seq2(model, seq_length, seed_text, lines_len, masks, handler, words, footness=4):
    result = list()
    in_text = seed_text
    # generate a fixed number of words

    # TODO: Footness isn't just count of words, but maximum count of stresses in one line

    index = 0
    shift = 0
    line_index = 0
    lines_count = 0
    mask_default = np.array([1] * len(words))
    rhyme_word = ["", ""]

    while (lines_count != lines_len):
        line_index = index % footness
        change_line = ((line_index + 1) % footness) == 0

        mask = masks[shift]
        if change_line and (lines_count % 4 == 2 or lines_count % 4 == 3):
            # print(predicted_word, lines_count, rhyme_word[lines_count % 4 - 2])
            mask_r = rhyme_mask(rhyme_word[lines_count % 4 - 2], words)
        else:
            mask_r = mask_default

        mask = np.multiply(mask, mask_r)

        encoded = tokenizer.text_to_seq([in_text])[0]
        # truncate sequences to a fixed length

        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')

        # predict probabilities for each word

        prediction = model.predict(encoded)[0]

        predicted_word = get_word(prediction, mask)
        shift = handler.get_next_shift(predicted_word, shift)

        if change_line and (lines_count % 4 == 0 or lines_count % 4 == 1):
            rhyme_word[lines_count % 4] = predicted_word
            # print(f"Rhyme to: {predicted_word}")

        if change_line:
            # rhyme_word = predicted_word

            shift = 0
            lines_count += 1

        # print(f"predicted_word: {predicted_word} next_shift is: {shift} stressed syllable is:\
        #      {handler.get_stress_syllable(predicted_word)}")

        result.append(predicted_word)

        in_text += ' ' + predicted_word
        index += 1
    result = [result[i:i + footness] for i in range(0, len(result), footness)]

    return '\n'.join(list(map(lambda l: ' '.join(l), result)))


def generate_word_list(tokenizer_):
    res = []
    for i in range(len(tokenizer_.word_to_id)):
        res.append(tokenizer_.id_to_word[i])
    return res


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
    mask = np.array([0] * len(words))
    for i, w in enumerate(words):
        if longest_common_suffix(w, word) >= 2:
            mask[i] = 1
    return mask

if __name__ == '__main__':
    # Environment variablestokenizer
    path_to_model = 'model.h5'
    path_to_tokenizer = '../tokenizer.plk'
    path_to_dataset = '../pushkin.txt'
    path_to_stress_model = '~/AutoPoetry/stress_ru.h5'
    path_to_stress_dict = '~/AutoPoetry/zaliznyak.txt'

    # Load model
    model = load_model(path_to_model)

    # TODO: Add tokenizer loading, it's replaced with dataset loading now
    # Load tokenizer dump
    # with open(path_to_tokenizer, encoding='utf-8') as f:
    #    text = f.read()
    with open(path_to_dataset, encoding='utf-8') as f:
        text = f.read()
    text = lang.clean_text(text)
    tokens = word_tokenize(text)
    tokenizer = MyTokenizer(lang.clean_text, word_tokenize)
    tokenizer.fit(text)

    # Load engine
    print('Engine loading...')
    engine = Engine(language='ru')

    # Takes long time
    engine.load(path_to_stress_model, path_to_stress_dict)
    print('Engine loaded')


    # This function would be replaced with out own stress model calling
    def get_stress(word):
        stress = engine.get_stresses(word)
        if not stress:
            return -1

        return stress[0]


    # Words dictionary of dataset
    words = generate_word_list(tokenizer)

    rhythm_handler = rh.RhythmHandler(get_stress, rhythm=(1, 2))
    rhythm_handler.generate_masks()
    masks = rhythm_handler.get_masks_for_words(words)

    # must contain only words from dataset

    seed = ""
    result = generate_seq2(model, 32, seed, 4, masks, rhythm_handler, words, footness=3)
    print(result)

