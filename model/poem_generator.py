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
from model.beam_search import SeqGenerator


def generate_probas(preds, temperature):
    # TODO: rewrite this
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    # probas = np.random.multinomial(1, preds, 1)  # Why multinomial?
    return preds


def get_word(prediction, mask, temperature=1.):
    p = generate_probas(prediction, temperature)
    mmask = np.multiply(p, mask)
    # print(np.max(mmask))
    return tokenizer.id_to_word[np.argmax(mmask)]
    # return tokenizer.id_to_word[np.argmax(p)]

def generating_beam_search(model, seq_len, seed, lines_len, stress_masks, rhythm_handler, words, footness,
                      rhyme_module, rhyme_dict,  beam_size=100):
    """
    current_text = [seed] * beam_size
    shift = [0] * beam_size
    last_words = [{-1: ""}] * beam_size
    for beam in range(beam_size):
        for index in range(rhyme_module):
            last_words[beam][index] = ""
    neutral_mask = np.array([1] * len(words))
    cur_beam = []

    for line_index in range(lines_len):
        # TODO: here you suppose that every word has stress, fix
        stress_count = 0
        shift = [0] * beam_size
        while
        if not cur_beam:
            mask_stress = stress_masks[0]
            mask_rhyme = neutral_mask
            mask = np.multiply(mask_stress, mask_rhyme)

            encoded = tokenizer.text_to_seq([seed])[0]
            # TODO: Padded with zeros! Horrible solution! REDO!!!
            encoded = pad_sequences([encoded], maxlen=seq_len, truncating='pre')

            prediction = model.predict(encoded)[0]
            probas = generate_probas(prediction, temperature=0.7)

            weight_matrix = list(zip(list(map(lambda x: encoded + [tokenizer.word_to_id[x]], words)),
                                     np.log(probas)))
            cur_beam = sorted(weight_matrix, key=lambda x: x[1])[-beam_size:]
            for i, (seqbeam, weight) in enumerate(cur_beam):
                word = tokenizer.id_to_word(seqbeam[-1])
                shift[i] = rhythm_handler.get_next_shift(shift[i], word)
                current_text[i] = current_text[i] + word
            stress_count += 1
            continue
    """
# Greedy
def generate_sequence(model, seq_len, seed, lines_len, stress_masks, rhythm_handler, words, footness,
                      rhyme_module, rhyme_dict):
    current_text = seed
    shift = 0
    last_words = {-1: ""}
    for index in range(rhyme_module):
        last_words[index] = ""
    poem = ""
    neutral_mask = np.array([1] * len(words))

    for line_index in range(lines_len):
        stress_count = 0
        shift = 0
        while stress_count != footness:
            mask_stress = stress_masks[shift]
            # While generation is straightforward
            if footness - 1 == stress_count:
                mask_rhyme = rhyme_mask(last_words[rhyme_dict[line_index % rhyme_module]], words)
            else:
                mask_rhyme = neutral_mask
            mask = np.multiply(mask_stress, mask_rhyme)


            encoded = tokenizer.text_to_seq([current_text])[0]
            # TODO: Padded with zeros! Horrible solution! REDO!!!
            encoded = pad_sequences([encoded], maxlen=seq_len, truncating='pre')

            prediction = model.predict(encoded)[0]

            predicted_word = get_word(prediction, mask, temperature=0.7)
            poem += predicted_word+ ' '
            current_text += ' ' + predicted_word

            shift = rhythm_handler.get_next_shift(predicted_word, shift)

            # Check if stressed
            if rhythm_handler.get_stress_syllable(predicted_word) != -1:
                stress_count += 1
            if footness == stress_count:
                last_words[line_index % rhyme_module] = predicted_word
                poem += '\n'
        if line_index % rhyme_module == rhyme_module - 1:
            poem += '\n'
    return poem


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
    if word == "":
        return np.array([1] * len(words))
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

    rhythm_handler = rh.RhythmHandler(get_stress, rhythm=(0, 2))
    rhythm_handler.generate_masks()
    masks = rhythm_handler.get_masks_for_words(words)

    # must contain only words from dataset
    print(tokenizer.id_to_word[0])

    seed = """Уж темна ночь на небеса всходила,
		Уж в городах утих вседневный шум,
		Луна в окно Монаха осветила. —
		В молитвенник весь устремивший ум,
		Панкратий наш Николы пред иконой
		Со вздохами земные клал поклоны. —
		Пришел Молок (так дьявола зовут),
		Панкратия под черной ряской скрылся.
		Святой Монах молился уж, молился,
		Вздыхал, вздыхал, а дьявол тут как тут.
		Бьет час, Молок не хочет отцепиться,
		Бьет два, бьет три – нечистый всё сидит.
		«Уж будешь мой», – он сам с собой ворчит."""
    seed = seed.lower()
    seed = lang.clean_text(seed)

    generator = SeqGenerator(model, tokenizer, 32, masks, rhythm_handler, words, {0: -1, 1: -1, 2: 0, 3: 1}, 3)
    generator.fit_seed(seed)
    generator.generate_poem(1, footness=2)
