import copy

vowels = 'аеёиоуыэюя'
consonants = 'бвгджзйклмнпрстфхцчшщъь'
alphabet = vowels + consonants


def split_syllables(word):
    syllables = []
    cur_syl_tpl = {
        'syl': [],
        'has_vowel': False
    }
    cur_syl = copy.deepcopy(cur_syl_tpl)
    for i, letter in enumerate(word):
        if cur_syl['has_vowel'] is False:
            cur_syl['syl'].append(letter)
        elif letter in consonants \
                and (((i < len(word) - 1) and word[i + 1] in consonants) or (i == len(word) - 1)):
            cur_syl['syl'].append(letter)
        elif letter in ['ь', 'ъ']:
            cur_syl['syl'].append(letter)
        elif letter == "'":
            cur_syl['syl'].append(letter)
        else:
            syllables.append(''.join(cur_syl['syl']))
            cur_syl = copy.deepcopy(cur_syl_tpl)
            cur_syl['syl'].append(letter)

        if letter in vowels:
            cur_syl['has_vowel'] = True
    syllables.append(''.join(cur_syl['syl']))
    return syllables
