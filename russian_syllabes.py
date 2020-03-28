from symbols_dictionary import *


def get_vowel_index(word, index):
    for i in range(index, len(word)):
        if is_vowel(word[i]):
            return i
    return len(word)


def split_syllabes(word):
    syllabes = []
    last_index = 0
    while get_vowel_index(word, last_index) < len(word):
        position = get_vowel_index(word, last_index)
        next_vowel_position = get_vowel_index(word, position + 1)

        if next_vowel_position == len(word):
            syll = word[last_index:]
            syllabes.append(syll)
            last_index = len(word)
            continue

        syll = word[last_index: position + 1]
        if next_vowel_position - position > 2:
            if word[position + 2] in special:
                syll += word[position + 1: position + 3]
                position += 2

        if next_vowel_position - position >= 2 \
                and word[position + 1] in sonoric_consonants \
                and word[position + 2] in deaf:
            syll += word[position + 1: position + 2]
            position += 1
        elif next_vowel_position - position >= 2 \
                and word[position + 1].lower() == 'й' \
                and word[position + 2] in consonants:
            syll += word[position + 1: position + 2]
            position += 1

        syllabes.append(syll)
        last_index = position + 1
    return syllabes


def count_vowels(word):
    counter = 0
    for v in word:
        counter += int(v in vowels)
    return counter
