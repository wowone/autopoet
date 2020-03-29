import copy
import csv
import pandas as pd
import re


def squeeze_sibilants(word):
    word = re.sub('(с|ст|сс|з|зд|ж|ш)ч', 'щ', word)
    word = re.sub('(с|зд|з)щ', 'щ', word)
    word = re.sub('(тч|тш|дш)', 'ч', word)
    word = re.sub('(с|з)ш', 'ш', word)
    word = re.sub('сж', 'ж', word)
    word = re.sub('(т|ть|д)с', 'ц', word)
    word = re.sub('(ст|сть)с', 'ц', word)
    return word


class Node:
    def __init__(self, index=None, letter=None):
        self.index = index
        self.letter = letter
        self.phoneme = letter
        self.type = None
        self.soft = None
        self.voice = None
        self.stressed = None
        self.prev = None
        self.next = None
        self.yot = None


class LingTools:
    def __init__(self):
        self.vowels = 'аеёиоуыэюя'
        self.consonants = 'бвгджзйклмнпрстфхцчшщъь'
        self.vowel_phonemes = pd.read_csv('phonetic_data/vowels.csv', index_col='name', sep=';')
        self.consonant_phonemes = pd.read_csv('phonetic_data/cons_.csv', index_col='name')
        self.phoneme_properties = pd.read_csv('phonetic_data/phoneme_properties.csv', index_col='name')

        self.dict = {}
        with open('../yadisk/stress_data.csv') as f:
            for row in csv.reader(f, delimiter=','):
                self.dict[row[0]] = {
                    # 'stressed': row[1],
                    'stressed_syllable': abs(int(row[1]))
                }

    def split_syllables(self, word):
        syllables = []
        current_syllable_template = {
            'syl': [],
            'has_vowel': False
        }
        current_syllable = copy.deepcopy(current_syllable_template)
        for i, letter in enumerate(word):
            if current_syllable['has_vowel'] is False:
                current_syllable['syl'].append(letter)
            elif letter in self.consonants \
                    and (((i < len(word) - 1) and word[i + 1] in self.consonants) or (i == len(word) - 1)):
                current_syllable['syl'].append(letter)
            elif letter in ['ь', 'ъ']:
                current_syllable['syl'].append(letter)
            elif letter == "'":
                current_syllable['syl'].append(letter)
            else:
                syllables.append(''.join(current_syllable['syl']))
                current_syllable = copy.deepcopy(current_syllable_template)
                current_syllable['syl'].append(letter)

            if letter in self.vowels:
                current_syllable['has_vowel'] = True
        syllables.append(''.join(current_syllable['syl']))
        return syllables

    def _get_nodes(self, word, stressed_vowel):
        rword = list(reversed(word))
        prev_item = Node(0, rword[0])
        if rword[0] in self.vowels:
            prev_item.type = 'v'
        elif rword[0] in self.consonants:
            prev_item.type = 'c'
        elif rword[0] in ['ь', 'ъ']:
            prev_item.type = 'm'
        else:
            raise ValueError('Unknown token ' + rword[0])
        head = prev_item
        n_vowel = 1 if rword[0] in self.vowels else 0
        for i, letter in enumerate(rword[1:]):
            new_item = Node(i + 1, letter)
            if letter in self.vowels:
                new_item.type = 'v'
                n_vowel += 1
                new_item.stressed = (n_vowel == stressed_vowel)
            elif letter in self.consonants:
                new_item.type = 'c'
            elif letter in ['ь', 'ъ']:
                new_item.type = 'm'
            else:
                raise ValueError('Unknown token "' + letter + '"')
            prev_item.next = new_item
            new_item.prev = prev_item
            prev_item = new_item
        return head

    def _insert_yot(self, p, letter, position, length):
        p.phoneme = self.vowel_phonemes.loc[letter][position]
        if p.letter in ['ю', 'е', 'ё', 'я']:
            if p.index == length - 1:
                p.yot = True
            elif p.next is not None and (p.next.type == 'v' or p.next.letter in ['ь', 'ъ']):
                p.yot = True
                if 'vn' in position:
                    p.phoneme = self.vowel_phonemes.loc[letter]['vn_hard']
        elif p.letter in ['и', 'о'] and p.next is not None and p.next.letter == 'ь':
            p.yot = True
            if 'vn' in position:
                p.phoneme = self.vowel_phonemes.loc[letter]['vn_hard']

    def _get_vowel_phoneme(self, p, vowel_num, stressed_syllable, length):
        letter = p.letter

        # TODO: ударный Е может давать Э: тире -> тирэ, но горе -> гаре
        if p.stressed is True:
            if p.letter not in ['а', 'о', 'у', 'ы'] and p.next is not None and p.next.type == 'c':
                # e.g.: мяч, лист
                if p.letter != 'э' and p.next.letter not in ['ш', 'ж', 'ц']:
                    p.next.soft = True
                # e.g.: желчь, шест
                elif p.letter in ['э', 'е'] and p.next.letter in ['ш', 'ж', 'ц']:
                    letter = 'э'
                # e.g.: жир, цирк
                elif p.letter == 'и':
                    letter = 'ы'
            self._insert_yot(p, letter, 'V', length)

        # начало
        elif p.index == length - 1:
            self._insert_yot(p, letter, '#', length)

        # первый предударный
        elif vowel_num == stressed_syllable + 1:
            # e.g.: живот, шумок
            if p.next is not None and p.next.letter in ['ц', 'ж', 'ш']:
                p.phoneme = self.vowel_phonemes.loc[letter]['v1_sh']
            # e.g.: ведро, вьюнок, связно́й, привет
            elif letter in ['е', 'ё', 'и', 'ю', 'я']:
                p.phoneme = self.vowel_phonemes.loc[letter]['v1_soft']
                p.next.soft = True
            # e.g.: чепец, щавель, майдан
            elif p.next is not None and p.next.letter in ['щ', 'ч', 'й']:
                p.phoneme = self.vowel_phonemes.loc[letter]['v1_soft']
            # e.g.: душа, тосол, наказ, дымо́к
            else:
                p.phoneme = self.vowel_phonemes.loc[letter]['v1_hard']

        # второй предударный
        elif vowel_num >= stressed_syllable + 2:
            # e.g.: шоколад, циферблат, журавля
            if p.next is not None and p.next.letter in ['ц', 'ж', 'ш']:
                p.phoneme = self.vowel_phonemes.loc[letter]['v2_hard']
            elif p.next is not None and p.next.type == 'v':
                # e.g.: иерархичный, эякуляция
                if letter in ['е', 'ё', 'и', 'ю', 'я']:
                    p.phoneme = self.vowel_phonemes.loc[letter]['v1_soft']
                # e.g.: аэростат, ионизация, аудиенция
                else:
                    p.phoneme = self.vowel_phonemes.loc[letter]['v1_hard']
            # e.g.: сенокос, динамит
            elif letter in ['е', 'ё', 'и', 'ю', 'я']:
                p.phoneme = self.vowel_phonemes.loc[letter]['v2_soft']
                p.next.soft = True
            # e.g.: щупловатый, йодоформ, чаепитие
            elif p.next is not None and p.next.letter in ['щ', 'ч', 'й']:
                p.phoneme = self.vowel_phonemes.loc[letter]['v2_soft']
            # e.g.: гуталин, барабан, ломоносов, крысолов
            else:
                p.phoneme = self.vowel_phonemes.loc[letter]['v2_hard']

        # заударные
        elif vowel_num < stressed_syllable:
            # e.g.: бежевый, ницца, ковшик
            if p.next is not None and p.next.letter in ['ц', 'ж', 'ш']:
                self._insert_yot(p, letter, 'vn_hard', length)
            elif letter in ['е', 'ё', 'и', 'ю', 'я']:
                # e.g.: дует, аист, заяц
                if vowel_num == stressed_syllable - 1 and p.next is not None and p.next.type == 'v':
                    self._insert_yot(p, letter, 'vn_soft', length)
                # e.g.: море, гири, бурые, делаю
                else:
                    p.phoneme = self.vowel_phonemes.loc[letter]['vn_soft']
                p.next.soft = True
            # e.g.: общую, мачо, огайо
            elif p.next is not None and p.next.letter in ['щ', 'ч', 'й']:
                p.phoneme = self.vowel_phonemes.loc[letter]['vn_soft']
            else:
                # e.g.: анчоус, какао
                if vowel_num == stressed_syllable - 1 and p.next is not None and p.next.type == 'v':
                    self._insert_yot(p, letter, 'vn_hard', length)
                # e.g.: руку, тихо, дерево, мачеха
                else:
                    p.phoneme = self.vowel_phonemes.loc[letter]['vn_hard']

    def _get_consonant_phoneme(self, p, vcd):
        if p.letter in ['ч', 'ш', 'щ', 'ж']:
            p.phoneme = self.consonant_phonemes.loc[p.letter]['hard']
            if p.soft is True:
                p.soft = False
            if p.letter in ['ч', 'щ']:
                if p.voice is not None:
                    p.voice = None
        if p.voice is True:
            p.phoneme = self.consonant_phonemes.loc[p.letter]['voice']
        elif p.voice is False:
            p.phoneme = self.consonant_phonemes.loc[p.letter]['no_voice']

        if vcd is True:
            if p.prev is None:
                p.phoneme = self.consonant_phonemes.loc[p.letter]['voiced']
        else:
            if p.prev is None:
                if p.next is not None:
                    p.next.voice = False
                if self.phoneme_properties.loc[p.phoneme]['vcd'] == '+':
                    p.phoneme = self.consonant_phonemes.loc[p.letter]['no_voice']

        if p.next is not None:
            # оглушение следующих
            if p.phoneme in self.phoneme_properties.index\
                    and self.phoneme_properties.loc[p.phoneme]['vcd'] == '-'\
                    and p.next is not None:
                if p.next.phoneme in self.phoneme_properties.index\
                        and self.phoneme_properties.loc[p.next.phoneme]['son'] == '-':
                    p.next.voice = False
            # озвончение следующих
            elif p.phoneme in self.phoneme_properties.index\
                    and self.phoneme_properties.loc[p.phoneme]['son'] == '-'\
                    and p.phoneme not in ['в', "в’"]\
                    and self.phoneme_properties.loc[p.phoneme]['vcd'] == '+':
                p.next.voice = True

        if p.prev is not None:
            if p.prev.phoneme == 'к' and p.letter == 'г':
                p.phoneme = 'х'
            elif p.prev.letter in [p.letter, p.letter + "’"]:
                p.phoneme = ''

    @staticmethod
    def _get_transcription_from_structure(p):
        res = []
        while p is not None:
            res.append(p.phoneme)
            p = p.next
        return res

    def get_transcription(self, word, vcd=False):
        stressed_syllable = self.dict[word]['stressed_syllable']
        word = squeeze_sibilants(word)
        head = self._get_nodes(word, stressed_syllable)
        vowel_num = 0
        pointer = head
        length = len(word)
        special = False
        while pointer is not None:
            if pointer.type == 'v':
                vowel_num += 1
                self._get_vowel_phoneme(pointer, vowel_num, stressed_syllable, length)
                if pointer.yot is True and pointer.index == 0 and pointer.stressed is False:
                    special = True
            elif pointer.type == 'm':
                pointer.phoneme = ''
                if pointer.letter == 'ь':
                    pointer.next.soft = True
            elif pointer.type == 'c':
                if pointer.letter == 'й':
                    pointer.phoneme = 'ṷ'
                self._get_consonant_phoneme(pointer, vcd)
            else:
                raise ValueError('Unknown character ' + pointer.letter)

            if pointer.yot is True:
                if special:
                    pointer.phoneme = pointer.phoneme + 'ṷ'
                    special = False
                else:
                    pointer.phoneme = pointer.phoneme + 'j'

            pointer = pointer.next
        return self._get_transcription_from_structure(head)


# lt = LingTools()
# print(lt.get_transcription('мама'))

import sys
if __name__ == '__main__':
    lt = LingTools()
    if len(sys.argv) >= 3:
        print(lt.get_transcription(sys.argv[1], stress=int(sys.argv[2])))
    else:
        print(lt.get_transcription(sys.argv[1]))
