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
        self.phoneme = None
        self.type = None
        self.soft = None
        self.stressed = None
        self.prev = None
        self.next = None


class LingTools:
    def __init__(self):
        self.vowels = 'аеёиоуыэюя'
        self.consonants = 'бвгджзйклмнпрстфхцчшщъь'

        self.dict = {}
        with open('../yadisk/stresses.csv') as f:
            for row in csv.reader(f, delimiter=','):
                self.dict[row[0]] = {
                    'stressed': row[1],
                    'stressed_syllable': row[2]
                }

        self.vowel_phonemes = pd.read_csv('phonetic_data/vowels.csv', index_col='name', sep=';')

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
        head = prev_item
        n_vowel = 0
        for i, letter in rword[1:]:
            new_item = Node(i, letter)
            if letter in self.vowels:
                new_item.type = 'v'
                n_vowel += 1
                new_item.stressed = (n_vowel == abs(stressed_vowel))
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

    def _get_consonant_phoneme(self, pointer):
        pass

    def get_transcription(self, word):
        stressed_syllable = self.dict[word]['stressed_syllable']
        word = squeeze_sibilants(word)
        head = self._get_nodes(word, stressed_syllable)
        vowel_num = 0, 0
        pointer = head
        length = len(word)
        special = False
        while pointer.next is not None:
            if pointer.type == 'v':
                vowel_num += 1
                phoneme = self._get_vowel_phoneme(pointer, vowel_num, stressed_syllable, length)
                if pointer.yot is True and pointer.index == 0 and pointer.stressed is False:
                    special = True
            elif pointer.type == 'm':
                phoneme = ''
                if pointer.letter == 'ь':
                    pointer.next.soft = True
            elif pointer.type == 'c':
                phoneme = self._get_consonant_phoneme(pointer)
            else:
                phoneme = ''
            pointer.phoneme = phoneme

            pointer = pointer.next

