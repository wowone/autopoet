import copy
import csv
import numpy as np
import pandas as pd
import re


class Letter:
    def __init__(self, letter=None):
        self.index = None
        self.letter = letter
        self.phoneme = letter
        self.type = None
        self.soft = None
        self.voice = None
        self.stressed = None
        self.prev = None
        self.next = None
        self.yot = None
        self.vowel_num = None


class Word:
    def __init__(self, word, stressed_vowel):
        self.length = len(word)
        self.pointer = self.length - 1
        self.stressed_vowel = stressed_vowel
        self.word = [Letter(letter) for letter in word]

        vowel_num = 0
        for i, l in enumerate(reversed(self.word)):
            l.phoneme = l.letter
            l.index = i
            if l.letter in 'аеёиоуыэюя':
                l.type = 'v'
                vowel_num += 1
                l.vowel_num = vowel_num
                l.stressed = (vowel_num == self.stressed_vowel)
            elif l.letter in 'бвгджзйклмнпрстфхцчшщ':
                l.type = 'c'
            elif l.letter in 'ьъ':
                l.type = 'm'
            else:
                raise ValueError('Unknown character ' + l.letter)

            l.prev = self.word[self.length - i] if i > 0 else None
            l.next = self.word[self.length - i - 2] if i < self.length - 1 else None

    def __reversed__(self):
        for i in range(self.length - 1, -1, -1):
            yield self.word[i]

    def __iter__(self):
        for letter in self.word:
            yield letter


class LingTools:

    def __init__(self):
        self.vowels = 'аеёиоуыэюя'
        self.consonants = 'бвгджзйклмнпрстфхцчшщъь'
        self.vowel_phonemes = pd.read_csv('phonetic_data/vowels.csv', index_col='name', sep=';')
        self.consonant_phonemes = pd.read_csv('phonetic_data/consonants.csv', index_col='name')
        self.phoneme_properties = pd.read_csv('phonetic_data/phoneme_properties.csv', index_col='name')
        self.stop_words = pd.read_csv('phonetic_data/stop_words.txt', header=None)[0].to_list()
        self.phoneme_distance = pd.read_csv('phonetic_data/distance_matrix_jaccard.csv', index_col='name')

        self.sonoric_consonants = ['л', 'м', 'н', 'р', 'й']
        self.noisy_consonants = ['б', 'в', 'г', 'д', 'з', 'ж']
        self.deaf_consonants = ['п', 'ф', 'к', 'т', 'с', 'ш',
                                'х', 'ц', 'ч', 'щ']
        self.special = ['ь', 'ъ']

        # TODO: Remove the heavy file from the repository
        self.stresses = pd.read_csv('../yadisk/stress_data.csv', header=None, index_col=0)[1].to_dict()

    def _get_vowel_index(self, word, index):
        for i in range(index, len(word)):
            if word[i] in self.vowels:
                return i
        return len(word)

    def split_syllables(self, word):
        syllabes = []
        last_index = 0
        while self._get_vowel_index(word, last_index) < len(word):
            position = self._get_vowel_index(word, last_index)
            next_vowel_position = self._get_vowel_index(word, position + 1)

            if next_vowel_position == len(word):
                syll = word[last_index:]
                syllabes.append(syll)
                last_index = len(word)
                continue

            syll = word[last_index: position + 1]
            if next_vowel_position - position > 2:
                if word[position + 2] in self.special:
                    syll += word[position + 1: position + 3]
                    position += 2

            if next_vowel_position - position >= 2 \
                    and word[position + 1] in self.sonoric_consonants \
                    and word[position + 2] in self.deaf_consonants:
                syll += word[position + 1: position + 2]
                position += 1
            elif next_vowel_position - position >= 2 \
                    and word[position + 1].lower() == 'й' \
                    and word[position + 2] in self.consonants:
                syll += word[position + 1: position + 2]
                position += 1

            syllabes.append(syll)
            last_index = position + 1
        return syllabes

    @staticmethod
    def levenshtein_distance(a, b):
        first = ['#'] + list(a)
        second = ['#'] + list(b)
        dp = [[]] * len(first)

        for i in range(len(first)):
            dp[i] = [0] * len(second)

        for i in range(len(first)):
            for j in range(len(second)):
                if i == 0 and j == 0:
                    dp[i][j] = 0
                elif i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1,
                                   dp[i][j - 1] + 1,
                                   dp[i - 1][j - 1] + int(first[i] != second[j]))

        return dp[len(first) - 1][len(second) - 1]

    @staticmethod
    def _squeeze_sibilants(word):
        word = re.sub('(с|ст|сс|з|зд|ж|ш)ч', 'щ', word)
        word = re.sub('(с|зд|з)щ', 'щ', word)
        word = re.sub('(тч|тш|дш)', 'ч', word)
        word = re.sub('(с|з)ш', 'ш', word)
        word = re.sub('сж', 'ж', word)
        word = re.sub('(т|ть|д)с', 'ц', word)
        word = re.sub('(ст|сть)с', 'ц', word)
        return word

    def _insert_yot(self, word, item, position, letter=None):
        if letter is None:
            letter = item.letter

        item.phoneme = self.vowel_phonemes.loc[letter][position]
        if item.letter in ['ю', 'е', 'ё', 'я']:
            if item.index == word.length - 1:
                item.yot = True
            elif item.next is not None and (item.next.type == 'v' or item.next.letter in ['ь', 'ъ']):
                item.yot = True
                if 'vn' in position:
                    item.phoneme = self.vowel_phonemes.loc[letter]['vn_hard']
        elif item.letter in ['и', 'о'] and item.next is not None and item.next.letter == 'ь':
            item.yot = True
            if 'vn' in position:
                item.phoneme = self.vowel_phonemes.loc[letter]['vn_hard']

    def _set_vowel_phoneme(self, word, item):
        letter = item.letter

        # TODO: ударный Е может давать Э: тире -> тирэ, но горе -> гаре
        if item.stressed is True:
            if item.letter not in ['а', 'о', 'у', 'ы'] and item.next is not None and item.next.type == 'c':
                # e.g.: мяч, лист
                if item.letter != 'э' and item.next.letter not in ['ш', 'ж', 'ц']:
                    item.next.soft = True
                # e.g.: желчь, шест
                elif item.letter in ['э', 'е'] and item.next.letter in ['ш', 'ж', 'ц']:
                    letter = 'э'
                # e.g.: жир, цирк
                elif item.letter == 'и':
                    letter = 'ы'
            self._insert_yot(word, item, 'V', letter)

        # начало
        elif item.index == word.length - 1:
            self._insert_yot(word, item, '#', letter)

        # первый предударный
        elif item.vowel_num == word.stressed_vowel + 1:
            # e.g.: живот, шумок
            if item.next is not None and item.next.letter in ['ц', 'ж', 'ш']:
                item.phoneme = self.vowel_phonemes.loc[letter]['v1_sh']
            # e.g.: ведро, вьюнок, связно́й, привет
            elif letter in ['е', 'ё', 'и', 'ю', 'я']:
                item.phoneme = self.vowel_phonemes.loc[letter]['v1_soft']
                item.next.soft = True
            # e.g.: чепец, щавель, майдан
            elif item.next is not None and item.next.letter in ['щ', 'ч', 'й']:
                item.phoneme = self.vowel_phonemes.loc[letter]['v1_soft']
            # e.g.: душа, тосол, наказ, дымо́к
            else:
                item.phoneme = self.vowel_phonemes.loc[letter]['v1_hard']

        # второй предударный
        elif item.vowel_num >= word.stressed_vowel + 2:
            # e.g.: шоколад, циферблат, журавля
            if item.next is not None and item.next.letter in ['ц', 'ж', 'ш']:
                item.phoneme = self.vowel_phonemes.loc[letter]['v2_hard']
            elif item.next is not None and item.next.type == 'v':
                # e.g.: иерархичный, эякуляция
                if letter in ['е', 'ё', 'и', 'ю', 'я']:
                    item.phoneme = self.vowel_phonemes.loc[letter]['v1_soft']
                # e.g.: аэростат, ионизация, аудиенция
                else:
                    item.phoneme = self.vowel_phonemes.loc[letter]['v1_hard']
            # e.g.: сенокос, динамит
            elif letter in ['е', 'ё', 'и', 'ю', 'я']:
                item.phoneme = self.vowel_phonemes.loc[letter]['v2_soft']
                item.next.soft = True
            # e.g.: щупловатый, йодоформ, чаепитие
            elif item.next is not None and item.next.letter in ['щ', 'ч', 'й']:
                item.phoneme = self.vowel_phonemes.loc[letter]['v2_soft']
            # e.g.: гуталин, барабан, ломоносов, крысолов
            else:
                item.phoneme = self.vowel_phonemes.loc[letter]['v2_hard']

        # заударные
        elif item.vowel_num < word.stressed_vowel:
            # e.g.: бежевый, ницца, ковшик
            if item.next is not None and item.next.letter in ['ц', 'ж', 'ш']:
                self._insert_yot(word, item, 'vn_hard')
            elif letter in ['е', 'ё', 'и', 'ю', 'я']:
                # e.g.: дует, аист, заяц
                if item.vowel_num == word.stressed_vowel - 1 and item.next is not None and item.next.type == 'v':
                    self._insert_yot(word, item, 'vn_soft')
                # e.g.: море, гири, бурые, делаю
                else:
                    item.phoneme = self.vowel_phonemes.loc[letter]['vn_soft']
                item.next.soft = True
            # e.g.: общую, мачо, огайо
            elif item.next is not None and item.next.letter in ['щ', 'ч', 'й']:
                item.phoneme = self.vowel_phonemes.loc[letter]['vn_soft']
            else:
                # e.g.: анчоус, какао
                if item.vowel_num == word.stressed_vowel - 1 and item.next is not None and item.next.type == 'v':
                    self._insert_yot(word, item, 'vn_hard')
                # e.g.: руку, тихо, дерево, мачеха
                else:
                    item.phoneme = self.vowel_phonemes.loc[letter]['vn_hard']

    def _set_consonant_phoneme(self, item, vcd):
        if item.letter in ['ч', 'ш', 'щ', 'ж']:
            item.phoneme = self.consonant_phonemes.loc[item.letter]['hard']
            if item.soft is True:
                item.soft = False
            if item.letter in ['ч', 'щ']:
                item.voice = None

        if item.voice is True:
            item.phoneme = self.consonant_phonemes.loc[item.letter]['voiced']
        elif item.voice is False:
            item.phoneme = self.consonant_phonemes.loc[item.letter]['no_voice']

        if vcd is True:
            if item.prev is None:
                item.phoneme = self.consonant_phonemes.loc[item.letter]['voiced']
        else:
            if item.prev is None:
                if item.next is not None:
                    item.next.voice = False
                if self.phoneme_properties.loc[item.phoneme]['vcd'] == '+':
                    item.phoneme = self.consonant_phonemes.loc[item.letter]['no_voice']

        if item.next is not None:
            # оглушение следующих
            if item.phoneme in self.phoneme_properties.index \
                    and self.phoneme_properties.loc[item.phoneme]['vcd'] == '-' \
                    and item.next is not None:
                if item.next.phoneme in self.phoneme_properties.index \
                        and self.phoneme_properties.loc[item.next.phoneme]['son'] == '-':
                    item.next.voice = False
            # озвончение следующих
            elif item.phoneme in self.phoneme_properties.index \
                    and self.phoneme_properties.loc[item.phoneme]['son'] == '-' \
                    and item.phoneme not in ['в', "в’"] \
                    and self.phoneme_properties.loc[item.phoneme]['vcd'] == '+':
                item.next.voice = True

        if item.prev is not None:
            if item.prev.phoneme == 'к' and item.letter == 'г':
                item.phoneme = 'х'
            elif item.prev.letter in [item.letter, item.letter + "’"]:
                item.phoneme = ''

        if item.soft is True:
            item.phoneme = self.consonant_phonemes.loc[item.letter]['soft']

    @staticmethod
    def _isolate_yot_phoneme(transcription):
        phonemes = []
        for phoneme in transcription:
            if re.match('[jṷ]', phoneme) and len(phoneme) > 1:
                phonemes.append(phoneme[0])
                phonemes.append(phoneme[1])
            else:
                phonemes.append(phoneme)
        return phonemes

    def get_transcription(self, word, stress=None, stop=False, vcd=False, get_tail=False, as_string=False):
        if stress is not None:
            stressed_syllable = stress
        elif stop is True:
            stressed_syllable = -1
        elif word in self.stresses:
            stressed_syllable = self.stresses[word]
        else:
            stressed_syllable = np.random.randint(1, len(re.findall('[аеёиоуыэюя]', word)) + 1)

        word_initial = self._squeeze_sibilants(word)
        word = Word(word_initial, stressed_syllable)
        special = False
        for item in reversed(word):
            if item.type == 'v':
                self._set_vowel_phoneme(word, item)
                if item.yot is True and item.index == 0 and item.stressed is False:
                    special = True
            elif item.type == 'm':
                item.phoneme = ''
                if item.letter == 'ь':
                    item.next.soft = True
            elif item.type == 'c':
                if item.letter == 'й':
                    item.phoneme = 'ṷ'
                self._set_consonant_phoneme(item, vcd)
            else:
                raise ValueError('Unknown character ' + item.letter)

            if item.yot is True:
                if special:
                    item.phoneme = 'ṷ' + item.phoneme
                    special = False
                else:
                    item.phoneme = 'j' + item.phoneme

        syllables = self.split_syllables(word_initial)
        tail_length = len(''.join(syllables[-stressed_syllable:]))
        transcription = [w.phoneme for w in word]
        tail = transcription[-tail_length:]

        transcription = self._isolate_yot_phoneme(transcription)
        tail = self._isolate_yot_phoneme(tail)

        if as_string is True:
            transcription = ''.join(transcription)
            tail = ''.join(tail)

        if get_tail is True:
            return transcription, tail
        return transcription

    def get_phrase_transcription(self, phrase, as_string=False):
        res = []
        words = phrase.split(' ')
        for i, word in enumerate(words):
            stop, vcd = False, False
            if word in self.stop_words:
                if len(words) > 1:
                    stop = True
                if i != len(words) - 1 \
                        and word[0] in self.phoneme_properties.index \
                        and self.phoneme_properties.loc[word[0]]['vcd'] == '+':
                    vcd = True
            res.append(self.get_transcription(word, stop=stop, vcd=vcd, as_string=as_string))
        if as_string is False:
            return res
        return ' '.join(res)

    def _set_phoneme_distance(self):
        dist_matrix = []
        for phoneme1 in self.phoneme_properties.index:
            row = []
            coord1 = self.phoneme_properties.loc[phoneme1].dropna()
            for phoneme2 in self.phoneme_properties.index:
                coord2 = self.phoneme_properties.loc[phoneme2].dropna()
                common_index = coord1.index & coord2.index
                distance = 1 - (coord1[common_index] == coord2[common_index]).sum() / max(len(coord1), len(coord2))
                row.append(distance)
            dist_matrix.append(row)
        self.phoneme_distance = pd.DataFrame(dist_matrix,
                                             index=self.phoneme_properties.index,
                                             columns=self.phoneme_properties.index)
        # self.phoneme_distance.to_csv('phonetic_data/distance_matrix_jaccard.csv', float_format='%.3f')

    def get_rhyme_scores(self, word0, words, debug=False):
        transcription0, tail0 = self.get_transcription(word0, get_tail=True)
        result = []
        for word1 in words:
            transcription1, tail1 = self.get_transcription(word1, get_tail=True)
            penalty = 0
            min_len, max_len = min(len(tail0), len(tail1)), max(len(tail0), len(tail1))
            for i in range(min_len):
                penalty += self.phoneme_distance.loc[tail0[i], tail1[i]]
            for i in range(min_len, max_len):
                penalty += 1
            result.append([word0, word1, transcription0, transcription1, penalty])

        if debug is True:
            res = pd.DataFrame(result, columns=['word0', 'word1', 'transcription0', 'transcription1', 'score'])
            with pd.option_context('display.max_rows', 100):
                print(res.sort_values('score').head(100))
        return result

# if __name__ == '__main__':
#     lt = LingTools()
#
#     word = 'борода'
#     words = [
#         'борода',
#         'провода',
#         'вражда',
#         'борта',
#         'вождя',
#         'ладах',
#         'морс'
#     ]
#
#     res = lt.get_rhyme_scores(word, words, debug=True)
