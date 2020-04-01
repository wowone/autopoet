import numpy as np


class RhythmHandler:
    def __init__(self, get_stress, rhythm=None, skip_not_stressed=True):
        """
        :param get_stress: Функция get_stress(word: String) возвращает позицию ударного гласного или -1
        :param rhythm: размер в формате (ударение, кол-во слогов)
                       Хорей по умолчанию = (0, 2)
                       Ямб                = (1, 2)
                       Дактиль            = (0, 3)
                       Амфибрахий         = (1, 3)
                       Анапест            = (2, 3)
        """
        if rhythm is None:
            rhythm = (1, 2)
        self.rhythm = rhythm

        # Если этот параметр выставлен true, допускается использование слов без ударений
        self.skip_not_stressed = skip_not_stressed

        self.get_stress = get_stress
        self.masks_list = []

        # Считаем, что не бывает слов, где больше, чем 12 слогов
        self.max_syllables = rhythm[1] * 6
        self.vowels = ['и', 'а', 'у', 'е', 'о', 'ы', 'ю', 'я', 'э', 'ё']

    def get_word_vowels(self, word):
        """
        :param word:
        :return: Список позиций гласных слова, нумерация с нуля
        """
        vowels_pos = []
        for i, c in enumerate(word):
            if c in self.vowels:
                vowels_pos.append(i)
        return vowels_pos

    def get_syllables_count(self, word):
        return (len(self.get_word_vowels(word)))

    def generate_masks(self):
        """
        Генерирует список масок для конкретного ритма
        По маске на каждый сдвиг
        Нам нужно несколько масок, потому что
        Количество слогов в слове может отличаться от размерности стиха
        Например:
        из полутЕмной зАлы вдрУг  -- ямб
        По слогам:
        [из] [по][лу][тЕм][ной] [зА][лы] [вдрУг]
          0   0   0    1    0    1   0     1
        полутЕмной заканчивается на безударной позиции, поэтому
        следующее за ним слово должно иметь ударение на нечетной позиции
        """
        for i in range(0, self.rhythm[1]):
            mask = np.zeros(self.max_syllables)
            real_shift = (i + self.rhythm[0]) % self.rhythm[1]
            for syll_ptr in range(0, self.max_syllables):
                if syll_ptr % self.rhythm[1] != real_shift:
                    mask[syll_ptr] = 1
            self.masks_list.append(mask)

    def get_masks_for_stresses(self, stress_list):
        """
        :param stress_list: список ударений для множества слов, ударение в формате: номер ударного слога,
                                                                                    нумерация с нуля
        :return: список масок, в которых единичные значения установлены для слов, попадающих по размеру
        0-ая маска, если размер не сдвинулся
        1-ая маска, если размер сдвинулся на 1
        и т.д
        """
        masks = []
        self.generate_masks()
        for i in range(0, self.rhythm[1]):
            mask = np.zeros(len(stress_list))
            for j, stress in enumerate(stress_list):
                if stress == -1:
                    mask[j] = 0 if self.skip_not_stressed else 1
                    continue
                if stress >= len(self.masks_list[i]):
                    # TODO: Throw exception
                    print('OUT OF SYLLABLES ERROR')
                if self.masks_list[i][stress] == 0:
                    mask[j] = 1
            masks.append(mask)
        return masks

    def get_stress_syllable(self, word):
        """
        :param word:
        :return: номер слога, в котором стоит ударение
        """
        result = 0
        index = self.get_stress(word)
        if index == -1:
            return -1
        for i, c in enumerate(word):
            if c in self.vowels and i < index:
                result += 1
        return result

    def list_to_stressed(self, word_list):
        return list(map(self.get_stress_syllable, word_list))

    def get_masks_for_words(self, word_list):
        return self.get_masks_for_stresses(self.list_to_stressed(word_list))

    def get_next_shift(self, word, prev_shift=0):
        return (prev_shift - self.get_syllables_count(word)) % self.rhythm[1]

