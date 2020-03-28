sonoric_consonants = ['л', 'м', 'н', 'р', 'й']
vowels = ['а', 'я', 'е', 'э', 'у', 'ю', 'о', 'ё', 'и', 'ы']
noisy_consonants = ['б', 'в', 'г', 'д', 'з', 'ж']
deaf_consonants = ['п', 'ф', 'к', 'т', 'с', 'ш',
                   'х', 'ц', 'ч', 'щ']
special = ['ь', 'ъ']

consonants = sonoric_consonants + deaf_consonants + noisy_consonants


def is_sonoric_consonant(c):
    return c.lower() in sonoric_consonants
    # 5 symbols


def is_vowel(c):
    return c.lower() in vowels
    # 10 symbols


def is_noisy(c):
    return c.lower() in noisy_consonants
    # 6 symbols


def is_deaf(c):
    return c.lower() in deaf_consonants
    # 10 symbol


def is_special(c):
    return c.lower() in special
    # 2 symbols
