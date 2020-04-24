import re

import numpy as np
import pandas as pd

import lingtools


def lt_rhyme_scores(words):
    lt = lingtools.LingTools()
    word = 'борода'

    res = lt.get_rhyme_scores(word, words)
    res = pd.DataFrame(res, columns=['word0', 'word1', 'transcription0', 'transcription1', 'score'])
    res = res.sort_values('score')
    return res
    # print(res)

    # assert res['word1'].to_list() == ['борода', 'провода', 'вражда', 'борта', 'вождя', 'ладах', 'морс']


def get_end_part(lt, template):
    if template in lt.stresses:
        stress_position = lt.stresses[template]
    elif template.count('ё'):
        stress_position = template.find('ё')
    else:
        stress_position = np.random.randint(1, len(re.findall('[аеиоуыэюя]', template)) + 1)

    if stress_position == len(template) - 1:
        part = template[-2:]
    else:
        part = template[stress_position:]

    return part


def get_rhyme_scores(template, words, debug=False):
    lt = lingtools.LingTools()

    part = get_end_part(lt, template)

    # print(template, part)

    result = []
    for word in words:
        word_part = get_end_part(lt, word)
        penalty = lt.levenshtein_distance(word_part, part)
        result.append([template, word, part, word_part, penalty])

    if debug is True:
        res = pd.DataFrame(result, columns=['template', 'example', 'end', 'example_end', 'score1'])
        with pd.option_context('display.max_rows', 100):
            print(res.sort_values('score1').head(100))
    return result


def my_rhyme_scores(words):
    word = 'борода'

    res = get_rhyme_scores(word, words)
    res = pd.DataFrame(res, columns=['word0', 'check', 'word_part', 'check_part', 'score2'])
    res = res.sort_values('score2')
    return res


lines = list(open("word_rus.txt", "r", encoding="UTF8").readlines())
words = list(map(lambda x: x[:-1:], lines))[:10]
words = list(filter(lambda x: x.count('-') == 0, words))

first = lt_rhyme_scores(words)

second = my_rhyme_scores(words)

print(first)
print(second)

result = pd.concat([first, second], axis=1)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
print(result)
