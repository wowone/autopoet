from .. import lingtools
import pandas as pd


def test_transcription():
    lt = lingtools.LingTools()
    word = 'борода'
    words = [
        'борода',
        'провода',
        'вражда',
        'борта',
        'вождя',
        'ладах',
        'морс'
    ]

    res = lt.get_rhyme_scores(word, words, debug=True)
    res = pd.DataFrame(res, columns=['word0', 'word1', 'transcription0', 'transcription1', 'score'])
    res = res.sort_values('score')

    assert res['word1'].to_list() == ['борода', 'провода', 'вражда', 'борта', 'вождя', 'ладах', 'морс']
