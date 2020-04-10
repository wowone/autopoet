from .. import lingtools


def test_levenshtein_distance():
    lt = lingtools.LingTools()
    assert lt.levenshtein_distance("caba", "aba") == 1
    assert lt.levenshtein_distance("capac", "paca") == 3
    assert lt.levenshtein_distance("", "aaa") == 3
    assert lt.levenshtein_distance("", "") == 0
    assert lt.levenshtein_distance("aba", "aba") == 0
    assert lt.levenshtein_distance("dabababasda", "cabababbacasfds") == lt.levenshtein_distance("cabababbacasfds",
                                                                                                "dabababasda")
    assert lt.levenshtein_distance(["dj", "o", "kak"], ["o", "dj", "ka"]) == 3
    assert lt.levenshtein_distance([1, 2, 3], [1, 2, 3, 5, 5, 5]) == 3
    assert lt.levenshtein_distance("машина", "шина") == 2
    assert lt.levenshtein_distance("дубина", "кабина") == 2
    assert lt.levenshtein_distance("собака", "кошка") == 3
