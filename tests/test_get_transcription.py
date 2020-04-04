from .. import lingtools


def test_transcription():
    lt = lingtools.LingTools()
    # примеры из КнязеваПожарицкой (гл. Фонематическая транскрипция МФШ, стр. 266)
    assert lt.get_transcription("мама", as_string=True) == "мамъ"
    assert lt.get_transcription("тяжесть", as_string=True) == "т’ажъст’"
    assert lt.get_transcription("пиши", stress=1, as_string=True) == "п’ишы"
    assert lt.get_transcription("пищи", stress=2, as_string=True) == "п’иш’ь"
    assert lt.get_transcription("легко", as_string=True) == "л’ихко"
    assert lt.get_transcription("подход", as_string=True) == "патхот"
    assert lt.get_transcription("корзина", as_string=True) == "карз’инъ"
    assert lt.get_transcription("кивая", as_string=True) == "к’иваṷъ"
    assert lt.get_transcription("снегом", as_string=True) == "сн’егъм"
    assert lt.get_transcription("идёшь", stress=1, as_string=True) == "ид’ош"
    assert lt.get_transcription("стакан", as_string=True) == "стакан"
    assert lt.get_transcription("стянуть", as_string=True) == "ст’инут’"
    assert lt.get_transcription("железный", as_string=True) == "жыл’езнъṷ"
    assert lt.get_transcription("железный", as_string=True) == "жыл’езнъṷ"
    assert lt.get_transcription("рассчитавшись", as_string=True) == "ръш’итафшъс’"
    # assert lt.get_transcription("пытаются", as_string=True) == "пытаṷуцъ"
    # assert lt.get_transcription("справедливость", as_string=True) == "спръв’идл’ивъс’т’"
    # assert lt.get_transcription("печенье", as_string=True) == "п’ич’ен’ṷъ"
    # assert lt.get_transcription("шагаешь", as_string=True) == "шагаьш"
    # assert lt.get_transcription("нарочно", as_string=True) == "нарошнъ"

