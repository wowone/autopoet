from .. import lingtools


def test_transcription():
    lt = lingtools.LingTools()
    # примеры из КнязеваПожарицкой (гл. Фонематическая транскрипция МФШ, стр. 266)
    assert lt.get_transcription("мама") == "мамъ"
    assert lt.get_transcription("тяжесть") == "т’ажъст’"
    assert lt.get_transcription("пиши", stress=1) == "п’ишы"
    assert lt.get_transcription("пищи", stress=2) == "п’иш’ь"
    assert lt.get_transcription("легко") == "л’ихко"
    assert lt.get_transcription("подход") == "патхот"
    assert lt.get_transcription("корзина") == "карз’инъ"
    assert lt.get_transcription("кивая") == "к’иваṷъ"
    assert lt.get_transcription("снегом") == "сн’егъм"
    assert lt.get_transcription("идёшь", stress=1) == "ид’ош"
    assert lt.get_transcription("стакан") == "стакан"
    assert lt.get_transcription("стянуть") == "ст’инут’"
    assert lt.get_transcription("железный") == "жыл’езнъṷ"
    assert lt.get_transcription("железный") == "жыл’езнъṷ"
    assert lt.get_transcription("рассчитавшись") == "ръш’итафшъс’"
    # assert lt.get_transcription("пытаются") == "пытаṷуцъ"
    # assert lt.get_transcription("справедливость") == "спръв’идл’ивъс’т’"
    # assert lt.get_transcription("печенье") == "п’ич’ен’ṷъ"
    # assert lt.get_transcription("шагаешь") == "шагаьш"
    # assert lt.get_transcription("нарочно") == "нарошнъ"

