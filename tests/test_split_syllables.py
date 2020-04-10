from .. import lingtools


def get(lt, word):
    return '-'.join(lt.split_syllables(word))


def test_split_syllables():
    lt = lingtools.LingTools()
    assert get(lt, 'женщина') == 'жен-щи-на'
    assert get(lt, 'мужчина') == 'му-жчи-на'
    assert get(lt, 'программа') == 'про-гра-мма'
    assert get(lt, 'война') == 'вой-на'
    assert get(lt, 'вопрос') == 'во-прос'
    assert get(lt, 'неделя') == 'не-де-ля'
    assert get(lt, 'чувство') == 'чу-вство'
    assert get(lt, 'вова') == 'во-ва'
    assert get(lt, 'футбол') == 'фу-тбол'
    assert get(lt, 'песок') == 'пе-сок'
    assert get(lt, 'стол') == 'стол'
    assert get(lt, 'борщ') == 'борщ'
    assert get(lt, 'эхо') == 'э-хо'
    assert get(lt, 'баян') == 'ба-ян'
    assert get(lt, 'пельмень') == 'пель-мень'
    assert get(lt, 'транспорт') == 'тран-спорт'
    assert get(lt, 'упорядочить') == 'у-по-ря-до-чить'
    assert get(lt, 'абракадабра') == 'а-бра-ка-да-бра'
    assert get(lt, 'презумпция') == 'пре-зум-пци-я'
    assert get(lt, 'бельэтаж') == 'бель-э-таж'
    assert get(lt, 'племянник') == 'пле-мя-нник'
    assert get(lt, 'подкаблучник') == 'по-дка-блу-чник'
    assert get(lt, 'праздник') == 'пра-здник'
    assert get(lt, 'пресмыкающееся') == 'пре-смы-ка-ю-ще-е-ся'
    assert get(lt, 'мерзостный') == 'ме-рзо-стный'
    assert get(lt, 'постмодернизм') == 'по-стмо-де-рнизм'
    # assert get('фильтрпресс') == 'филь-трпресс' надо либо принять, что это ок, либо додумать
