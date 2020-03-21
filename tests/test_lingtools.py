from .. import lingtools as lt


def test_split_syllables():
    assert '-'.join(lt.split_syllables('вова')) == 'во-ва'
    assert '-'.join(lt.split_syllables('футбол')) == 'фут-бол'
    assert '-'.join(lt.split_syllables('песок')) == 'пе-сок'
    assert '-'.join(lt.split_syllables('стол')) == 'стол'
    assert '-'.join(lt.split_syllables('борщ')) == 'борщ'
    assert '-'.join(lt.split_syllables('эхо')) == 'э-хо'
    assert '-'.join(lt.split_syllables('баян')) == 'ба-ян'
    assert '-'.join(lt.split_syllables('пельмень')) == 'пель-мень'
    assert '-'.join(lt.split_syllables('транспорт')) == 'транс-порт'
    assert '-'.join(lt.split_syllables('упорядочить')) == 'у-по-ря-до-чить'
    assert '-'.join(lt.split_syllables('абракадабра')) == 'аб-ра-ка-даб-ра'
    assert '-'.join(lt.split_syllables('презумпция')) == 'пре-зумп-ци-я'
    assert '-'.join(lt.split_syllables('бельэтаж')) == 'бель-э-таж'
    assert '-'.join(lt.split_syllables('племянник')) == 'пле-мян-ник'
    assert '-'.join(lt.split_syllables('подкаблучник')) == 'под-каб-луч-ник'
    assert '-'.join(lt.split_syllables('праздник')) == 'празд-ник'
    assert '-'.join(lt.split_syllables('пресмыкающееся')) == 'прес-мы-ка-ю-ще-е-ся'
    assert '-'.join(lt.split_syllables('мерзостный')) == 'мер-зост-ный'
    assert '-'.join(lt.split_syllables('постмодернизм')) == 'пост-мо-дер-низм'
    # assert '-'.join(lt.split_syllables('представление')) == 'пред-став-ле-ни-е'
