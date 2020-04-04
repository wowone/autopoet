from russian_syllabes.russian_syllabes_lib import split_syllabes


def get(word):
    return '-'.join(split_syllabes(word))


def do_check():
    assert get('женщина') == 'жен-щи-на'
    assert get('мужчина') == 'му-жчи-на'
    assert get('программа') == 'про-гра-мма'
    assert get('война') == 'вой-на'
    assert get('вопрос') == 'во-прос'
    assert get('неделя') == 'не-де-ля'
    assert get('чувство') == 'чу-вство'
    # assert get('фильтрпресс') == 'филь-трпресс' надо либо принять, что это ок, либо додумать

    print("All correct")


do_check()
