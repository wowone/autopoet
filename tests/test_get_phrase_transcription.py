from .. import lingtools


def test_transcription():
    lt = lingtools.LingTools()
    assert lt.get_phrase_transcription("под", as_string=True) == "пот"
    assert lt.get_phrase_transcription("под сосной", as_string=True) == "път сасноṷ"