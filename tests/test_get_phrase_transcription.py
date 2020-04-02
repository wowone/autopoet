from .. import lingtools


def test_transcription():
    lt = lingtools.LingTools()
    assert lt.get_phrase_transcription("под") == "пот"
    assert lt.get_phrase_transcription("под сосной") == "път сасноṷ"
    assert lt.get_phrase_transcription("под сосной") == "път сасноṷ"