from llm_token_optimiser.compression.pruner import prune_sentence, prune_sentences


def test_prune_sentence_removes_filler_word():
    sentence = "This is basically the main point of the document."
    pruned = prune_sentence(sentence)
    assert "basically" not in pruned.lower()


def test_prune_sentence_removes_filler_phrase():
    sentence = "At the end of the day, the results were positive."
    pruned = prune_sentence(sentence)
    assert "at the end of the day" not in pruned.lower()


def test_prune_sentence_shortens_text():
    sentence = "I feel like this is really quite a very good result, honestly."
    pruned = prune_sentence(sentence)
    assert len(pruned) < len(sentence)


def test_prune_sentence_never_touches_negation_words():
    # prune_sentence() itself has no knowledge of protection — callers
    # are responsible for skipping protected sentences via
    # prune_sentences(). This test just confirms the filler list itself
    # doesn't accidentally overlap with a negation.
    sentence = "Do not basically ignore the deadline."
    pruned = prune_sentence(sentence)
    assert "not" in pruned.lower()


def test_prune_sentence_empty_string():
    assert prune_sentence("") == ""
    assert prune_sentence("   ") == "   "


def test_prune_sentence_never_returns_empty_for_pure_filler():
    sentence = "Basically, really."
    pruned = prune_sentence(sentence)
    assert pruned != ""


def test_prune_sentences_skips_protected_sentences():
    sentences = [
        "This is basically a filler sentence.",
        "Do not, under any circumstances, basically ignore this rule.",
    ]
    protect = [False, True]
    pruned = prune_sentences(sentences, protect)
    assert "basically" not in pruned[0].lower()
    assert pruned[1] == sentences[1]  # protected sentence untouched


def test_prune_sentences_preserves_list_length():
    sentences = ["First sentence here.", "Second sentence here.", "Third one."]
    protect = [False, False, False]
    pruned = prune_sentences(sentences, protect)
    assert len(pruned) == len(sentences)
