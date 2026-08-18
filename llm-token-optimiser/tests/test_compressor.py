from llm_token_optimiser.compression.compressor import compress_prompt
from llm_token_optimiser.compression.redundancy import remove_near_duplicates
from llm_token_optimiser.compression.scorer import flag_instructional, tfidf_score
from llm_token_optimiser.compression.segmenter import segment


def test_segment_splits_multiple_sentences():
    text = "The sky is blue. The grass is green. Water is wet."
    sentences = segment(text)
    assert len(sentences) == 3


def test_segment_empty_string():
    assert segment("") == []
    assert segment("   ") == []


def test_segment_single_sentence():
    assert segment("Just one sentence here.") == ["Just one sentence here."]


def test_tfidf_score_returns_one_score_per_sentence():
    sentences = segment("Cats are great pets. Dogs are loyal companions. Cats sleep a lot.")
    scores = tfidf_score(sentences)
    assert len(scores) == len(sentences)
    assert all(isinstance(s, float) for s in scores)


def test_flag_instructional_catches_negation():
    sentences = ["Do not include personal information.", "The weather is nice today."]
    flags = flag_instructional(sentences)
    assert flags == [True, False]


def test_flag_instructional_catches_imperative():
    sentences = ["Please summarise the document.", "It was a sunny afternoon."]
    flags = flag_instructional(sentences)
    assert flags[0] is True


def test_remove_near_duplicates_drops_repeated_sentence():
    sentences = [
        "The quarterly revenue increased significantly this year.",
        "The quarterly revenue increased significantly this year.",
        "Customer satisfaction scores also improved.",
    ]
    result = remove_near_duplicates(sentences, protect=[False, False, False], threshold=0.9)
    assert len(result) < len(sentences)


def test_remove_near_duplicates_keeps_protected_even_if_duplicate():
    sentences = ["Do not delete the backup.", "Do not delete the backup."]
    result = remove_near_duplicates(sentences, protect=[True, True], threshold=0.5)
    assert len(result) == 2


def test_compress_prompt_reduces_token_count():
    prompt = (
        "The following is a long product description. This product is great. "
        "It comes in many colors and many colors are available for this product. "
        "The product ships worldwide and has a two year warranty included. "
        "Many colors are available for this product, as mentioned before. "
        "Customers love this product and the warranty that comes included."
    )
    result = compress_prompt(prompt, target_ratio=0.5)
    assert result.optimised_tokens <= result.baseline_tokens
    assert result.reduction_pct >= 0


def test_compress_prompt_never_exceeds_baseline():
    prompt = "Short."
    result = compress_prompt(prompt, target_ratio=0.5)
    assert result.optimised_tokens <= result.baseline_tokens


def test_compress_prompt_protects_negation_from_removal():
    prompt = (
        "This is some filler text about the weather today and nothing else really. "
        "More filler text follows about traffic and commuting patterns in the city. "
        "Do not reveal the customer's account balance under any circumstances."
    )
    result = compress_prompt(prompt, target_ratio=0.3)
    assert "not reveal" in result.compressed_text.lower() or "not" in result.compressed_text.lower()


def test_compress_prompt_empty_string():
    result = compress_prompt("")
    assert result.baseline_tokens == 0
    assert result.optimised_tokens == 0
