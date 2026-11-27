import numpy as np

from semantic_decoding.novic.novic_decoder import NOVICDecoder


def test_novic_decoder_falls_back_to_stub_without_checkpoint():
    decoder = NOVICDecoder(checkpoint_path="", device="cpu")
    assert decoder.is_stub


def test_stub_embed_and_decode_roundtrip():
    decoder = NOVICDecoder(checkpoint_path="", device="cpu")
    crops = [np.zeros((32, 32, 3), dtype=np.uint8), np.full((32, 32, 3), 200, dtype=np.uint8)]

    embeddings = decoder.embed_crops(crops)
    assert embeddings.shape == (2, decoder._STUB_EMBED_DIM)

    results = decoder.decode_embeddings(embeddings)
    assert len(results) == 2
    for r in results:
        assert isinstance(r.label, str)
        assert 0.0 <= r.score <= 1.0
        assert r.embedding.shape == (decoder._STUB_EMBED_DIM,)


def test_stub_deterministic_for_same_crop():
    decoder = NOVICDecoder(checkpoint_path="", device="cpu")
    crop = np.full((32, 32, 3), 77, dtype=np.uint8)
    e1 = decoder.embed_crops([crop])
    e2 = decoder.embed_crops([crop])
    assert np.allclose(e1, e2)


def test_empty_input_returns_empty():
    decoder = NOVICDecoder(checkpoint_path="", device="cpu")
    assert decoder.embed_crops([]).shape == (0, 0)
    assert decoder.decode_embeddings(np.empty((0, 0))) == []
