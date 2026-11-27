import numpy as np

from semantic_decoding.novic.crop_utils import extract_crop


def test_fractional_padding_scales_with_box_size():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    small = extract_crop(img, np.array([100, 100, 140, 140]), mode="bbox", padding_frac=0.15)
    large = extract_crop(img, np.array([100, 100, 300, 300]), mode="bbox", padding_frac=0.15)
    assert small.shape[:2] == (40 + 2 * 6, 40 + 2 * 6)       # 15% of 40 = 6 px
    assert large.shape[:2] == (200 + 2 * 30, 200 + 2 * 30)   # 15% of 200 = 30 px


def test_pixel_padding_is_used_when_no_fraction_given():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    crop = extract_crop(img, np.array([100, 100, 140, 140]), mode="bbox", padding_px=10)
    assert crop.shape[:2] == (60, 60)
