import numpy as np

from semantic_decoding.novic.crop_utils import extract_crop


def _make_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:60, 20:60] = 255
    return img


def test_extract_crop_bbox_mode():
    img = _make_image()
    bbox = np.array([20, 20, 60, 60])
    crop = extract_crop(img, bbox, mask=None, mode="bbox", padding_px=5)
    assert crop.shape[0] == 50 and crop.shape[1] == 50  # 40 + 5 padding each side


def test_extract_crop_segmented_mode_zeroes_outside_mask():
    img = _make_image()
    bbox = np.array([20, 20, 60, 60])
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:50, 30:50] = True

    crop = extract_crop(img, bbox, mask=mask, mode="segmented", padding_px=0)
    assert crop.shape[0] == 40 and crop.shape[1] == 40
    # corner of the crop (bbox-relative) should be zeroed since it's outside the mask
    assert (crop[0, 0] == 0).all()
    # center should retain original pixel values (mask covers it)
    assert (crop[15, 15] == 255).all()


def test_extract_crop_degenerate_bbox_returns_placeholder():
    img = _make_image()
    bbox = np.array([200, 200, 210, 210])  # entirely outside the image
    crop = extract_crop(img, bbox, mask=None, mode="bbox", padding_px=0)
    assert crop.shape == (1, 1, 3)
