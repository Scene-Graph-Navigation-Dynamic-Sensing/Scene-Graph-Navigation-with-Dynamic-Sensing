from math import isclose

from perception.depth_estimation.depth_anything_estimator import da3_metric_scale


def test_canonical_focal_gives_unit_scale():
    assert isclose(da3_metric_scale((480, 640), (480, 640), fx=300, fy=300), 1.0)


def test_focal_is_measured_at_the_depth_resolution():
    # the network ran at half the input resolution, so the focal it saw is halved
    assert isclose(da3_metric_scale((240, 320), (480, 640), fx=600, fy=600), 1.0)


def test_fx_and_fy_are_averaged():
    assert isclose(da3_metric_scale((480, 640), (480, 640), fx=300, fy=600), 1.5)
