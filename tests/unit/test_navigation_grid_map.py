import numpy as np

from navigation.core.grid_map import Map


def test_to_pt_to_xy_roundtrip():
    grid = np.zeros((10, 10), dtype=bool)
    m = Map(grid=grid, resolution=0.5, origin=(1.0, 2.0))
    xy = (3.0, 4.0)
    pt = m.to_pt(xy)
    assert pt == (4, 4)  # (3-1)/0.5=4, (4-2)/0.5=4
    back = m.to_xy(pt)
    assert back == (3.0, 4.0)


def test_is_occupied_reads_row_col_order():
    grid = np.zeros((5, 5), dtype=bool)
    grid[2, 3] = True  # grid[y, x]
    m = Map(grid=grid, resolution=1.0, origin=(0.0, 0.0))
    assert m.is_occupied((3, 2)) is True
    assert m.is_occupied((2, 3)) is False
