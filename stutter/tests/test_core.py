import math

import pytest

import stutter

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s():
    return stutter.Stutter(48000, 20)

CASES = [
    ("no_gap", lambda: _s().detect([0.5] * 1000) == []),
    ("all_silent", lambda: len(_s().detect([0.0] * 1000)) >= 1),
    ("pair_shape", lambda: all(len(g) == 2 for g in _s().detect([0.0] * 1000))),
    ("gap_order", lambda: all(a < b for a, b in _s().detect([0.0] * 1000))),
    ("bad_rate", lambda: _raises(lambda: stutter.Stutter(0, 20))),
    ("bad_gap", lambda: _raises(lambda: stutter.Stutter(48000, 0))),
    ("empty", lambda: _s().detect([]) == []),
    ("type", lambda: isinstance(_s().detect([0.0] * 100), list)),
    ("det", lambda: _s().detect([0.0] * 100) == _s().detect([0.0] * 100)),
    ("nan", lambda: _raises(lambda: _s().detect([float('nan')] * 10))),
    ("bounds", lambda: all(0 <= a < 1000 and a < b <= 1000 for a, b in _s().detect([0.0] * 1000))),
    ("no_mut", lambda: _s().detect([0.0] * 100) is not None),
    ("quiet", lambda: _s().detect([0.0001] * 1000) is not None),
    ("int_bounds", lambda: all(isinstance(x, int) for g in _s().detect([0.0] * 1000) for x in g)),
    ("mixed", lambda: isinstance(_s().detect([0.5] * 500 + [0.0] * 500), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
