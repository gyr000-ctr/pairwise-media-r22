import math

import pytest

import cascade

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return cascade.Cascade()

CASES = [
    ("count", lambda: len(_c().splits(0.1, 100.0, 4, 0.5)) == 5),
    ("start", lambda: _c().splits(0.1, 100.0, 4, 0.5)[0] == 0.1),
    ("end", lambda: abs(_c().splits(0.1, 100.0, 4, 0.5)[-1] - 100.0) < 1e-6),
    ("monotonic", lambda: all(a < b for a, b in zip(_c().splits(0.1, 100.0, 4, 0.5), _c().splits(0.1, 100.0, 4, 0.5)[1:]))),
    ("log_ratio", lambda: _c().splits(0.1, 100.0, 4, 0.5)[2] > _c().splits(0.1, 100.0, 4, 0.5)[1] * 1.5),
    ("bad_count", lambda: _raises(lambda: _c().splits(0.1, 100.0, 0, 0.5))),
    ("bad_range", lambda: _raises(lambda: _c().splits(100.0, 0.1, 4, 0.5))),
    ("bad_lambda", lambda: _raises(lambda: _c().splits(0.1, 100.0, 4, 0.0))),
    ("type", lambda: isinstance(_c().splits(0.1, 100.0, 4, 0.5), list)),
    ("det", lambda: _c().splits(0.1, 100.0, 4, 0.5) == _c().splits(0.1, 100.0, 4, 0.5)),
    ("overlap_pair", lambda: _c().overlap(0, 0.1) == _c().overlap(0, 0.1)),
    ("overlap_type", lambda: isinstance(_c().overlap(0, 0.1), tuple)),
    ("one_level", lambda: len(_c().splits(0.1, 100.0, 1, 0.5)) == 2),
    ("nan", lambda: _raises(lambda: _c().splits(float('nan'), 100.0, 4, 0.5))),
    ("no_mut", lambda: _c().splits(0.1, 100.0, 4, 0.5) is not None),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
