import math

import pytest

import uvproject

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _u(s=1.0):
    return uvproject.UvProject(s)

CASES = [
    ("sum_one", lambda: abs(sum(_u().weights((1.0, 0.0, 0.0))) - 1.0) < 1e-12),
    ("axis_plane", lambda: _u().weights((0.0, 0.0, 1.0))[2] > 0.99),
    ("axis_x", lambda: _u().weights((1.0, 0.0, 0.0))[0] > 0.99),
    ("continuity", lambda: max(abs(a - b) for a, b in zip(_u().weights((0.7, 0.7, 0.1)), _u().weights((0.7001, 0.6999, 0.1)))) < 1e-2),
    ("bad_normal", lambda: _raises(lambda: _u().weights((0.0, 0.0)))),
    ("zero_normal", lambda: _raises(lambda: _u().weights((0.0, 0.0, 0.0)))),
    ("bad_sharp", lambda: _raises(lambda: uvproject.UvProject(0.0))),
    ("type", lambda: isinstance(_u().weights((0.1, 0.9, 0.0)), tuple)),
    ("range", lambda: all(0.0 <= w <= 1.0 for w in _u().weights((0.3, 0.6, 0.1)))),
    ("det", lambda: _u().weights((0.2, 0.3, 0.4)) == _u().weights((0.2, 0.3, 0.4))),
    ("uv_pair", lambda: len(_u().uv((0.5, 0.5, 0.5), (0.0, 0.0, 1.0))) == 2),
    ("sharp_effect", lambda: _u(1.0).weights((0.6, 0.8, 0.0)) != _u(4.0).weights((0.6, 0.8, 0.0))),
    ("nan", lambda: _raises(lambda: _u().weights((float('nan'), 0.0, 0.0)))),
    ("no_mut", lambda: _u().uv((0.1, 0.2, 0.3), (0.0, 1.0, 0.0)) is not None),
    ("uv_float", lambda: isinstance(_u().uv((0.5, 0.5, 0.5), (1.0, 0.0, 0.0))[0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
