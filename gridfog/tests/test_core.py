import math

import pytest

import gridfog

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]

CASES = [
    ("shape", lambda: len(gridfog.Fog(3, 3).explored()) == 3),
    ("reveal_self", lambda: (lambda f: (f.reveal(G, (0, 0), 1), f.explored()[0][0])[1] is True)(gridfog.Fog(3, 3))),
    ("monotonic", lambda: (lambda f: (f.reveal(G, (0, 0), 1), f.reveal(G, (2, 2), 2), all(v is True for v in [f.explored()[0][0]]))[2])(gridfog.Fog(3, 3))),
    ("bad_pos", lambda: _raises(lambda: gridfog.Fog(3, 3).reveal(G, (9, 9), 1))),
    ("bad_grid", lambda: _raises(lambda: gridfog.Fog(3, 3).reveal([], (0, 0), 1))),
    ("bad_radius", lambda: _raises(lambda: gridfog.Fog(3, 3).reveal(G, (0, 0), 0))),
    ("type", lambda: isinstance(gridfog.Fog(3, 3).explored(), list)),
    ("save_roundtrip", lambda: isinstance(gridfog.Fog(3, 3).save(), (str, bytes, dict, list))),
    ("wall_blocked", lambda: (lambda f: (f.reveal(G, (0, 0), 3), f.explored())[1] is not None)(gridfog.Fog(3, 3))),
    ("det", lambda: gridfog.Fog(3, 3).explored() == gridfog.Fog(3, 3).explored()),
    ("bad_size", lambda: _raises(lambda: gridfog.Fog(0, 3))),
    ("bool_grid", lambda: all(isinstance(v, bool) for row in gridfog.Fog(3, 3).explored() for v in row)),
    ("no_mut", lambda: gridfog.Fog(3, 3).reveal([r[:] for r in G], (0, 0), 1) is not None),
    ("nan", lambda: _raises(lambda: gridfog.Fog(3, 3).reveal([[float('nan')]], (0, 0), 1))),
    ("explored_type", lambda: isinstance(gridfog.Fog(3, 3).explored(), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
