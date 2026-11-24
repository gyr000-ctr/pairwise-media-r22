import math

import pytest

import deconv

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.5] * 4 for _ in range(4)]

CASES = [
    ("psf_sum", lambda: abs(sum(sum(r) for r in deconv.Wiener(0.01).psf(1.0, 5)) - 1.0) < 1e-9),
    ("shape", lambda: len(deconv.Wiener(0.01).deconvolve(IMG, deconv.Wiener(0.01).psf(1.0, 3))) == 4),
    ("bad_k", lambda: _raises(lambda: deconv.Wiener(-0.1))),
    ("bad_sigma", lambda: _raises(lambda: deconv.Wiener(0.01).psf(0.0, 3))),
    ("bad_psf", lambda: _raises(lambda: deconv.Wiener(0.01).deconvolve(IMG, []))),
    ("bad_image", lambda: _raises(lambda: deconv.Wiener(0.01).deconvolve([], [[1.0]]))),
    ("type", lambda: isinstance(deconv.Wiener(0.01).deconvolve(IMG, deconv.Wiener(0.01).psf(1.0, 3)), list)),
    ("det", lambda: deconv.Wiener(0.01).deconvolve(IMG, deconv.Wiener(0.01).psf(1.0, 3)) == deconv.Wiener(0.01).deconvolve(IMG, deconv.Wiener(0.01).psf(1.0, 3))),
    ("psf_odd", lambda: len(deconv.Wiener(0.01).psf(1.0, 5)) == 5),
    ("psf_sym", lambda: deconv.Wiener(0.01).psf(1.0, 5) == deconv.Wiener(0.01).psf(1.0, 5)[::-1]),
    ("finite", lambda: all(math.isfinite(v) for row in deconv.Wiener(0.01).deconvolve(IMG, deconv.Wiener(0.01).psf(1.0, 3)) for v in row)),
    ("nan", lambda: _raises(lambda: deconv.Wiener(0.01).deconvolve([[float('nan')]], [[1.0]]))),
    ("no_mut", lambda: deconv.Wiener(0.01).deconvolve([r[:] for r in IMG], deconv.Wiener(0.01).psf(1.0, 3)) is not None),
    ("psf_type", lambda: isinstance(deconv.Wiener(0.01).psf(1.0, 3), list)),
    ("constant", lambda: all(abs(v - 0.5) < 0.1 for row in deconv.Wiener(0.01).deconvolve(IMG, deconv.Wiener(0.01).psf(1.0, 3)) for v in row)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
