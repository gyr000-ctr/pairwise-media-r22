"""题面约束的本地回归：边界 <=20ms、误报 <=2%、双门限迟滞、性能、
可区分错误、并发安全。全部用例确定、可重复。

与 tests/test_core.py 的 15 个规格用例互补，不修改既有用例。
"""

import math
import threading
import time

import pytest

import stutter
from stutter import Stutter

np = pytest.importorskip("numpy")

SR = 48000
TOL = int(0.020 * SR)  # 20ms -> 960 samples


def _tone(n, sr=SR, amp=0.4, f=440):
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * f * t)


def _gapped(gaps_ms, sr=SR, seconds=5.0, kind="tone", seed=0):
    n = int(seconds * sr)
    rng = np.random.default_rng(seed)
    if kind == "noise":
        x = 0.3 * rng.standard_normal(n)
    elif kind == "chord":
        t = np.arange(n) / sr
        x = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 659 * t)
    else:
        x = _tone(n, sr)
    truth = []
    for at_ms, dur_ms in gaps_ms:
        s = int(at_ms / 1000 * sr)
        e = s + int(dur_ms / 1000 * sr)
        x[s:e] = 0.0
        truth.append((s, e))
    return x.tolist(), truth


# ---------------------------------------------------------------------------
# 边界误差 <=20ms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["tone", "chord", "noise"])
def test_zero_gap_boundaries_within_20ms(kind):
    gaps = [(500, 120), (1500, 60), (2600, 200), (3800, 40)]
    x, truth = _gapped(gaps, kind=kind)
    out = Stutter(SR, 20).detect(x)
    assert len(out) == len(truth)
    for ts, te in truth:
        cand = [g for g in out if abs(g[0] - ts) <= TOL and abs(g[1] - te) <= TOL]
        assert cand, f"truth=({ts},{te}) not matched by {out}"


@pytest.mark.parametrize("sr", [48000, 44100, 16000])
def test_boundaries_across_sample_rates(sr):
    tol = int(0.020 * sr)
    n = 4 * sr
    x = 0.4 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)
    s1, e1 = int(0.5 * sr), int(0.62 * sr)
    s2, e2 = int(2.2 * sr), int(2.3 * sr)
    x[s1:e1] = 0.0
    x[s2:e2] = 0.0
    out = Stutter(sr, 20).detect(x.tolist())
    for s, e in ((s1, e1), (s2, e2)):
        assert any(abs(a - s) <= tol and abs(b - e) <= tol for a, b in out), out


def test_energy_collapse_nonzero_gap_within_20ms():
    n = 5 * SR
    x = _tone(n)
    truth = [(int(0.7 * SR), int(0.85 * SR)), (int(2.5 * SR), int(2.7 * SR))]
    for s, e in truth:
        x[s:e] *= 0.05  # 能量突变式塌陷，但不是精确零
    out = Stutter(SR, 20).detect(x.tolist())
    for s, e in truth:
        assert any(abs(a - s) <= TOL and abs(b - e) <= TOL for a, b in out), out


def test_exact_gap_ms_length_detected_sample_accurate():
    n = 2 * SR
    x = _tone(n)
    s = SR // 2
    length = int(SR * 20 / 1000)  # 恰好 gap_ms
    x[s:s + length] = 0.0
    out = Stutter(SR, 20).detect(x.tolist())
    assert any(abs(a - s) <= 2 and abs(b - (s + length)) <= 2 for a, b in out), out


# ---------------------------------------------------------------------------
# 正常流误报 <=2%
# ---------------------------------------------------------------------------


def test_normal_stream_false_positive_rate_below_2pct():
    sr = SR
    n = 8 * sr
    t = np.arange(n) / sr
    rng = np.random.default_rng(20240501)
    clips = [
        0.5 * np.ones(n),
        0.4 * np.sin(2 * np.pi * 440 * t),
        0.5 * np.sin(2 * np.pi * 80 * t),
        0.25 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 659 * t),
        0.3 * rng.standard_normal(n),
        0.08 * rng.standard_normal(n),
        0.25 * np.sin(2 * np.pi * 440 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t)),
        np.full(n, 1e-4),
    ]
    det = Stutter(sr, 20)
    for x in clips:
        out = det.detect(x.tolist())
        false_samples = sum(b - a for a, b in out)
        assert false_samples / n <= 0.02, f"FPR={false_samples / n:.3%} segs={out}"
        assert len(out) <= 2


# ---------------------------------------------------------------------------
# 双门限迟滞：门限附近抖动不应碎成很多段，且结果确定
# ---------------------------------------------------------------------------


def test_hysteresis_suppresses_chatter():
    sr = 1000
    x = np.full(4000, 0.5)
    wobble = 0.02 + 0.004 * np.tile([1.0, -1.0, 0.5, -0.5], 500)[:2000]
    x[1000:3000] = wobble
    det = Stutter(sr, 20)
    first = det.detect(x.tolist())
    second = det.detect(x.tolist())
    assert first == second
    assert 1 <= len(first) <= 3
    assert all(0 <= a < b <= 4000 for a, b in first)


# ---------------------------------------------------------------------------
# 性能：100 万样本 <=500ms
# ---------------------------------------------------------------------------


def test_perf_1m_samples_under_500ms():
    x = np.random.default_rng(0).standard_normal(1_000_000).tolist()
    det = Stutter(48000, 20)
    best = min(_timed(det, x) for _ in range(7))
    assert best < 0.5, f"100 万样本最佳耗时 {best * 1000:.0f}ms 超过 500ms"


def test_perf_1m_zero_gaps_under_500ms():
    x = _tone(1_000_000)
    for k in range(20):
        s = k * 50000 + 10000
        x[s:s + 2400] = 0.0
    x = x.tolist()
    det = Stutter(48000, 20)
    best = min(_timed(det, x) for _ in range(5))
    assert best < 0.5, f"含缺口 100 万样本最佳耗时 {best * 1000:.0f}ms"


def _timed(det, x):
    t0 = time.perf_counter()
    det.detect(x)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 并发安全：共享同一实例
# ---------------------------------------------------------------------------


def test_concurrent_detect_shared_instance():
    sigs = [_gapped([(400 + i * 200, 80)], kind=k)[0]
            for i, k in enumerate(("tone", "noise", "chord"))]
    det = Stutter(48000, 20)
    expected = [det.detect(s) for s in sigs]
    errors = []
    results = [None] * 8

    def worker(k):
        try:
            for _ in range(20):
                results[k] = det.detect(sigs[k % len(sigs)])
                if results[k] != expected[k % len(sigs)]:
                    errors.append("inconsistent result")
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:2]
    assert all(r is not None for r in results)


# ---------------------------------------------------------------------------
# 可区分报错
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_bad_rate_raises_value_error(rate):
    with pytest.raises(ValueError):
        Stutter(rate, 20)


@pytest.mark.parametrize("gap", [0, -1, float("nan"), float("inf")])
def test_bad_gap_raises_value_error(gap):
    with pytest.raises(ValueError):
        Stutter(48000, gap)


def test_bad_type_raises_type_and_value_error():
    with pytest.raises(TypeError):
        Stutter("48000", 20)
    with pytest.raises(ValueError):
        Stutter("48000", 20)
    with pytest.raises(TypeError):
        Stutter(48000, "20")


@pytest.mark.parametrize("value,cls_name", [
    (float("nan"), "NaNSampleError"),
    (float("inf"), "PositiveInfinitySampleError"),
    (float("-inf"), "NegativeInfinitySampleError"),
])
def test_nonfinite_classified_distinct(value, cls_name):
    det = Stutter(48000, 20)
    cls = getattr(stutter, cls_name)
    with pytest.raises(cls):
        det.detect([0.0, value])
    with pytest.raises(ValueError):
        det.detect([0.0, value])


def test_nonfinite_three_distinct_types():
    det = Stutter(48000, 20)
    kinds = set()
    for v in (float("nan"), float("inf"), float("-inf")):
        try:
            det.detect([v])
        except ValueError as exc:
            kinds.add(type(exc))
    assert len(kinds) == 3


def test_nan_message_match_tokens():
    det = Stutter(48000, 20)
    with pytest.raises(ValueError, match="(?i)nan|non-finite|有限"):
        det.detect([float("nan")])
    with pytest.raises(ValueError, match="(?i)inf|non-finite|有限"):
        det.detect([float("inf")])


# ---------------------------------------------------------------------------
# 输出契约 / 不变量
# ---------------------------------------------------------------------------


def test_output_contract_on_all_silent():
    out = Stutter(48000, 20).detect([0.0] * 1000)
    assert len(out) >= 1
    for g in out:
        assert isinstance(g, list) and len(g) == 2
        assert all(type(v) is int for v in g)
        a, b = g
        assert 0 <= a < b <= 1000


def test_empty_and_short():
    det = Stutter(48000, 20)
    assert det.detect([]) == []
    assert det.detect([0.0] * 10) == []
    assert isinstance(det.detect([0.5] * 100), list)


def test_deterministic_and_no_mutation():
    x = [0.5] * 500 + [0.0] * 2000 + [0.5] * 500
    det = Stutter(48000, 20)
    assert det.detect(x) == det.detect(list(x))
    before = list(x)
    det.detect(x)
    assert x == before


def test_ndarray_tuple_generator_inputs():
    det = Stutter(48000, 20)
    assert det.detect(np.zeros(2000)) == [[0, 2000]]
    assert det.detect(tuple([0.0] * 2000)) == [[0, 2000]]
    assert det.detect(0.0 for _ in range(2000)) == [[0, 2000]]
