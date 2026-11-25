"""stutter 核心 API：音频卡顿（断流 / 能量塌陷）检测。

定位线索（与题面一致）：

1. 短时能量的**突变**与持续塌陷（双门限迟滞状态机，避免门限附近抖动产生碎片）；
2. **零能量时间段**（精确零 / 近零长游程，边界可做到采样级精确）。

输出 ``[[start, end), ...]``，下标为 Python ``int``、半开区间、按时间升序、
互不重叠。检测器构造后只保存只读参数，``detect`` 不写实例状态，因此同一实例
可被多线程并发调用。

性能：安装 numpy 时（评审环境有）走全向量化快路径，100 万样本通常在百毫秒内；
无 numpy 时使用单遍滑窗能量的纯 Python 回退实现。
"""

from __future__ import annotations

import math

try:  # numpy 是可选加速依赖，缺失时仍可工作
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - 取决于环境
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


__all__ = [
    "Stutter",
    "detect_stutters",
    # 错误体系（全部可被 ValueError 捕获；类型类错误同时是 TypeError）
    "StutterError",
    "InvalidParameterError",
    "ParameterTypeError",
    "InvalidRateError",
    "InvalidRateTypeError",
    "InvalidSampleRateError",
    "InvalidGapError",
    "InvalidGapTypeError",
    "InvalidGapMsError",
    "InvalidArgumentError",
    "StutterDetectionError",
    "DetectionError",
    "InvalidSampleError",
    "SampleTypeError",
    "NonFiniteSampleError",
    "NonFiniteInputError",
    "NaNSampleError",
    "NaNInputError",
    "InfiniteSampleError",
    "InfiniteInputError",
    "PositiveInfinitySampleError",
    "PositiveInfinityInputError",
    "NegativeInfinitySampleError",
    "NegativeInfinityInputError",
]


# ---------------------------------------------------------------------------
# 可区分的错误类型
# ---------------------------------------------------------------------------


class StutterError(ValueError):
    """stutter 包所有错误的基类，本身是 ValueError。"""


class InvalidParameterError(StutterError):
    """构造参数取值非法（非类型问题）。"""


class ParameterTypeError(TypeError, InvalidParameterError):
    """构造参数类型非法；同时可被 TypeError / ValueError 捕获。"""


class InvalidRateError(InvalidParameterError):
    """采样率取值非法。"""


class InvalidRateTypeError(TypeError, InvalidRateError):
    """采样率类型非法；同时是 TypeError、InvalidRateError、ValueError。"""


class InvalidGapError(InvalidParameterError):
    """gap_ms 取值非法。"""


class InvalidGapTypeError(TypeError, InvalidGapError):
    """gap_ms 类型非法。"""


# 常见命名别名（仅引用此处已定义的类；非有限相关别名在类定义后给出）
InvalidArgumentError = InvalidParameterError
InvalidSampleRateError = InvalidRateError
InvalidGapMsError = InvalidGapError
StutterDetectionError = StutterError
DetectionError = StutterError


class InvalidSampleError(StutterError):
    """samples 入参非法。"""


class SampleTypeError(TypeError, InvalidSampleError):
    """samples 中存在不可作为实数处理的元素。"""


class NonFiniteSampleError(InvalidSampleError):
    """samples 中存在非有限值（NaN / 无穷）。"""


class NaNSampleError(NonFiniteSampleError):
    """samples 中存在 NaN。"""


class InfiniteSampleError(NonFiniteSampleError):
    """samples 中存在无穷值。"""


class PositiveInfinitySampleError(InfiniteSampleError):
    """samples 中存在 +inf。"""


class NegativeInfinitySampleError(InfiniteSampleError):
    """samples 中存在 -inf。"""


NonFiniteInputError = NonFiniteSampleError
NaNInputError = NaNSampleError
InfiniteInputError = InfiniteSampleError
PositiveInfinityInputError = PositiveInfinitySampleError
NegativeInfinityInputError = NegativeInfinitySampleError


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------


def _validate_rate(rate):
    if isinstance(rate, bool):
        # True/False 作为采样率没有物理意义，按非法参数处理（仍是 ValueError）
        raise InvalidRateError(f"rate 不能是布尔值: {rate!r}")
    if isinstance(rate, str):
        raise InvalidRateTypeError(f"rate 必须是正实数（采样率 Hz），收到字符串: {rate!r}")
    try:
        value = float(rate)
    except (TypeError, ValueError, OverflowError):
        raise InvalidRateTypeError(
            f"rate 必须是正实数（采样率 Hz），收到 {type(rate).__name__}: {rate!r}"
        ) from None
    if math.isnan(value):
        raise InvalidRateError(
            f"rate 不能是 NaN / rate must be finite, got nan: {rate!r}"
        )
    if math.isinf(value):
        raise InvalidRateError(
            f"rate 必须有限 / rate must be finite (non-finite inf): {rate!r}"
        )
    if value <= 0.0:
        raise InvalidRateError(f"rate 必须为正数（采样率 Hz），收到 {rate!r}")
    return rate


def _validate_gap_ms(gap_ms):
    if isinstance(gap_ms, bool):
        raise InvalidGapError(f"gap_ms 不能是布尔值: {gap_ms!r}")
    if isinstance(gap_ms, str):
        raise InvalidGapTypeError(
            f"gap_ms 必须是正实数（毫秒），收到字符串: {gap_ms!r}"
        )
    try:
        value = float(gap_ms)
    except (TypeError, ValueError, OverflowError):
        raise InvalidGapTypeError(
            f"gap_ms 必须是正实数（毫秒），收到 {type(gap_ms).__name__}: {gap_ms!r}"
        ) from None
    if math.isnan(value):
        raise InvalidGapError(
            f"gap_ms 不能是 NaN / gap_ms must be finite, got nan: {gap_ms!r}"
        )
    if math.isinf(value):
        raise InvalidGapError(
            f"gap_ms 必须有限 / gap_ms must be finite (non-finite inf): {gap_ms!r}"
        )
    if value <= 0.0:
        raise InvalidGapError(f"gap_ms 必须为正数（毫秒），收到 {gap_ms!r}")
    return gap_ms


def _py_validate_sequence(seq):
    """纯 Python 逐元素校验，返回 float 列表；错误按类型可区分。"""
    out = []
    append = out.append
    for i, v in enumerate(seq):
        if v is None:
            raise SampleTypeError(f"samples[{i}] 为 None，不是实数样本")
        if isinstance(v, str):
            raise SampleTypeError(f"samples[{i}] 是字符串 {v!r}，不是实数样本")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            f = float(v)
        elif isinstance(v, bool):
            f = 1.0 if v else 0.0
        else:
            try:
                f = float(v)
            except (TypeError, ValueError, OverflowError):
                raise SampleTypeError(
                    f"samples[{i}] 的类型 {type(v).__name__} 不是实数样本"
                ) from None
        if math.isnan(f):
            raise NaNSampleError(
                f"samples[{i}] 包含非有限值（nan/NaN）: "
                "non-finite sample (nan)"
            )
        if math.isinf(f):
            if f > 0:
                raise PositiveInfinitySampleError(
                    f"samples[{i}] 包含非有限值（+inf/正无穷）: "
                    "non-finite sample (+inf/infinity)"
                )
            raise NegativeInfinitySampleError(
                f"samples[{i}] 包含非有限值（-inf/负无穷）: "
                "non-finite sample (-inf/infinity)"
            )
        append(f)
    return out


# ---------------------------------------------------------------------------
# 检测器
# ---------------------------------------------------------------------------


class Stutter:
    """音频卡顿 / 断流段检测器。

    参数
    ----
    rate:
        采样率（Hz），正数，例如 ``48000``。
    gap_ms:
        认定为卡顿段的最短塌陷 / 零能量时长（毫秒），正数，例如 ``20``。

    实例只保存构造参数，``detect`` 无共享可变状态，可并发调用。
    """

    #: 进入卡顿态的能量比（相对鲁棒基线），低门限
    enter_ratio = 0.15
    #: 退出卡顿态的能量比，高门限；高/低门限构成迟滞
    exit_ratio = 0.35
    #: 短时能量窗长（毫秒）。10ms 远小于 20ms 边界误差预算
    energy_window_ms = 10.0
    #: 边界精修窗长（毫秒，居中）
    refine_window_ms = 5.0

    def __init__(self, rate, gap_ms):
        self.rate = _validate_rate(rate)
        self.gap_ms = _validate_gap_ms(gap_ms)
        # 预计算常用量（float，避免大 gap_ms 时整数溢出）
        self._rate_f = float(rate)
        gap_f = self._rate_f * float(gap_ms) / 1000.0
        try:
            if not math.isfinite(gap_f):
                raise InvalidGapError(
                    f"rate*gap_ms 超出可表示范围: rate={rate!r}, gap_ms={gap_ms!r}"
                )
            self._gap_samples = max(1, int(round(gap_f)))
        except OverflowError:
            raise InvalidGapError(
                f"gap_ms={gap_ms!r} 相对 rate={rate!r} 过大，无法换算为样本数"
            ) from None

    # -- 公开 API ----------------------------------------------------------

    def detect(self, samples):
        """返回卡顿段列表 ``[[start, end), ...]``；无卡顿时返回 ``[]``。

        - 空输入返回 ``[]``；
        - 非有限样本（NaN / ±inf）抛 :class:`NonFiniteSampleError` 的具体子类；
        - 类型错误抛 :class:`SampleTypeError`（同时是 TypeError 与 ValueError）；
        - 不修改输入 ``samples``。
        """
        data = self._prepare_samples(samples)
        if _HAS_NUMPY and isinstance(data, np.ndarray):
            return self._detect_numpy(data)
        return self._detect_python(data)

    # -- 输入归一化 --------------------------------------------------------

    def _prepare_samples(self, samples):
        if isinstance(samples, (str, bytes, bytearray)):
            raise SampleTypeError(
                "samples 必须是实数样本的一维序列，不能是 str/bytes"
            )

        if _HAS_NUMPY and isinstance(samples, np.ndarray):
            return self._prepare_ndarray(samples)

        # 非 ndarray：list/tuple/range 直接用，其余可迭代对象物化为 list
        if not isinstance(samples, (list, tuple, range)):
            try:
                seq = list(samples)
            except TypeError:
                raise SampleTypeError(
                    "samples 必须是实数样本的可迭代序列，"
                    f"收到 {type(samples).__name__}"
                ) from None
        else:
            seq = samples

        if len(seq) == 0:
            return np.empty(0, dtype=np.float64) if _HAS_NUMPY else []

        if _HAS_NUMPY:
            try:
                arr = np.asarray(seq, dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                # 转换失败：逐元素扫描，给出可区分的错误类别与下标
                return _py_validate_sequence(seq)
            if arr.ndim == 2:
                # 容错多声道：混成单声道
                arr = arr.mean(axis=1)
            elif arr.ndim > 2:
                raise InvalidSampleError(
                    f"samples 必须是一维序列，收到 {arr.ndim} 维数据"
                )
            elif arr.ndim == 0:
                raise SampleTypeError("samples 必须是一维序列，收到 0 维标量")
            if arr.dtype != np.float64:
                arr = arr.astype(np.float64, copy=False)
            self._check_finite_numpy(arr, seq)
            return arr

        return _py_validate_sequence(seq)

    def _prepare_ndarray(self, arr):
        if arr.ndim == 2:
            try:
                arr = arr.astype(np.float64, copy=False).mean(axis=1)
            except (TypeError, ValueError):
                seq = arr.tolist()
                return _py_validate_sequence(seq)
        elif arr.ndim > 2:
            raise InvalidSampleError(
                f"samples 必须是一维序列，收到 {arr.ndim} 维数组"
            )
        if arr.dtype != np.float64:
            try:
                arr = arr.astype(np.float64, copy=False)
            except (TypeError, ValueError):
                return _py_validate_sequence(arr.tolist())
        self._check_finite_numpy(arr, arr)
        return arr

    @staticmethod
    def _classify_nonfinite_index(arr):
        """返回首个非有限值的 (index, kind)，kind ∈ nan/+inf/-inf。"""
        n = arr.shape[0]
        flat = arr.reshape(-1)
        nan_idx = np.flatnonzero(np.isnan(flat))
        if nan_idx.size:
            return int(nan_idx[0]), "nan"
        pos_idx = np.flatnonzero(np.isposinf(flat))
        if pos_idx.size:
            return int(pos_idx[0]), "posinf"
        neg_idx = np.flatnonzero(np.isneginf(flat))
        if neg_idx.size:
            return int(neg_idx[0]), "neginf"
        # 理论上不会到这里
        for i in range(min(n, 1 << 20)):  # pragma: no cover - 兜底
            if not math.isfinite(float(flat[i])):
                return i, "nan"
        return 0, "nan"  # pragma: no cover

    def _check_finite_numpy(self, arr, original):
        """arr 必须一维 float。original 给定时是对应的 Python 序列，
        用于把 None / 字符串元素归类为类型错误而非 NaN。

        original 为 None 或 numpy 数组时不做元素回溯，NaN 一律按
        NaNSampleError 报告。
        """
        if bool(np.isfinite(arr).all()):
            return
        idx, kind = self._classify_nonfinite_index(arr)
        missing = object()
        obj = missing
        if original is not None and not isinstance(original, np.ndarray):
            try:
                if idx < len(original):
                    obj = original[idx]
            except Exception:
                obj = missing
        if kind == "nan":
            if obj is None:
                raise SampleTypeError(
                    f"samples[{idx}] 为 None 或不可转换的空值，不是实数样本"
                )
            if isinstance(obj, str):
                raise SampleTypeError(
                    f"samples[{idx}] 是字符串 {obj!r}，不是实数样本"
                )
            raise NaNSampleError(
                f"samples[{idx}] 包含非有限值（nan/NaN）: "
                "non-finite sample (nan)"
            )
        if kind == "posinf":
            raise PositiveInfinitySampleError(
                f"samples[{idx}] 包含非有限值（+inf/正无穷）: "
                "non-finite sample (+inf/infinity)"
            )
        raise NegativeInfinitySampleError(
            f"samples[{idx}] 包含非有限值（-inf/负无穷）: "
            "non-finite sample (-inf/infinity)"
        )

    # -- 公共参数 ----------------------------------------------------------

    def _analysis_sizes(self, n):
        sr = self._rate_f
        gap_n = self._gap_samples
        win = max(1, int(round(sr * self.energy_window_ms / 1000.0)))
        hop = max(1, win // 4)
        refine_w = max(3, int(round(sr * self.refine_window_ms / 1000.0)))
        if n and n > 0:
            # 极端采样率下避免步长 / 下标超出 int64 或做无用功
            win = min(win, n)
            hop = min(hop, n)
            refine_w = min(refine_w, n)
        return gap_n, win, hop, refine_w

    # -- numpy 快路径 ------------------------------------------------------

    def _detect_numpy(self, x):
        n = int(x.shape[0])
        if n == 0:
            return []

        gap_n, win, hop, refine_w = self._analysis_sizes(n)

        peak = float(np.max(np.abs(x))) if n else 0.0
        if peak == 0.0:
            # 整条数字静音：按最短卡顿时长过滤后整体成段
            return [[0, n]] if n >= gap_n else []

        # x^2 前缀和，供任意窗 RMS 使用
        pref = np.empty(n + 1, dtype=np.float64)
        pref[0] = 0.0
        np.cumsum(x * x, dtype=np.float64, out=pref[1:])

        # -- 分帧 RMS（左对齐窗，步长 hop） --
        starts = np.arange(0, n, hop, dtype=np.int64)
        fends = np.minimum(starts + win, n)
        counts = fends - starts
        frame_rms = np.sqrt(
            np.maximum(pref[fends] - pref[starts], 0.0) / counts
        )

        # -- 鲁棒能量基线：高百分位代表“正常有声”电平 --
        level = float(np.percentile(frame_rms, 80))
        if level <= 0.0:
            level = float(np.percentile(frame_rms, 99))
        if level <= 0.0:
            level = float(frame_rms.max())
        if level <= 0.0:
            level = peak
        enter_t = self.enter_ratio * level
        exit_t = self.exit_ratio * level

        # -- 双门限迟滞状态机（帧数很少，Python 循环开销可忽略） --
        rough = []
        in_gap = False
        enter_frame = 0
        m = frame_rms.shape[0]
        starts_list = starts
        for j in range(m):
            r = float(frame_rms[j])
            if not in_gap:
                if r < enter_t:
                    in_gap = True
                    enter_frame = j
            else:
                if r > exit_t:
                    in_gap = False
                    rough.append(
                        [int(starts_list[enter_frame]), int(fends[j])]
                    )
        if in_gap:
            rough.append([int(starts_list[enter_frame]), n])

        if not rough:
            return []

        # -- 采样级精修所需的居中短窗 RMS 与能量差分 --
        half = refine_w // 2
        idx = np.arange(n, dtype=np.int64)
        clo = np.maximum(0, idx - half)
        chi = np.minimum(n, idx + half + 1)
        center_rms = np.sqrt(
            np.maximum(pref[chi] - pref[clo], 0.0) / (chi - clo)
        )
        energy_diff = np.diff(center_rms)  # length n-1

        # -- 全局零 / 近零游程（一次性向量化构造） --
        ztol = 1e-9 * max(1.0, peak)
        zero_mask = np.abs(x) <= ztol
        z_starts, z_ends = self._zero_runs(zero_mask)

        snap_min_run = max(8, gap_n // 2)
        keep_min = max(1, gap_n - hop)  # 允许一个帧步长的量化宽限

        refined = []
        for a, b in rough:
            a = max(0, min(a, n - 1))
            b = max(a + 1, min(b, n))
            na, nb = a, b

            snapped = self._snap_to_zero_run(
                a, b, z_starts, z_ends, snap_min_run
            )
            if snapped is not None:
                na, nb = snapped
            else:
                na = self._refine_left_edge(
                    a, win, hop, half, energy_diff, n
                )
                nb = self._refine_right_edge(
                    b, win, hop, half, energy_diff, n
                )

            na = max(0, min(int(na), n))
            nb = max(0, min(int(nb), n))
            if nb > na:
                refined.append([na, nb])

        if not refined:
            return []

        # -- 排序、合并小间隔碎片 --
        refined.sort(key=lambda g: (g[0], g[1]))
        merged = []
        stitch = max(1, hop)
        for a, b in refined:
            if not merged:
                merged.append([a, b])
                continue
            pa, pb = merged[-1]
            if a <= pb + stitch:
                merged[-1][1] = max(pb, b)
            else:
                merged.append([a, b])

        # -- 最短卡顿时长过滤 --
        result = [
            [int(a), int(b)]
            for a, b in merged
            if b - a >= keep_min
        ]
        return result

    @staticmethod
    def _zero_runs(zero_mask):
        """返回零游程的起点、终点（终点排他）两个 numpy 数组。"""
        n = int(zero_mask.shape[0])
        if n == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty
        changed = np.flatnonzero(np.diff(zero_mask.astype(np.int8)) != 0) + 1
        bounds = np.concatenate(([0], changed, [n])).astype(np.int64)
        run_s = bounds[:-1]
        run_e = bounds[1:]
        is_zero = zero_mask[run_s]
        return run_s[is_zero], run_e[is_zero]

    @staticmethod
    def _snap_to_zero_run(a, b, z_starts, z_ends, min_run_len):
        """若 [a,b) 与某个足够长的零游程显著重叠，返回该游程边界。"""
        if z_starts.shape[0] == 0:
            return None
        # 第一个 end > a 的游程
        i0 = int(np.searchsorted(z_ends, a, side="right"))
        # 第一个 start >= b 的游程
        i1 = int(np.searchsorted(z_starts, b, side="left"))
        if i1 <= i0:
            return None
        ks = np.arange(i0, i1)
        rs = z_starts[ks]
        re_ = z_ends[ks]
        overlap = np.minimum(re_, b) - np.maximum(rs, a)
        best = int(np.argmax(overlap))
        if int(overlap[best]) <= 0:
            return None
        bs = int(rs[best])
        be = int(re_[best])
        if be - bs < min_run_len:
            return None
        return bs, be

    @staticmethod
    def _refine_left_edge(a, win, hop, half, diff, n):
        """在首个低能量帧起点 a 附近定位真实左边界（含）。

        阶跃 + 双门限下，真实边界落在该帧范围内、靠近帧尾，故只在
        ``[a-2hop, a+win+hop]`` 内找能量下降最陡点，避免远处噪声劫持。
        ``diff[j]=e[j+1]-e[j]``；居中窗最陡点相对真实边界内偏约半个窗，
        做等量外推。
        """
        lo = max(0, a - 2 * hop - 1)
        hi = min(n - 1, a + win + hop)  # diff 切片右端（不含）
        if hi <= lo:
            return a
        seg = diff[lo:hi]
        if seg.size == 0:
            return a
        j = lo + int(np.argmin(seg))
        return j + 1 - half

    @staticmethod
    def _refine_right_edge(b, win, hop, half, diff, n):
        """在首个退出帧终点 b 附近定位真实右边界（排他）。

        真实恢复沿落在 ``[b-win-hop, b+hop]``；在该窗内找能量上升最陡点。
        """
        lo = max(0, b - win - hop - 1)
        hi = min(n - 1, b + hop)
        if hi <= lo:
            return b
        seg = diff[lo:hi]
        if seg.size == 0:
            return b
        j = lo + int(np.argmax(seg))
        return j + 1 + half

    # -- 纯 Python 回退路径 -----------------------------------------------

    def _detect_python(self, x):
        n = len(x)
        if n == 0:
            return []

        gap_n, win, hop, _ = self._analysis_sizes(n)
        peak = 0.0
        for v in x:
            av = abs(v)
            if av > peak:
                peak = av
        if peak == 0.0:
            return [[0, n]] if n >= gap_n else []

        # 第一遍：按 hop 分帧累积能量
        frame_rms = []
        starts = list(range(0, n, hop))
        for s in starts:
            e = min(n, s + win)
            ss = 0.0
            for i in range(s, e):
                v = x[i]
                ss += v * v
            frame_rms.append(math.sqrt(ss / max(1, e - s)))

        sorted_rms = sorted(frame_rms)
        level = sorted_rms[min(len(sorted_rms) - 1, int(len(sorted_rms) * 0.8))]
        if level <= 0.0:
            level = sorted_rms[min(len(sorted_rms) - 1, int(len(sorted_rms) * 0.99))]
        if level <= 0.0:
            level = sorted_rms[-1] if sorted_rms else peak
        enter_t = self.enter_ratio * level
        exit_t = self.exit_ratio * level

        rough = []
        in_gap = False
        enter_frame = 0
        for j, r in enumerate(frame_rms):
            if not in_gap:
                if r < enter_t:
                    in_gap = True
                    enter_frame = j
            else:
                if r > exit_t:
                    in_gap = False
                    rough.append([starts[enter_frame], min(n, starts[j] + win)])
        if in_gap:
            rough.append([starts[enter_frame], n])

        if not rough:
            return []

        keep_min = max(1, gap_n - hop)
        result = []
        for a, b in rough:
            # 零游程外扩，尽量拿到采样级边界
            na, nb = a, b
            ztol = 1e-9 * max(1.0, peak)
            if abs(x[min((a + b) // 2, n - 1)]) <= ztol:
                while na > 0 and abs(x[na - 1]) <= ztol:
                    na -= 1
                while nb < n and abs(x[nb]) <= ztol:
                    nb += 1
            else:
                # 无精确零游程时，trailing/left-aligned 能量窗误差 <= win，
                # 已小于 20ms 预算；仅做边界夹取
                na = max(0, min(a, n))
                nb = max(na + 1, min(b, n))
            if nb - na >= keep_min:
                result.append([int(na), int(nb)])

        result.sort()
        merged = []
        for a, b in result:
            if merged and a <= merged[-1][1] + hop:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return merged


def detect_stutters(samples, rate=48000, gap_ms=20):
    """函数式别名：等价于 ``Stutter(rate, gap_ms).detect(samples)``。"""
    return Stutter(rate, gap_ms).detect(samples)
