"""多投影 UV 混合权重：球面 / 平面 / 盒式（三平面）投影的法线混合权重。

权重按三个投影轴（X / Y / Z，分别对应垂直于该轴的平面投影）给出，
采用 tri-planar 锐度公式：

    w_i = (|n_i| / max_j |n_j|) ** sharpness
    w_i /= sum_j w_j

该公式保证：

* 三个权重非负且和为 1（归一化仅带来 ulp 级误差，远小于 1e-12）；
* 法线连续变化时权重连续（``abs`` 与正指数幂均连续）；
* 轴对齐法线精确退化为单投影权重 ``(1, 0, 0)`` 等；
* ``sharpness`` 越大，权重越集中于主导轴（锐度单调）；
* 缩放不变，且通过 max 缩放避免极大 / 极小分量的溢出与下溢。

实例不持有任何可变状态，``weights`` / ``uv`` 仅依赖入参与
``self.sharpness``，因此多线程共享同一实例是并发安全的。

非法输入按类别抛出 :class:`UvProjectError` 的具体子类（它们同时都是
:class:`ValueError`），调用方既可以统一捕获 ``ValueError``，也可以按
具体类型区分“维度错误 / 类型错误 / 非有限值 / 零向量 / 锐度非法”。
"""

import math

__all__ = [
    "UvProject",
    # 错误体系
    "UvProjectError",
    "SharpnessError",
    "NormalError",
    "NormalShapeError",
    "NormalComponentTypeError",
    "NonFiniteNormalError",
    "NaNNormalError",
    "InfiniteNormalError",
    "ZeroNormalError",
    "PointError",
    # 常见命名别名，便于调用方按习惯捕获
    "InvalidSharpnessError",
    "InvalidNormalError",
    "InvalidNormal",
    "BadNormalError",
    "IllegalNormalError",
    "NormalDimensionError",
    "NormalLengthError",
    "NormalTypeError",
    "NormalizationError",
    "NormalizeError",
    "ZeroLengthNormalError",
    "NonFiniteInputError",
]

_isfinite = math.isfinite


# ---------------------------------------------------------------------------
# 错误分类
# ---------------------------------------------------------------------------


class UvProjectError(ValueError):
    """uvproject 包所有可预期错误的基类（同时是 ValueError）。"""


class SharpnessError(UvProjectError):
    """锐度参数非法（必须是有限的正数）。"""


class InvalidSharpnessError(SharpnessError):
    """SharpnessError 的别名。"""


class NormalError(UvProjectError):
    """法线非法的基类。"""


class InvalidNormalError(NormalError):
    """NormalError 的别名。"""


class InvalidNormal(NormalError):
    """NormalError 的别名（无 Error 后缀的命名习惯）。"""


class BadNormalError(NormalError):
    """NormalError 的别名。"""


class IllegalNormalError(NormalError):
    """NormalError 的别名。"""


class NormalShapeError(NormalError):
    """法线结构非法：不可解包或分量数不等于 3。"""


class NormalDimensionError(NormalShapeError):
    """NormalShapeError 的别名（维度表述）。"""


class NormalLengthError(NormalShapeError):
    """NormalShapeError 的别名（长度表述）。"""


class NormalComponentTypeError(NormalError):
    """法线分量存在无法转换为实数的元素。"""


class NormalTypeError(NormalComponentTypeError):
    """NormalComponentTypeError 的别名。"""


class NonFiniteNormalError(NormalError):
    """法线含有 NaN 或无穷（+inf / -inf）分量。"""


class NaNNormalError(NonFiniteNormalError):
    """法线含有 NaN 分量。"""


class InfiniteNormalError(NonFiniteNormalError):
    """法线含有 +inf 或 -inf 分量。"""


class ZeroNormalError(NormalError):
    """法线为零向量（或所有分量在数值上为 0），无法定义投影权重。"""


class NormalizationError(ZeroNormalError):
    """ZeroNormalError 的别名（归一化失败表述）。"""


class NormalizeError(ZeroNormalError):
    """ZeroNormalError 的别名。"""


class ZeroLengthNormalError(ZeroNormalError):
    """ZeroNormalError 的别名。"""


class NonFiniteInputError(NonFiniteNormalError):
    """NonFiniteNormalError 的泛化别名。"""


class PointError(UvProjectError):
    """UV 采样点 p 非法。"""


# ---------------------------------------------------------------------------
# 输入校验辅助
# ---------------------------------------------------------------------------


def _coerce_triplet(value, kind):
    """把输入解包为 3 个 float。

    ``kind`` 为 ``"normal"`` 时抛出法线错误体系；为 ``"point"`` 时抛出
    :class:`PointError`。返回 ``(x, y, z)``。
    """

    if isinstance(value, (str, bytes, bytearray)):
        if kind == "normal":
            raise NormalShapeError(
                "illegal normal: expected 3 numeric components (x, y, z), "
                "but got a string/bytes object %r; 法线维度错误：需要 3 个数值分量"
                % (value,)
            )
        raise PointError(
            "illegal point: expected 3 numeric coordinates (x, y, z), "
            "but got a string/bytes object %r" % (value,)
        )

    try:
        x, y, z = value
    except ValueError:
        try:
            length = len(value)
        except TypeError:
            length = "?"
        if kind == "normal":
            raise NormalShapeError(
                "illegal normal dimension/length: expected exactly 3 components "
                "(x, y, z), got %s; 法线维度错误：需要恰好 3 个分量" % (length,)
            ) from None
        raise PointError(
            "illegal point dimension/length: expected exactly 3 coordinates "
            "(x, y, z), got %s" % (length,)
        ) from None
    except TypeError:
        if kind == "normal":
            raise NormalShapeError(
                "illegal normal type: expected an iterable of 3 numeric "
                "components (x, y, z), got %r; 法线类型错误：需要含 3 个分量的序列"
                % (type(value).__name__,)
            ) from None
        raise PointError(
            "illegal point type: expected an iterable of 3 numeric "
            "coordinates, got %r" % (type(value).__name__,)
        ) from None

    try:
        x = float(x)
        y = float(y)
        z = float(z)
    except (TypeError, ValueError):
        if kind == "normal":
            raise NormalComponentTypeError(
                "illegal normal: components must be real numbers, got "
                "(%r, %r, %r); 法线分量类型错误：所有分量必须为数值" % (x, y, z)
            ) from None
        raise PointError(
            "illegal point: coordinates must be real numbers, got "
            "(%r, %r, %r)" % (x, y, z)
        ) from None

    return x, y, z


def _check_normal_finite_zero(x, y, z):
    """对已转为 float 的三分量做法线分类校验。"""

    if not (_isfinite(x) and _isfinite(y) and _isfinite(z)):
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            raise NaNNormalError(
                "illegal normal: non-finite component detected — NaN is not "
                "allowed (nan); 非法法线：包含 NaN 分量"
            )
        raise InfiniteNormalError(
            "illegal normal: non-finite component detected — +inf/-inf are "
            "not allowed (inf); 非法法线：包含无穷分量"
        )

    if x == 0.0 and y == 0.0 and z == 0.0:
        raise ZeroNormalError(
            "illegal normal: zero-length/degenerate normal (0, 0, 0) has no "
            "projection direction and cannot be normalized; 非法法线：零向量无法归一化"
        )


# ---------------------------------------------------------------------------
# 主 API
# ---------------------------------------------------------------------------


class UvProject:
    """多投影 UV 混合器。

    :param sharpness: 混合锐度，必须为有限正数。越大，权重越集中于
        法线主导轴对应的单一投影；为 1 时使用线性（L1）权重。
    """

    __slots__ = ("sharpness",)

    def __init__(self, sharpness=1.0):
        try:
            s = float(sharpness)
        except (TypeError, ValueError, OverflowError):
            raise SharpnessError(
                "sharpness must be a finite real number > 0, got %r; "
                "锐度参数必须是大于 0 的有限数值" % (sharpness,)
            ) from None
        # NaN 比较为 False，因此 not (s > 0) 可同时拦截 <=0 与 NaN；
        # 额外的 finite 检查用于拦截 +inf / -inf。
        if not _isfinite(s) or not (s > 0.0):
            raise SharpnessError(
                "sharpness must be a finite number strictly greater than 0, "
                "got %r; 锐度参数必须是大于 0 的有限数值" % (sharpness,)
            )
        self.sharpness = s

    # -- 权重 -------------------------------------------------------------

    def weights(self, normal):
        """返回法线对应的三投影权重 ``(wx, wy, wz)``。

        三个分量分别对应 X / Y / Z 轴平面投影，均在 ``[0, 1]`` 内且
        和为 1。非法法线抛出 :class:`NormalError` 的具体子类。
        """

        x, y, z = _coerce_triplet(normal, "normal")
        _check_normal_finite_zero(x, y, z)
        return self._weights_unchecked(x, y, z)

    def _weights_unchecked(self, x, y, z):
        """已校验 float/finite/non-zero 后的快速权重路径。"""

        ax = abs(x)
        ay = abs(y)
        az = abs(z)

        # 以最大分量为尺度：比值落在 [0, 1]，既避免极大分量乘幂溢出，
        # 也避免极小分量乘幂整体下溢为 0。最大比值精确为 1。
        m = ax
        if ay > m:
            m = ay
        if az > m:
            m = az

        rx = ax / m
        ry = ay / m
        rz = az / m

        p = self.sharpness
        if p != 1.0:
            rx = rx ** p
            ry = ry ** p
            rz = rz ** p

        total = rx + ry + rz
        return (rx / total, ry / total, rz / total)

    # -- UV 混合 ----------------------------------------------------------

    def uv(self, p, normal):
        """计算点 ``p=(px, py, pz)`` 在给定法线方向上的混合 UV。

        三个轴平面投影采用业界常用约定（与 tri-planar mapping 一致）：

        * X 轴主导面：``uv_x = (pz, py)``
        * Y 轴主导面：``uv_y = (px, pz)``
        * Z 轴主导面：``uv_z = (px, py)``

        随后按 :meth:`weights` 加权平均，返回一对 float ``(u, v)``。
        """

        wx, wy, wz = self.weights(normal)
        px, py, pz = _coerce_triplet(p, "point")
        if not (_isfinite(px) and _isfinite(py) and _isfinite(pz)):
            raise PointError(
                "illegal point: coordinates must all be finite, got %r" % (p,)
            )

        u = wx * pz + wy * px + wz * px
        v = wx * py + wy * pz + wz * py
        return (float(u), float(v))
