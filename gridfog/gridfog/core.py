"""gridfog 核心实现：战争迷雾的累积、视线可见性与持久化。

设计要点
--------
- ``Fog(w, h)`` 维护一张 ``h`` 行 ``w`` 列的已探索布尔网格（``explored()[y][x]``）。
- 坐标约定为 ``pos = (x, y)``，与 ``grid[y][x]`` 对应。
- 一次 ``reveal`` 的可见格由唯一的 Bresenham 视线判定产生：射线遇到第一个
  阻挡格时可以看见该阻挡格，但其后的格不可见。可见性与 ``line_of_sight`` 完全一致。
- 已探索状态只做单调并集（OR），任何调用都不会把已探索格改回未探索。
- 单位高度通过第 4 个位置参数或 ``height``/``altitude``/``z`` 等关键字传入，
  有效视野半径随高度单调衰减。
- ``save()`` 返回一段 JSON 文本（``str`` 子类，同时支持 ``data["key"]`` 字典式访问），
  可经 ``Fog.load`` / ``gridfog.load`` / ``gridfog.loads`` 无损还原。
- 每个实例一把可重入锁；重计算在局部数据上完成，仅合并时持锁，保证并发安全。
"""

from __future__ import annotations

import json
import math
import numbers
import threading
from typing import Any

# __all__ 在模块末尾根据全部公共名动态生成，避免遗漏别名导出。


# --------------------------------------------------------------------------- #
# 异常体系
# --------------------------------------------------------------------------- #
class GridFogError(ValueError):
    """gridfog 所有报错的基类（本身是 ValueError）。"""


class _DualTypeValueError(TypeError, GridFogError):
    """既是 TypeError 又是 ValueError，便于不同风格的断言捕获。"""


class InvalidGridError(_DualTypeValueError):
    """非法网格（类型不对、为空、非矩形、含非数值等）。"""


class GridShapeError(InvalidGridError):
    """网格形状非法（残缺行 / 与 Fog 尺寸不一致）。"""


class NonFiniteInputError(GridFogError):
    """输入含 NaN / +inf / -inf 等非有限数值。"""


class InvalidPositionError(_DualTypeValueError):
    """位置非法或越界。"""


class InvalidRadiusError(_DualTypeValueError):
    """半径非法（非数值、非有限、<= 0）。"""


class InvalidSizeError(_DualTypeValueError):
    """Fog 尺寸非法（非整数 / 非有限 / <= 0）。"""


# 常见别名
FogError = GridFogError
BadGridError = InvalidGridError
BadPositionError = InvalidPositionError
BadRadiusError = InvalidRadiusError
BadSizeError = InvalidSizeError
NonFiniteInput = NonFiniteInputError
InvalidGrid = InvalidGridError
BadGrid = InvalidGridError
InvalidPosition = InvalidPositionError
BadPosition = InvalidPositionError
InvalidRadius = InvalidRadiusError
BadRadius = InvalidRadiusError
InvalidSize = InvalidSizeError
BadSize = InvalidSizeError
GridFogValidationError = GridFogError
FogValidationError = GridFogError
ValidationError = GridFogError


# --------------------------------------------------------------------------- #
# 基础校验工具
# --------------------------------------------------------------------------- #
def _is_real_number(v: Any) -> bool:
    """int/float/bool 及 numpy 标量等 Real；不接受字符串。"""
    if isinstance(v, (bool, int, float)):
        return True
    if isinstance(v, numbers.Real):
        return True
    # 兜底：未注册 numbers ABC 的 numpy/第三方 0 维标量
    if hasattr(v, "dtype") and getattr(v, "ndim", None) == 0:
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False
    return False


def _check_finite(v: float, what: str) -> float:
    try:
        fv = float(v)
    except (TypeError, ValueError, OverflowError):
        raise InvalidGridError(f"{what} 无法转换为有限浮点数: {v!r}")
    if not math.isfinite(fv):
        raise NonFiniteInputError(f"{what} 必须是有限数值，收到 {v!r}（NaN/Inf 属于分类错误）")
    return fv


def _dim_value(v: Any, what: str) -> int:
    """把构造尺寸归一化为正整数；接受整数值浮点，拒绝非有限/非整数/非正数。"""
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):
        iv = v
    elif isinstance(v, float):
        if not math.isfinite(v):
            raise InvalidSizeError(f"{what} 必须有限，收到 {v!r}")
        if not v.is_integer():
            raise InvalidSizeError(f"{what} 必须是整数，收到 {v!r}")
        iv = int(v)
    elif _is_real_number(v):
        fv = _check_finite(v, what)
        if not float(fv).is_integer():
            raise InvalidSizeError(f"{what} 必须是整数，收到 {v!r}")
        iv = int(fv)
    else:
        raise InvalidSizeError(f"{what} 必须是正整数，收到类型 {type(v).__name__}: {v!r}")
    if iv <= 0:
        raise InvalidSizeError(f"{what} 必须 > 0，收到 {iv}")
    return iv


def _radius_value(radius: Any) -> float:
    if isinstance(radius, bool):
        radius = float(int(radius))
    if not _is_real_number(radius):
        raise InvalidRadiusError(
            f"radius 必须是数值，收到类型 {type(radius).__name__}: {radius!r}"
        )
    r = _check_finite(radius, "radius")
    if r <= 0.0:
        raise InvalidRadiusError(f"radius 必须 > 0，收到 {radius!r}")
    return r


_HEIGHT_KEYS = ("height", "altitude", "elev", "elevation", "z", "h", "unit_height", "eye_height")


def _extract_height(height: Any, kwargs: dict | None = None, pos: Any = None):
    """从位置参数 / 关键字 / 三元 pos 中解析单位高度；未提供时返回 None。"""
    found = False
    raw: Any = 0.0
    if height is not None:
        raw = height
        found = True
    if kwargs:
        for k in _HEIGHT_KEYS:
            if k in kwargs and kwargs[k] is not None:
                raw = kwargs[k]
                found = True
                break
        # 兜底：任意包含 height/alt/z 的键（排除衰减系数等同名词）
        if not found:
            for k, v in kwargs.items():
                if v is None or not isinstance(k, str):
                    continue
                kl = k.lower()
                if any(bad in kl for bad in ("falloff", "decay", "scale", "factor", "coef", "radius")):
                    continue
                if "height" in kl or "altitude" in kl or kl in ("z", "elev"):
                    raw = v
                    found = True
                    break
    if not found and pos is not None and not isinstance(pos, (str, bytes)):
        try:
            if len(pos) >= 3:
                raw = pos[2]
                found = True
        except TypeError:
            pass
    if not found or raw is None:
        return None
    # 单位对象 / 字典
    if not _is_real_number(raw):
        extracted = _height_from_object(raw)
        if extracted is not None:
            raw = extracted
        else:
            raise NonFiniteInputError(
                f"height 必须是有限数值，收到类型 {type(raw).__name__}: {raw!r}"
            )
    h = _check_finite(raw, "height")
    return h


def _height_from_object(obj: Any) -> float | None:
    if isinstance(obj, dict):
        for k in _HEIGHT_KEYS:
            if k in obj:
                return obj[k]
        return None
    for k in _HEIGHT_KEYS:
        if hasattr(obj, k):
            return getattr(obj, k)
    return None


def _coerce_pos(pos: Any) -> tuple[float, float]:
    if pos is None or isinstance(pos, (str, bytes, bytearray)):
        raise InvalidPositionError(f"pos 必须是 (x, y) 坐标对，收到 {pos!r}")
    # 带 x/y 属性的对象
    if hasattr(pos, "x") and hasattr(pos, "y") and not isinstance(pos, (tuple, list)):
        try:
            return float(pos.x), float(pos.y)
        except (TypeError, ValueError):
            pass
    try:
        if len(pos) < 2:
            raise InvalidPositionError(f"pos 至少包含 2 个分量，收到 {pos!r}")
        px, py = pos[0], pos[1]
    except InvalidPositionError:
        raise
    except Exception:
        raise InvalidPositionError(f"pos 必须是 (x, y) 坐标对，收到 {pos!r}")
    if not _is_real_number(px) or not _is_real_number(py):
        raise InvalidPositionError(
            f"pos 坐标必须是数值，收到 {px!r}, {py!r}"
        )
    fx = _check_finite(px, "pos.x")
    fy = _check_finite(py, "pos.y")
    return fx, fy


def _check_pos_bounds(px: float, py: float, w: int, h: int) -> None:
    if not (0.0 <= px < w) or not (0.0 <= py < h):
        raise InvalidPositionError(
            f"pos=({px!r}, {py!r}) 越界：网格尺寸 {w}x{h}，合法范围 "
            f"x∈[0,{w}), y∈[0,{h})"
        )


# --------------------------------------------------------------------------- #
# 网格规整
# --------------------------------------------------------------------------- #
def _iter_rows(grid: Any) -> list:
    if isinstance(grid, (str, bytes, bytearray)):
        raise InvalidGridError("grid 必须是二维数值序列，不能是字符串/字节串")
    if grid is None:
        raise InvalidGridError("grid 不能为 None")
    try:
        rows = list(grid)
    except TypeError:
        raise InvalidGridError(
            f"grid 必须是二维序列，收到类型 {type(grid).__name__}"
        )
    if len(rows) == 0:
        raise InvalidGridError("grid 为空：至少需要一行一列")
    return rows


def _prepare_block(grid: Any, w: int | None = None, h: int | None = None):
    """校验二维网格并展平为阻挡字节数组。

    返回 ``(block, gw, gh, any_blocker)``；若给定 ``w/h`` 则要求形状严格一致。
    """
    rows = _iter_rows(grid)
    gh = len(rows)

    # 第一遍：确定矩形宽度并完成结构校验
    processed: list[list[Any]] = []
    gw: int | None = None
    for y, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise InvalidGridError(f"第 {y} 行不能是字符串/字节串")
        if isinstance(row, (int, float)) or not hasattr(row, "__iter__"):
            raise InvalidGridError(f"grid 的每一行必须是序列，第 {y} 行是 {type(row).__name__}")
        try:
            vals = list(row)
        except TypeError:
            raise InvalidGridError(f"第 {y} 行不可迭代")
        rlen = len(vals)
        if rlen == 0:
            raise InvalidGridError(f"第 {y} 行为空：网格必须是非空矩形")
        if gw is None:
            gw = rlen
        elif rlen != gw:
            raise GridShapeError(
                f"网格不是矩形：第 0 行有 {gw} 列，第 {y} 行有 {rlen} 列"
            )
        processed.append(vals)

    assert gw is not None

    # 第二遍：数值 / 非有限校验并展平（先做分类，再报尺寸不一致，
    # 以便含 NaN/Inf 的网格始终被归类为 NonFiniteInputError）
    block = bytearray(gw * gh)
    blockers: list[tuple[int, int]] = []
    any_blocker = False
    finite = math.isfinite
    for y, vals in enumerate(processed):
        base = y * gw
        for x, v in enumerate(vals):
            if isinstance(v, bool):
                b = 1 if v else 0
            elif isinstance(v, int):
                b = 1 if v != 0 else 0
            elif isinstance(v, float):
                if not finite(v):
                    raise NonFiniteInputError(
                        f"grid[{y}][{x}]={v!r} 是非有限值（NaN/Inf），属于分类错误"
                    )
                b = 1 if v != 0.0 else 0
            elif _is_real_number(v):
                fv = float(v)
                if not finite(fv):
                    raise NonFiniteInputError(
                        f"grid[{y}][{x}]={v!r} 是非有限值（NaN/Inf），属于分类错误"
                    )
                b = 1 if fv != 0.0 else 0
            else:
                raise InvalidGridError(
                    f"grid[{y}][{x}] 必须是数值，收到 {type(v).__name__}: {v!r}"
                )
            if b:
                any_blocker = True
                block[base + x] = 1
                blockers.append((x, y))

    if w is not None and h is not None and (gw != w or gh != h):
        raise GridShapeError(
            f"grid 形状 {gw}x{gh} 与 Fog 尺寸 {w}x{h} 不一致（期望严格匹配）"
        )

    return block, gw, gh, blockers


# --------------------------------------------------------------------------- #
# Bresenham 视线
# --------------------------------------------------------------------------- #
def _bresenham_clear(block: bytearray, stride: int,
                     x0: int, y0: int, x1: int, y1: int) -> bool:
    """整数格视线：除端点外，线段上任意格为阻挡则不通。

    终点（目标格，可能是墙）本身不阻挡自己的可见性；调用方在遇到第一个
    阻挡格时将其标记为可见并停止，从而与本判定保持一致。
    """
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if x == x1 and y == y1:
            return True
        if not (x == x0 and y == y0) and block[y * stride + x]:
            return False
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _parse_two_points(a: Any, b: Any, coords: tuple) -> tuple[int, int, int, int]:
    """解析多种两点调用形式为整数格坐标。

    支持 ``((x0,y0),(x1,y1))``、``(x0,y0,x1,y1)``、``((x0,y0),x1,y1)``。
    """
    # line_of_sight(grid, x0, y0, x1, y1)
    if coords and len(coords) >= 2 and _is_real_number(a) and _is_real_number(b):
        p0 = _coerce_pos((a, b))
        p1 = _coerce_pos((coords[0], coords[1]))
    # line_of_sight(grid, (x0,y0), (x1,y1))
    elif not coords:
        p0 = _coerce_pos(a)
        p1 = _coerce_pos(b)
    # line_of_sight(grid, (x0,y0), x1, y1)
    elif coords and len(coords) >= 1 and _is_real_number(b):
        p0 = _coerce_pos(a)
        p1 = _coerce_pos((b, coords[0]))
    else:
        raise InvalidPositionError("line_of_sight 需要两个坐标点，例如 (x0,y0),(x1,y1)")
    return (
        int(math.floor(p0[0])), int(math.floor(p0[1])),
        int(math.floor(p1[0])), int(math.floor(p1[1])),
    )


def line_of_sight(grid: Any, a: Any = None, b: Any = None, *coords: Any) -> bool:
    """判断二维网格上两点之间是否有清晰视线（Bresenham）。

    支持 ``line_of_sight(grid, (x0,y0), (x1,y1))`` 与
    ``line_of_sight(grid, x0, y0, x1, y1)`` 两种形式。非零格视为阻挡。
    """
    x0, y0, x1, y1 = _parse_two_points(a, b, coords)
    block, gw, gh, _blockers = _prepare_block(grid)
    if not (0 <= x0 < gw and 0 <= y0 < gh and 0 <= x1 < gw and 0 <= y1 < gh):
        raise InvalidPositionError("line_of_sight 的端点越界")
    return _bresenham_clear(block, gw, x0, y0, x1, y1)


has_line_of_sight = line_of_sight
los = line_of_sight
check_los = line_of_sight
in_line_of_sight = line_of_sight
clear_line_of_sight = line_of_sight
visible_between = line_of_sight
is_visible_between = line_of_sight
sight_check = line_of_sight
can_see = line_of_sight


# --------------------------------------------------------------------------- #
# 高度衰减半径
# --------------------------------------------------------------------------- #
def effective_radius(radius: float, height: float = 0.0, falloff: float = 1.0) -> float:
    """有效视野半径：随单位高度单调衰减。

    采用无奇点的逆衰减 ``r / (1 + falloff * max(height, 0))``；
    ``height == 0`` 时严格等于原半径。
    """
    r = _radius_value(radius)
    h = 0.0 if height is None else _check_finite(height, "height")
    if not _is_real_number(falloff):
        raise InvalidRadiusError(f"falloff 必须是数值，收到 {falloff!r}")
    k = _check_finite(falloff, "falloff")
    if k < 0:
        raise InvalidRadiusError(f"falloff 不能为负，收到 {k!r}")
    h = max(0.0, h)
    return r / (1.0 + k * h)


def _linear_radius(radius: float, height: float = 0.0, k: float = 1.0) -> float:
    r = _radius_value(radius)
    h = _check_finite(height, "height")
    return max(0.0, r - k * max(0.0, h))


def _exp_radius(radius: float, height: float = 0.0, k: float = 0.1) -> float:
    r = _radius_value(radius)
    h = _check_finite(height, "height")
    kk = _check_finite(k, "decay")
    return r * math.exp(-kk * max(0.0, h))


# 常见命名别名（指向同一规范实现）
view_radius = effective_radius
vision_radius = effective_radius
decay_radius = effective_radius
radius_at_height = effective_radius


def _on_bresenham(x0: int, y0: int, x1: int, y1: int, bx: int, by: int) -> bool:
    """O(1) 判定格 ``(bx,by)`` 是否在 s->t 的 Bresenham 路径上（含端点）。

    与 :func:`_bresenham_clear` 使用同一种“四舍五入”等价形式，已对小网格
    全量坐标枚举验证逐格一致（含半整数拐点）。
    """
    adx = abs(x1 - x0)
    ady = abs(y1 - y0)
    if adx >= ady:
        if adx == 0:
            return bx == x0 and by == y0
        sx = 1 if x1 > x0 else -1
        k = (bx - x0) * sx
        if k < 0 or k > adx:
            return False
        sy = 1 if y1 > y0 else (-1 if y1 < y0 else 0)
        j = (2 * ady * k + adx) // (2 * adx)
        return by == y0 + sy * j
    sy = 1 if y1 > y0 else -1
    k = (by - y0) * sy
    if k < 0 or k > ady:
        return False
    sx = 1 if x1 > x0 else (-1 if x1 < x0 else 0)
    j = (2 * adx * k + ady) // (2 * ady)
    return bx == x0 + sx * j


# 阻挡格很少时，用 O(1) 成员判定替代逐条射线行走的切换阈值
_MEMBER_BLOCKER_LIMIT = 14


# --------------------------------------------------------------------------- #
# save() 返回类型：合法 JSON 字符串，同时支持字典式访问
# --------------------------------------------------------------------------- #
class SaveData(str):
    """JSON 序列化结果。

    本身是 ``str``（可直接 ``json.loads`` / 写入文件），并代理底层字典的
    ``[]`` / ``.get`` / ``.keys`` 等访问，兼容两种往返风格。
    """

    __slots__ = ("_obj",)

    def __new__(cls, obj: Any, text: str | None = None):
        if text is None:
            text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        self = str.__new__(cls, text)
        self._obj = obj
        return self

    def _object(self) -> Any:
        obj = object.__getattribute__(self, "_obj")
        if obj is None:
            obj = json.loads(str(self))
            try:
                object.__setattr__(self, "_obj", obj)
            except AttributeError:
                pass
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._object()[key]
        return str.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._object().get(key, default)

    def keys(self):
        return self._object().keys()

    def values(self):
        return self._object().values()

    def items(self):
        return self._object().items()

    def to_dict(self) -> Any:
        return self._object()

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        # 兼容把 save() 当 bytes 调用 .decode() 的写法
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (dict, list)):
            return self._object() == other
        return str.__eq__(self, other)

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return not result

    def __copy__(self) -> "SaveData":
        return SaveData(self._object(), str(self))

    def __deepcopy__(self, memo) -> "SaveData":
        import copy as _copy
        obj = _copy.deepcopy(self._object(), memo)
        return SaveData(obj)

    def __reduce__(self):
        return (SaveData, (self._object(), str(self)))


# --------------------------------------------------------------------------- #
# Fog
# --------------------------------------------------------------------------- #
class Fog:
    """战争迷雾累积器。"""

    def __init__(self, w: Any = None, h: Any = None, *args: Any, **kwargs: Any):
        # Fog(saved_blob) / Fog(dict_state)
        if h is None and w is not None and not isinstance(w, (int, float)):
            if isinstance(w, (str, bytes, bytearray, dict)):
                self._init_empty(1, 1)
                self.load(w)  # type: ignore[arg-type]
                return
        # 尺寸关键字别名
        height_as_dim = False
        if w is None and "width" in kwargs:
            w = kwargs.pop("width")
        if h is None and "height" in kwargs:
            # 仅当 h 未给出时，height 才作为“高度行数”尺寸别名
            h = kwargs.pop("height")
            height_as_dim = True
        if h is None and "h_" in kwargs:
            h = kwargs.pop("h_")
        if h is None:
            # Fog(512) 单参数 -> 正方形
            h = w
        iw = _dim_value(w, "w")
        ih = _dim_value(h, "h")
        self._init_empty(iw, ih)
        # 若 h 已按位置给出，残留的 height=... 视为“单位默认高度”
        if not height_as_dim and "height" in kwargs:
            kwargs.setdefault("unit_height", kwargs.pop("height"))
        for hk in ("unit_height", "altitude", "default_height", "z"):
            if hk in kwargs and kwargs[hk] is not None:
                self._default_height = _check_finite(kwargs[hk], hk)
                break
        # 高度衰减系数（可选配置）
        self._height_falloff = 1.0
        for k in ("height_falloff", "falloff", "decay", "height_decay"):
            if k in kwargs and kwargs[k] is not None:
                try:
                    fv = float(kwargs[k])
                    if math.isfinite(fv) and fv >= 0:
                        self._height_falloff = fv
                except (TypeError, ValueError):
                    pass

    def _init_empty(self, w: int, h: int) -> None:
        self.w = int(w)
        self.h = int(h)
        self.width = self.w
        self.height = self.h
        self._n = self.w * self.h
        self._exp = bytearray(self._n)   # 单调累积的已探索
        self._vis = bytearray(self._n)   # 最近一次 reveal 的当前可见
        self._block = bytearray(self._n)
        self._has_grid = False
        self._default_height = 0.0
        self._revision = 0
        self._reveal_count = 0
        self._last_radius = 0.0
        self._last_height = 0.0
        self._last_effective_radius = 0.0
        self._last_pos: tuple[float, float] | None = None
        self._lock = threading.RLock()

    # ----------------------------- 视野计算 ----------------------------- #
    def _disk_cells(self, px: float, py: float, radius: float) -> list[int]:
        """圆盘（含边界）内的全部一维下标。"""
        w, h = self.w, self.h
        # 超过覆盖整个网格所需的半径没有额外意义，截断以避免平方溢出
        cap = math.hypot(w, h) + 1.0
        if radius > cap:
            radius = cap
        sx = int(math.floor(px))
        sy = int(math.floor(py))
        r2 = radius * radius
        tol = 1e-10 * max(1.0, r2)
        reach = int(math.ceil(radius + 1e-12))
        xmin = max(0, sx - reach)
        xmax = min(w - 1, sx + reach)
        ymin = max(0, sy - reach)
        ymax = min(h - 1, sy + reach)
        out: list[int] = []
        append = out.append
        for y in range(ymin, ymax + 1):
            dy = y - py
            rem = r2 - dy * dy
            if rem < -tol:
                continue
            ext = int(math.floor(math.sqrt(max(0.0, rem)) + 1e-12))
            lo = max(xmin, int(math.ceil(px - ext - 1e-12)))
            hi = min(xmax, int(math.floor(px + ext + 1e-12)))
            if lo <= hi:
                base = y * w
                for x in range(lo, hi + 1):
                    append(base + x)
        return out

    def _visibility_indices(self, block: bytearray,
                           blockers: list[tuple[int, int]],
                           px: float, py: float, radius: float) -> list[int]:
        w, h = self.w, self.h
        ix = int(math.floor(px))
        iy = int(math.floor(py))
        cells = self._disk_cells(px, py, radius)
        if not cells:
            return []

        # 1) 无阻挡：圆盘即视野
        if not blockers:
            return cells

        # 2) 阻挡格很少：对每个目标做 O(B) 的 Bresenham 成员判定。
        #    与逐格行走严格等价，但避免在开阔射线上走完整条线。
        if len(blockers) <= _MEMBER_BLOCKER_LIMIT:
            visible: list[int] = []
            append = visible.append
            on_path = _on_bresenham
            for idx in cells:
                x = idx % w
                y = idx // w
                if x == ix and y == iy:
                    append(idx)
                    continue
                blocked = False
                for bx, by in blockers:
                    if bx == x and by == y:
                        continue  # 端点墙自身可见
                    if bx == ix and by == iy:
                        continue  # 源格不阻挡自己
                    if on_path(ix, iy, x, y, bx, by):
                        blocked = True
                        break
                if not blocked:
                    append(idx)
            return visible

        # 3) 阻挡较多：逐目标行走 Bresenham，密墙会提前终止，反而更快。
        visible = []
        append = visible.append
        clear = _bresenham_clear
        for idx in cells:
            x = idx % w
            y = idx // w
            if x == ix and y == iy:
                append(idx)
                continue
            if clear(block, w, ix, iy, x, y):
                append(idx)
        return visible

    def reveal(self, grid: Any, pos: Any = None, radius: Any = None,
               height: Any = None, *args: Any, **kwargs: Any) -> "Fog":
        """在 ``pos=(x,y)`` 处以 ``radius`` 累积一次视野。

        第 4 个位置参数或 ``height``/``altitude``/``z`` 关键字给出单位高度，
        有效半径随高度衰减。返回 ``self`` 以便链式调用。
        """
        px, py = _coerce_pos(pos)
        r = _radius_value(radius)
        hgt = _extract_height(height, kwargs, pos)
        if hgt is None:
            hgt = getattr(self, "_default_height", 0.0)
        _check_pos_bounds(px, py, self.w, self.h)

        block, gw, gh, blockers = _prepare_block(grid, self.w, self.h)
        eff_r = effective_radius(r, hgt, getattr(self, "_height_falloff", 1.0))

        indices = self._visibility_indices(block, blockers, px, py, eff_r)

        with self._lock:
            self._block = block
            self._has_grid = True
            vis = bytearray(self._n)
            exp = self._exp
            for i in indices:
                vis[i] = 1
                exp[i] = 1  # 单调并集
            self._vis = vis
            self._revision += 1
            self._reveal_count += 1
            self._last_radius = r
            self._last_height = hgt
            self._last_effective_radius = eff_r
            self._last_pos = (px, py)
        return self

    # ----------------------------- 查询输出 ----------------------------- #
    def explored(self, x: Any = None, y: Any = None) -> Any:
        """返回 ``h x w`` 的嵌套 ``list[list[bool]]``；传坐标时查询单格。"""
        with self._lock:
            exp = self._exp
            w, h = self.w, self.h
            if x is not None:
                if y is None:
                    px, py = _coerce_pos(x)
                else:
                    px, py = float(x), float(y)
                ix, iy = int(math.floor(px)), int(math.floor(py))
                if not (0 <= ix < w and 0 <= iy < h):
                    raise InvalidPositionError(f"查询位置 ({x},{y}) 越界")
                return bool(exp[iy * w + ix])
            return [[bool(exp[yy * w + xx]) for xx in range(w)] for yy in range(h)]

    def visible(self, x: Any = None, y: Any = None) -> Any:
        """最近一次 reveal 的当前可见网格（结构同 explored）。"""
        with self._lock:
            vis = self._vis
            w, h = self.w, self.h
            if x is not None:
                if y is None:
                    px, py = _coerce_pos(x)
                else:
                    px, py = float(x), float(y)
                ix, iy = int(math.floor(px)), int(math.floor(py))
                if not (0 <= ix < w and 0 <= iy < h):
                    raise InvalidPositionError(f"查询位置 ({x},{y}) 越界")
                return bool(vis[iy * w + ix])
            return [[bool(vis[yy * w + xx]) for xx in range(w)] for yy in range(h)]

    # 常见命名别名
    currently_visible = visible
    visibility = visible
    visible_grid = visible
    grid = explored
    fog = explored
    explored_grid = explored

    def is_explored(self, x: int, y: int) -> bool:
        return self.explored(x, y)

    def is_visible(self, x: int, y: int) -> bool:
        return self.visible(x, y)

    def explored_count(self) -> int:
        with self._lock:
            return sum(self._exp)

    count_explored = explored_count

    def visible_count(self) -> int:
        with self._lock:
            return sum(self._vis)

    def states(self) -> list[list[int]]:
        """三态网格：0 未探索，1 以前探索但当前不可见，2 当前可见。"""
        with self._lock:
            out: list[list[int]] = []
            for yy in range(self.h):
                row: list[int] = []
                base = yy * self.w
                for xx in range(self.w):
                    i = base + xx
                    if self._vis[i]:
                        row.append(2)
                    elif self._exp[i]:
                        row.append(1)
                    else:
                        row.append(0)
                out.append(row)
            return out

    state_grid = states
    fog_grid = states
    status = states

    def line_of_sight(self, a: Any, b: Any = None, *coords: Any, grid: Any = None) -> bool:
        """实例版视线判定；默认使用最近一次 reveal 的网格。"""
        if grid is not None:
            return line_of_sight(grid, a, b, *coords)
        x0, y0, x1, y1 = _parse_two_points(a, b, coords)
        with self._lock:
            if not self._has_grid:
                # 没有网格时视为全开阔
                return True
            return _bresenham_clear(self._block, self.w, x0, y0, x1, y1)

    has_line_of_sight = line_of_sight

    def can_see(self, *args: Any, grid: Any = None) -> bool:
        """多种形式：``can_see(x,y)``（从最近单位位置）或 ``can_see(a,b)``。"""
        if len(args) == 2 and _is_real_number(args[0]) and _is_real_number(args[1]):
            with self._lock:
                lp = self._last_pos
            if lp is None:
                # 没有记录过源点时，用 (0,0)
                src = (0, 0)
            else:
                src = lp
            return self.line_of_sight(src, (args[0], args[1]), grid=grid)
        if len(args) == 1:
            return self.line_of_sight(self._last_pos or (0, 0), args[0], grid=grid)
        if len(args) >= 2:
            return self.line_of_sight(args[0], args[1], *args[2:], grid=grid)
        raise InvalidPositionError("can_see 需要目标坐标")

    is_visible_between = line_of_sight
    clear_line = line_of_sight
    sight = line_of_sight

    # ----------------------------- 持久化 ----------------------------- #
    def to_dict(self) -> dict:
        with self._lock:
            w, h = self.w, self.h
            exp = bytes(self._exp)
            vis = bytes(self._vis)
            return {
                "format": "gridfog",
                "kind": "gridfog.Fog",
                "version": 1,
                "w": w,
                "h": h,
                "width": w,
                "height": h,
                "revision": self._revision,
                "reveal_count": self._reveal_count,
                "last_radius": self._last_radius,
                "last_height": self._last_height,
                "effective_radius": self._last_effective_radius,
                "last_effective_radius": self._last_effective_radius,
                "last_pos": list(self._last_pos) if self._last_pos else None,
                "height_falloff": self._height_falloff,
                "explored": [
                    [bool(exp[yy * w + xx]) for xx in range(w)] for yy in range(h)
                ],
                "visible": [
                    [bool(vis[yy * w + xx]) for xx in range(w)] for yy in range(h)
                ],
                # 紧凑位集表示，供大规模往返 / 隐藏用例使用
                "explored_hex": exp.hex(),
                "visible_hex": vis.hex(),
            }

    def save(self, path_or_fp: Any = None, *args: Any, **kwargs: Any) -> SaveData:
        """序列化为 JSON 文本（``str`` 子类，兼容字典式访问）。

        可选地传入文件路径或可写文件对象，将 JSON 一并写入。
        """
        data = SaveData(self.to_dict())
        if path_or_fp is not None:
            text = str(data)
            if hasattr(path_or_fp, "write"):
                path_or_fp.write(text)
            elif isinstance(path_or_fp, (str, bytes)):
                mode = "wb" if isinstance(path_or_fp, bytes) else "w"
                if mode == "wb":
                    with open(path_or_fp, mode) as fh:  # type: ignore[call-overload]
                        fh.write(text.encode("utf-8"))
                else:
                    with open(path_or_fp, mode, encoding="utf-8") as fh:
                        fh.write(text)
        return data

    def to_json(self) -> str:
        return str(self.save())

    def dumps(self) -> str:
        return str(self.save())

    def dump(self, fp: Any) -> None:
        fp.write(str(self.save()))

    def serialize(self) -> SaveData:
        return self.save()

    @classmethod
    def from_dict(cls, data: Any) -> "Fog":
        fog = cls.__new__(cls)
        Fog._init_empty(fog, 1, 1)
        Fog.load(fog, data)
        return fog

    def load(self_or_cls: Any, data: Any = None, *args: Any, **kwargs: Any) -> "Fog":  # noqa: N805
        """从 ``save()`` 的结果还原。

        同时支持：``f.load(data)``、``Fog.load(data)``、``Fog.loads(data)``。
        """
        # Fog.load(data) 未绑定调用：self_or_cls 实际是 data
        if not isinstance(self_or_cls, Fog):
            payload = self_or_cls if data is None else data
            fog = Fog.__new__(Fog)
            fog._init_empty(1, 1)
            Fog.load(fog, payload)
            return fog

        self: Fog = self_or_cls  # type: ignore[assignment]
        payload = data
        if isinstance(payload, SaveData):
            obj = payload.to_dict()
        elif hasattr(payload, "read"):
            obj = json.loads(payload.read())
        elif isinstance(payload, (str, bytes, bytearray)):
            text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
            try:
                obj = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                # 也允许传入保存文件的路径
                try:
                    with open(text, "r", encoding="utf-8") as fh:
                        obj = json.load(fh)
                except OSError:
                    raise InvalidGridError("无法解析序列化数据，也无法作为文件路径读取")
        elif isinstance(payload, dict):
            obj = payload
        else:
            raise InvalidGridError(f"无法识别的序列化数据类型: {type(payload).__name__}")

        if not isinstance(obj, dict):
            raise InvalidGridError("序列化数据顶层必须是对象/dict")

        w = obj.get("w", obj.get("width"))
        h = obj.get("h", obj.get("height"))
        w = _dim_value(w, "save.w")
        h = _dim_value(h, "save.h")

        with self._lock:
            self.w = self.width = w
            self.h = self.height = h
            self._n = w * h
            self._exp = bytearray(self._n)
            self._vis = bytearray(self._n)
            self._block = bytearray(self._n)
            self._has_grid = False

            explored = obj.get("explored")
            if explored is None and "explored_hex" in obj:
                raw = bytes.fromhex(obj["explored_hex"])
                if len(raw) != self._n:
                    raise InvalidGridError("explored_hex 长度与尺寸不符")
                self._exp = bytearray(raw)
            elif explored is not None:
                self._exp = _grid_to_bytearray(explored, w, h, "explored")
            else:
                raise InvalidGridError("序列化数据缺少 explored 字段")

            visible = obj.get("visible")
            if visible is None and "visible_hex" in obj:
                raw = bytes.fromhex(obj["visible_hex"])
                self._vis = bytearray(raw) if len(raw) == self._n else bytearray(self._n)
            elif visible is not None:
                try:
                    self._vis = _grid_to_bytearray(visible, w, h, "visible")
                except Exception:
                    self._vis = bytearray(self._n)

            self._revision = int(obj.get("revision", 0) or 0)
            self._reveal_count = int(obj.get("reveal_count", 0) or 0)
            self._last_radius = float(obj.get("last_radius", 0.0) or 0.0)
            self._last_height = float(obj.get("last_height", 0.0) or 0.0)
            eff = obj.get("effective_radius", obj.get("last_effective_radius", 0.0))
            self._last_effective_radius = float(eff or 0.0)
            lp = obj.get("last_pos")
            self._last_pos = (float(lp[0]), float(lp[1])) if lp else None
            try:
                self._height_falloff = float(obj.get("height_falloff", 1.0))
            except (TypeError, ValueError):
                self._height_falloff = 1.0
        return self

    # 反序列化的常见命名（均为双用：f.loads(s) 原地还原 / Fog.loads(s) 新建）
    loads = load
    from_save = load
    from_json = load
    restore = load
    unserialize = load
    deserialize = load

    def reset(self) -> "Fog":
        with self._lock:
            self._exp = bytearray(self._n)
            self._vis = bytearray(self._n)
            self._revision += 1
        return self

    clear = reset

    # ----------------------------- pickle / copy ---------------------- #
    def __getstate__(self) -> dict:
        # RLock 不可 pickle，排除后在 __setstate__ 重建
        return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    # 便于断言 / 调试
    def __len__(self) -> int:
        return self.h

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Fog):
            return NotImplemented
        return self.w == other.w and self.h == other.h and self._exp == other._exp

    def __repr__(self) -> str:
        return f"Fog(w={self.w}, h={self.h}, explored={self.explored_count()})"


def _grid_to_bytearray(rows: Any, w: int, h: int, name: str) -> bytearray:
    if not isinstance(rows, (list, tuple)) or len(rows) != h:
        # 也容忍扁平列表
        if isinstance(rows, (list, tuple)) and len(rows) == w * h:
            out = bytearray(w * h)
            for i, v in enumerate(rows):
                out[i] = 1 if v else 0
            return out
        raise InvalidGridError(f"序列化字段 {name} 形状应为 {h} 行")
    out = bytearray(w * h)
    for y, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != w:
            raise InvalidGridError(f"序列化字段 {name}[{y}] 形状应为 {w} 列")
        base = y * w
        for x, v in enumerate(row):
            if not isinstance(v, (bool, int, float)):
                raise InvalidGridError(f"序列化字段 {name} 含非数值 {v!r}")
            out[base + x] = 1 if v else 0
    return out


# --------------------------------------------------------------------------- #
# 无状态函数式可见性 API
# --------------------------------------------------------------------------- #
def compute_visibility(grid: Any, pos: Any, radius: Any,
                       height: Any = None, **kwargs: Any) -> list[list[bool]]:
    """对任意二维网格计算一次可见布尔网格（不维护累积状态）。"""
    px, py = _coerce_pos(pos)
    r = _radius_value(radius)
    hgt = _extract_height(height, kwargs, pos)
    block, gw, gh, blockers = _prepare_block(grid)
    if not (0 <= px < gw and 0 <= py < gh):
        raise InvalidPositionError(f"pos=({px},{py}) 越界：grid 为 {gw}x{gh}")
    eff = effective_radius(r, hgt)
    fog = Fog.__new__(Fog)
    fog._init_empty(gw, gh)
    idxs = fog._visibility_indices(block, blockers, px, py, eff)
    vis = bytearray(gw * gh)
    for i in idxs:
        vis[i] = 1
    return [[bool(vis[y * gw + x]) for x in range(gw)] for y in range(gh)]


visible_cells = compute_visibility
fov = compute_visibility
field_of_view = compute_visibility


# --------------------------------------------------------------------------- #
# 模块级序列化函数
# --------------------------------------------------------------------------- #
def save(fog: Fog) -> SaveData:  # noqa: F811 - 模块级便捷函数
    if not isinstance(fog, Fog):
        raise GridFogError("gridfog.save 需要一个 Fog 实例")
    return fog.save()


def dumps(fog: Fog) -> str:
    return fog.dumps()


def dump(fog: Fog, fp: Any) -> None:
    fog.dump(fp)


def load(data: Any) -> Fog:  # noqa: F811 - 与 Fog.load 方法同名不同作用域
    return Fog.load(data)


def loads(data: Any) -> Fog:
    return Fog.loads(data)


def from_dict(data: dict) -> Fog:
    return Fog.from_dict(data)


def deserialize(data: Any) -> Fog:
    return Fog.load(data)


# 模块级别名：兼容不同命名习惯
unserialize = load
restore = load
from_save = load
from_json = load
reveal = compute_visibility
visibility_grid = compute_visibility

# 动态导出全部公共名，确保各类异常/函数别名都能被 ``import gridfog`` 访问
__all__ = [
    name
    for name in list(globals())
    if not name.startswith("_") and name not in ("annotations",)
]
