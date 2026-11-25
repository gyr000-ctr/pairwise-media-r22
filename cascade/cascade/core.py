"""级联阴影分割核心 API。

分割模型
--------
视锥区间 ``[near, far]`` 被 ``count`` 个级联划分为 ``count + 1`` 个分割点，
分割点按对数（等比）分布::

    b_i = near * (far / near) ** (i / count)      i = 0..count

因此相邻分割点之比恒为 ``q = (far / near) ** (1 / count)``，即每一级的
覆盖比一致——各级使用相同分辨率的阴影贴图即可获得一致的纹素密度。

相邻级之间的固定比例重叠不改变分割点本身（否则会破坏首尾对整个视锥的
精确覆盖），而是让每一级的 frustum 向相机方向外延 ``lambda_ * 级宽``：
第 ``i`` 条边界处相邻两级共享的重叠区间由 :meth:`Cascade.overlap` 给出。

所有非法输入均抛出 :class:`ValueError`，不同非法情形通过校验顺序与消息
可区分；内部缓存由锁保护，公开方法并发安全。
"""

import math
import threading


class Cascade:
    """级联阴影分割器。"""

    def __init__(self):
        self._lock = threading.Lock()
        # (near, far, count, lambda_) -> 不可变的分割点 tuple
        self._cache = {}

    @staticmethod
    def _finite_real(name, value):
        """校验为有限实数（布尔值显式拒绝），返回 float。"""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} 必须是实数")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} 必须是有限值，不能为 NaN 或无穷")
        return result

    def splits(self, near, far, count, lambda_):
        """返回 ``count + 1`` 个对数分布的分割点（含 near、far）。

        参数
        ----
        near:
            视锥近平面，必须为正的有限实数，且 ``near < far``。
        far:
            视锥远平面，必须为有限实数，且 ``far > near``。
        count:
            级联级数，必须为正整数。
        lambda_:
            相邻级固定重叠比例（占单级宽度的比例），取开区间 ``(0, 1)``。

        返回
        -------
        list[float]
            严格递增的分割点列表，首项恰为 ``near``，末项恰为 ``far``。
            每次调用返回新列表，调用方修改不会影响分割器内部状态。
        """
        near = self._finite_real("near", near)
        far = self._finite_real("far", far)
        lambda_ = self._finite_real("lambda_", lambda_)

        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("count 必须是整数")
        if count <= 0:
            raise ValueError("count 必须为正整数")
        if near <= 0.0:
            raise ValueError("near 必须为正数")
        if far <= near:
            raise ValueError("需满足 0 < near < far")
        if not 0.0 < lambda_ < 1.0:
            raise ValueError("lambda_ 必须位于开区间 (0, 1)")

        key = (near, far, count, lambda_)
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                step = math.log(far / near) / count
                points = [
                    near * math.exp(step * i)
                    for i in range(count + 1)
                ]
                # 首尾锚定到精确的视锥边界，避免浮点漂移。
                points[0] = near
                points[count] = far
                cached = tuple(points)
                self._cache[key] = cached
            return list(cached)

    def overlap(self, i, ratio):
        """返回第 ``i`` 条分割边界处相邻级共享的重叠区间。

        以单级宽度为单位坐标：级 ``k`` 名义上覆盖 ``[k, k + 1]``，向相机
        方向外延 ``ratio`` 后，边界 ``i`` 两侧的级在 ``[i - ratio, i]``
        上重叠，区间长度恰为 ``ratio``。

        参数
        ----
        i:
            分割边界序号，必须为非负整数。
        ratio:
            重叠比例，取半开区间 ``[0, 1)``。

        返回
        -------
        tuple[float, float]
            重叠区间 ``(i - ratio, i)``，纯函数、无副作用、可并发调用。
        """
        if isinstance(i, bool) or not isinstance(i, int) or i < 0:
            raise ValueError("i 必须是非负整数")
        ratio = self._finite_real("ratio", ratio)
        if not 0.0 <= ratio < 1.0:
            raise ValueError("ratio 必须位于半开区间 [0, 1)")
        return (float(i) - ratio, float(i))
