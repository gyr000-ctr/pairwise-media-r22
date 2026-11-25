"""维纳反卷积（Wiener deconvolution）核心实现。

观测模型（频域圆周卷积）::

    Y(f) = H(f) X(f) + N(f)

每个频率 bin 上的维纳逐点解为::

    X_hat(f) = Y(f) * conj(H(f)) / (|H(f)|^2 + k)

其中 ``k`` 为可调正则系数（噪声/信号功率量级）。滤波器幅度
``|H| / (|H|^2 + k)`` 在全频带的上界为 ``1 / (2 sqrt(k))``，
因此噪声放大被 ``k`` 显式约束，调大 ``k`` 即收紧上界。

实现说明：
- 仅依赖 NumPy，FFT 完成正/反变换，除法按频率 bin 逐点进行，
  与手写逐点解在浮点误差内一致；
- 所有工作数组均为函数内局部变量，实例只保存不可变的 ``k``，
  无共享可变状态，多线程并发调用安全；
- 输入不被原地修改，输出始终为 Python ``list``（二维、float）。
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["Wiener"]


def _check_scalar(value, name, *, allow_zero):
    """校验实值标量参数，返回 float。

    类型错误抛 TypeError；非有限、越界抛 ValueError（信息带参数名，
    便于区分是哪个参数非法）。
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"{name} must be a real number, got {type(value).__name__}"
        )
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    if v < 0.0 or (v == 0.0 and not allow_zero):
        raise ValueError(
            f"{name} must be {'non-negative' if allow_zero else 'positive'}, got {v!r}"
        )
    return v


def _to_array(data, name):
    """把二维类数组输入校验并复制为 float64 数组。

    - 非类数组类型 -> TypeError
    - 空、非二维、阶梯形状、含非数值/非有限值 -> ValueError
    """
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(
            f"{name} must be a 2D array-like of real numbers, "
            f"got {type(data).__name__}"
        )
    try:
        # copy=True：与调用方内存隔离，保证不修改入参。
        arr = np.array(data, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a rectangular 2D array of real numbers"
        ) from exc
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D rectangular array")
    # NaN 与 Inf 分别报错，便于调用方区分非有限输入的具体类别。
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains infinity")
    return arr


class Wiener:
    """维纳反卷积器。

    Parameters
    ----------
    k:
        正则系数，要求非负有限实数。``k=0`` 退化为纯逆滤波（H 零频点
        处置零），``k`` 越大噪声放大上界越小、结果越平滑。
    """

    def __init__(self, k=0.01):
        self._k = _check_scalar(k, "k", allow_zero=True)

    @property
    def k(self):
        """正则系数。"""
        return self._k

    def psf(self, sigma, size):
        """构造归一化二维高斯点扩散函数。

        Parameters
        ----------
        sigma:
            高斯标准差，必须为正有限实数，否则抛 ValueError（信息中
            含 ``sigma``，可与其它参数错误区分）。
        size:
            核边长，必须为正奇数（需要唯一中心像素），否则抛
            ValueError；非整数抛 TypeError。

        Returns
        -------
        list[list[float]]
            ``size x size``、元素和为 1 的对称高斯核。
        """
        sigma = _check_scalar(sigma, "sigma", allow_zero=False)
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
            raise TypeError(
                f"size must be an integer, got {type(size).__name__}"
            )
        size = int(size)
        if size <= 0 or size % 2 == 0:
            raise ValueError(f"size must be a positive odd integer, got {size}")

        center = (size - 1) / 2.0
        axis = np.arange(size, dtype=np.float64) - center
        xx, yy = np.meshgrid(axis, axis)
        kernel = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()
        return kernel.tolist()

    def deconvolve(self, blurred, psf):
        """对模糊图做维纳反卷积。

        Parameters
        ----------
        blurred:
            模糊观测图，非空二维类数组、元素有限。
        psf:
            点扩散函数，非空二维类数组、元素有限，尺寸不超过原图。

        Returns
        -------
        list[list[float]]
            与 ``blurred`` 同形状的复原图（Python float 二维列表）。

        Raises
        ------
        ValueError
            输入为空、非二维、阶梯形状、含 NaN/Inf、PSF 比图大或结果
            非有限时抛出，信息中注明具体来源（blurred image / psf）。
        TypeError
            入参类型不是类数组时抛出。
        """
        image = _to_array(blurred, "blurred image")
        kernel = _to_array(psf, "psf")

        h, w = image.shape
        kh, kw = kernel.shape
        if kh > h or kw > w:
            raise ValueError(
                f"psf shape {(kh, kw)} must not exceed image shape {(h, w)}"
            )

        # PSF 中心平移到频域原点（圆周卷积约定），再零填充到图像尺寸。
        shifted = np.fft.ifftshift(kernel)
        H = np.fft.rfft2(shifted, s=(h, w))
        Y = np.fft.rfft2(image, s=(h, w))

        abs2 = H.real * H.real + H.imag * H.imag
        denominator = abs2 + self._k
        if self._k == 0.0:
            # 纯逆滤波：H 为 0 的频点无信息，滤波器置 0，避免 0/0。
            filt = np.divide(
                np.conj(H),
                denominator,
                out=np.zeros_like(H),
                where=denominator > 0.0,
            )
        else:
            # 维纳逐点解：每个频率 bin 独立的标量维纳增益。
            filt = np.conj(H) / denominator

        estimate = np.fft.irfft2(Y * filt, s=(h, w))
        if not np.all(np.isfinite(estimate)):
            raise ValueError(
                "deconvolution produced non-finite output; increase k"
            )
        return estimate.tolist()
