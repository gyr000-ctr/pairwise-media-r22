"""deconv 核心 API（初始未实现）。"""


class Wiener:
    def __init__(self, k=0.01):
        """构造"""
        raise NotImplementedError

    def psf(self, sigma, size):
        """高斯核"""
        raise NotImplementedError

    def deconvolve(self, blurred, psf):
        """反卷积"""
        raise NotImplementedError
