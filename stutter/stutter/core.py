"""stutter 核心 API（初始未实现）。"""


class Stutter:
    def __init__(self, rate, gap_ms):
        """构造"""
        raise NotImplementedError

    def detect(self, samples):
        """卡顿段"""
        raise NotImplementedError
