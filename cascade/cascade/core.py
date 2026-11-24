"""cascade 核心 API（初始未实现）。"""


class Cascade:
    def splits(self, near, far, count, lambda_):
        """分割点"""
        raise NotImplementedError

    def overlap(self, i, ratio):
        """重叠区间"""
        raise NotImplementedError
