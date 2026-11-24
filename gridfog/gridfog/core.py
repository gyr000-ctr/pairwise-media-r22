"""gridfog 核心 API（初始未实现）。"""


class Fog:
    def __init__(self, w, h):
        """构造"""
        raise NotImplementedError

    def reveal(self, grid, pos, radius):
        """累积视野"""
        raise NotImplementedError

    def explored(self):
        """已探索"""
        raise NotImplementedError

    def save(self):
        """序列化"""
        raise NotImplementedError
