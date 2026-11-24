"""uvproject 核心 API（初始未实现）。"""


class UvProject:
    def __init__(self, sharpness=1.0):
        """构造"""
        raise NotImplementedError

    def weights(self, normal):
        """三投影权重"""
        raise NotImplementedError

    def uv(self, p, normal):
        """混合 UV"""
        raise NotImplementedError
