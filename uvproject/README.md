# uvproject

## 题面（可直接粘贴）

子目录：uvproject。多投影 UV 混合权重。约束：球面/平面/盒式三投影权重和为 1（≤1e-12）；法线微变时权重连续；轴对齐法线退化为单投影；锐度参数单调；非法法线可区分报错；10 万次不超过 200 ms；并发安全。验收：cd uvproject && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd uvproject && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
