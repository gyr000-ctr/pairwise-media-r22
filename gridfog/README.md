# gridfog

## 题面（可直接粘贴）

子目录：gridfog。战争迷雾累积与持久化。约束：可见格与视线检测一致；已探索状态单调不丢失；单位视野半径随高度衰减；序列化往返无损（≤1e-12）；非法网格可区分报错；512x512 不超过 500 ms；并发安全。验收：cd gridfog && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gridfog && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
