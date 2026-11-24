# cascade

## 题面（可直接粘贴）

子目录：cascade。级联阴影分割。约束：分割点按对数分布且相邻级重叠固定比例；覆盖范围覆盖整个视锥；各级阴影贴图分辨率与覆盖比一致；非法级数可区分报错；1 万次分割不超过 200 ms；并发安全。验收：cd cascade && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd cascade && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
