# stutter

## 题面（可直接粘贴）

子目录：stutter。音频卡顿检测。约束：按短时能量突变与零长时间段定位卡顿；卡顿段边界误差 ≤20 ms；正常流误报 ≤2%；双门限迟滞；100 万样本不超过 500 ms；非法参数可区分报错；并发安全。验收：cd stutter && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd stutter && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
