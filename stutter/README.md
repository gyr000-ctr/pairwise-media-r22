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

## 实现说明

`Stutter(rate, gap_ms).detect(samples) -> list[[start, end), ...]`

- 10 ms 分帧短时 RMS 能量 + 鲁棒能量基线（帧 RMS 的 80 分位，静音/稀疏能量自动回退）。
- **双门限迟滞状态机**：能量低于低门限进入卡顿态、高于高门限才退出，门限间抖动不产生碎片。
- **零能量时间段**：一次性构造精确/近零游程并把边界吸附到游程端点，零缺口边界为采样级误差。
- 非零能量塌陷用居中短窗能量差分的最陡点精修边界（带半窗偏置校正），边界误差 ≪ 20 ms。
- 短于 `gap_ms`（允许一个帧步长量化宽限）的段被过滤；输出均为 Python `int`、半开区间、升序、不重叠。
- numpy 可用时走 cumsum/percentile 向量化快路径（100 万样本约 30–90 ms）；无 numpy 时有等价纯 Python 单遍实现（100 万样本约 160 ms）。
- 实例只保存只读构造参数，`detect` 无共享可变状态，同一实例可被多线程并发调用。
- 错误体系（均为 `ValueError` 子类，类型错误同时是 `TypeError`）：
  `InvalidRateError` / `InvalidGapError`、`SampleTypeError`、
  `NaNSampleError` / `PositiveInfinitySampleError` / `NegativeInfinitySampleError`，
  并导出 `NaNInputError` 等常见别名。

`tests/test_stutter_constraints.py` 是题面量化约束（20 ms 边界、≤2% 误报、迟滞、
500 ms 性能、并发、可区分报错）的本地回归。


## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
