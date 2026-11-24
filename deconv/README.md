# deconv

## 题面（可直接粘贴）

子目录：deconv。维纳反卷积。约束：已知高斯 PSF 时反卷积后与原始图 PSNR 提升 ≥3 dB；噪声放大有上界（正则项可调）；频域实现与逐点解一致（≤1e-6）；非法 sigma 可区分报错；100 万像素不超过 800 ms；并发安全。验收：cd deconv && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd deconv && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
