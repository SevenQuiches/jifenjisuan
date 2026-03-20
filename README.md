# jifenjisuan

数值积分方法比较项目

## 功能概述

实现了三种经典数值积分方法（矩形法、梯形法、辛普森法），用于计算定积分 I = ∫₀^π sin(x) dx。

## 运行方法

```bash
pip install numpy scipy matplotlib
python integral_comparison.py
```

## 收敛阶分析

| 方法 | 理论收敛阶 |
|:---:|:---:|
| 矩形法 | ~1 |
| 梯形法 | ~2 |
| 辛普森法 | ~4 |
