# 超参数敏感性实验运行说明

本文档对应 `10v7.md` 中 `4.6 超参数敏感性分析` 小节，用于生成 IEMOCAP 与 MELD 的补充实验数据。

## 实验脚本

IEMOCAP:

```bash
cd /data/zzb/BaseLine/ten/SDT
GPU_ID=0 ./exec_iemocap_param_sensitivity_wxy.sh
```

MELD:

```bash
cd /data/zzb/BaseLine/ten/SDT
GPU_ID=1 ./exec_meld_param_sensitivity_wxy.sh
```

默认只运行成本最低的质量门控阈值实验：

```bash
SENS_GROUP=quality
```

可选实验组：

| SENS_GROUP | 含义 | 是否需要重新生成 LLM cache |
| --- | --- | --- |
| `quality` | RAG 质量阈值与 LLM 置信度阈值敏感性 | 否，复用 Qwen2.5-7B cache |
| `rag_weight` | RAG 中文本相似度权重 $\lambda$ 敏感性 | 是，除默认 $\lambda=0.7$ 外 |
| `context` | 上下文窗口 $w$ 敏感性 | 是，除默认 $w=3$ 外 |
| `all` | 运行全部三类敏感性实验 | 部分需要 |

## 推荐运行顺序

优先运行质量门控阈值实验，因为不需要重新生成 cache：

```bash
RUNS=3 EPOCHS=150 SENS_GROUP=quality GPU_ID=0 ./exec_iemocap_param_sensitivity_wxy.sh
RUNS=3 EPOCHS=150 SENS_GROUP=quality GPU_ID=1 ./exec_meld_param_sensitivity_wxy.sh
```

再运行 RAG 融合权重实验：

```bash
RUNS=3 EPOCHS=150 SENS_GROUP=rag_weight GPU_ID=0 GEN_GPU_ID=0 ./exec_iemocap_param_sensitivity_wxy.sh
RUNS=3 EPOCHS=150 SENS_GROUP=rag_weight GPU_ID=1 GEN_GPU_ID=1 ./exec_meld_param_sensitivity_wxy.sh
```

最后运行上下文窗口实验：

```bash
RUNS=3 EPOCHS=150 SENS_GROUP=context GPU_ID=0 GEN_GPU_ID=0 ./exec_iemocap_param_sensitivity_wxy.sh
RUNS=3 EPOCHS=150 SENS_GROUP=context GPU_ID=1 GEN_GPU_ID=1 ./exec_meld_param_sensitivity_wxy.sh
```

如果只想快速检查流程，可以设置：

```bash
RUNS=1 EPOCHS=0 SENS_GROUP=quality GPU_ID=0 ./exec_iemocap_param_sensitivity_wxy.sh
```

## 参数范围

IEMOCAP 质量阈值：

```text
(0.50, 0.80)
(0.55, 0.825)
(0.60, 0.85)
(0.65, 0.875)
(0.70, 0.90)
```

MELD 质量阈值：

```text
(0.55, 0.85)
(0.60, 0.875)
(0.65, 0.90)
(0.70, 0.925)
(0.75, 0.95)
```

RAG 文本权重：

```text
0.3, 0.5, 0.7, 0.9, 1.0
```

上下文窗口：

```text
1, 2, 3, 5, 7
```

默认点 `text-rag-weight=0.7` 与 `context-window=3` 会复用已有 Qwen2.5-7B cache。若希望强制重新生成默认点 cache，可以加：

```bash
FORCE_REGENERATE_DEFAULT=1
```

## 结果汇总

实验日志保存在：

```text
/data/zzb/BaseLine/ten/SDT/result/
```

文件名示例：

```text
iemocap_param_sensitivity_quality_20260608_114220.txt
meld_param_sensitivity_quality_20260608_114221.txt
```

将日志解析为逐 run CSV 和均值方差 CSV：

```bash
cd /data/zzb/BaseLine/ten/SDT
/home/zzb/anaconda3/envs/wxy/bin/python parse_param_sensitivity_results.py \
  result/iemocap_param_sensitivity_*.txt \
  --output result/iemocap_param_sensitivity_runs.csv \
  --summary-output result/iemocap_param_sensitivity_summary.csv

/home/zzb/anaconda3/envs/wxy/bin/python parse_param_sensitivity_results.py \
  result/meld_param_sensitivity_*.txt \
  --output result/meld_param_sensitivity_runs.csv \
  --summary-output result/meld_param_sensitivity_summary.csv
```

论文中建议使用 summary CSV 中的 `mean_f1/std_f1/mean_acc/std_acc` 填表，并用 `mean_f1` 绘制敏感性曲线。

## 可视化建议

1. 对 $\lambda$ 和 $w$ 绘制双 Y 轴折线图：左轴为 Weighted-F1，右轴为 cache 生成时间或平均 prompt token 数。
2. 对 $(\delta_q,\delta_c)$ 绘制覆盖率-性能曲线：横轴为通过质量门控的样本比例，纵轴为 Weighted-F1。
3. 在图中标出默认点：IEMOCAP 为 $(0.60,0.85)$，MELD 为 $(0.65,0.90)$，RAG 默认 $\lambda=0.7$，上下文默认 $w=3$。
