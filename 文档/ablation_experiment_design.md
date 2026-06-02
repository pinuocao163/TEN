# 消融实验设计建议

本文档基于当前项目代码和 `10.docx` 的方法描述，整理适合 IEMOCAP 与 MELD 的消融实验设置。

## 核心消融目标

消融实验建议围绕论文中的四个核心贡献展开：

1. 多模态 RAG 增强认知推理。
2. 结构化认知变量，包括 LLM 情感软分布、VAD 评价、模态提示、推理置信度和检索质量。
3. 可靠性感知认知残差融合。
4. VAD 引导的情感语义对比学习。

## 主消融表

| 编号 | 实验名 | 目的 | 当前代码是否能直接跑 |
| --- | --- | --- | --- |
| A0 | TEN baseline | 原始 TEN，不使用 LLM/RAG/VAD | 可以 |
| A1 | + Cognitive Residual Fusion | 只加入 LLM/RAG 认知残差融合，不加额外 loss | 可以 |
| A2 | + LLM Distillation | 加入 LLM 情感软分布蒸馏 | 可以 |
| A3 | + Reliability Supervision | 加入可靠性门控监督 | 可以 |
| A4 | + VAD Contrast | 加入 VAD 引导对比学习 | 可以 |
| A5 | Full Model | 完整模型 | 可以 |
| A6 | Full w/o Quality Gate | 去掉质量阈值，验证 RAG 质量过滤是否有效 | 可以 |
| A7 | Full w/ Weak Residual | 将残差强度设得极小，验证认知残差是否起作用 | 可以，近似 |

## IEMOCAP 命令模板

### A0: TEN Baseline

```bash
cd /data/zzb/BaseLine/ten/SDT
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u train.py \
  --Dataset IEMOCAP \
  --epochs 150 \
  --batch-size 16 \
  --temp 1 \
  --lr 0.0001
```

### A1: Only Cognitive Residual Fusion

```bash
cd /data/zzb/BaseLine/ten/SDT
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u train.py \
  --Dataset IEMOCAP \
  --epochs 150 \
  --batch-size 16 \
  --temp 1 \
  --lr 0.0001 \
  --llm-cache data/iemocap_llm_reasoning_mmrag.jsonl \
  --use-llm-reasoning \
  --llm-loss-weight 0 \
  --llm-reliability-weight 0 \
  --vad-contrast-weight 0 \
  --llm-residual-init 0.01 \
  --llm-min-quality 0.60 \
  --llm-min-confidence 0.85
```

### A2: Add LLM Distillation

```bash
--llm-loss-weight 0.00005 \
--llm-reliability-weight 0 \
--vad-contrast-weight 0
```

### A3: Add Reliability Supervision

```bash
--llm-loss-weight 0.00005 \
--llm-reliability-weight 0.0001 \
--vad-contrast-weight 0
```

### A4/A5: Full Model

```bash
--llm-loss-weight 0.00005 \
--llm-reliability-weight 0.0001 \
--vad-contrast-weight 0.00002
```

### A6: Full w/o Quality Gate

```bash
--llm-min-quality 0.0 \
--llm-min-confidence 0.0
```

### A7: Full w/ Weak Residual

```bash
--llm-residual-init 0.0001
```

## MELD 命令模板

MELD 使用相同的消融结构，但基础参数换成历史记录中更适合 MELD 的设置。

```bash
cd /data/zzb/BaseLine/ten/SDT
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u train.py \
  --Dataset MELD \
  --epochs 150 \
  --batch-size 16 \
  --temp 1 \
  --lr 0.00005 \
  --llm-cache data/meld_llm_reasoning_mmrag.jsonl \
  --use-llm-reasoning \
  --llm-loss-weight 0.00001 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0.00003 \
  --llm-residual-init 0.03 \
  --llm-min-quality 0.65 \
  --llm-min-confidence 0.90
```

对应消融时只需要替换以下权重：

```bash
# A1
--llm-loss-weight 0 \
--llm-reliability-weight 0 \
--vad-contrast-weight 0

# A2
--llm-loss-weight 0.00001 \
--llm-reliability-weight 0 \
--vad-contrast-weight 0

# A3
--llm-loss-weight 0.00001 \
--llm-reliability-weight 0.00001 \
--vad-contrast-weight 0

# A4/A5
--llm-loss-weight 0.00001 \
--llm-reliability-weight 0.00001 \
--vad-contrast-weight 0.00003
```

## 适合论文的附加消融

### RAG 消融

建议比较：

| 设置 | 含义 |
| --- | --- |
| MM-RAG | 当前多模态 RAG cache |
| Text-only RAG | 只用文本相似度检索生成 cache |
| No RAG / LLM-only | 不提供检索样本，只让 LLM 根据上下文输出认知变量 |

当前训练代码可以通过替换 `--llm-cache` 直接读取不同 cache。需要额外生成不同版本的 LLM cache。

### VAD 信息消融

建议比较：

| 设置 | 含义 |
| --- | --- |
| Full | 使用 VAD appraisal + VAD contrast |
| w/o VAD contrast | 设置 `--vad-contrast-weight 0` |
| w/o VAD appraisal | 需要加代码开关，将 `llm_features` 中 VAD/appraisal 切片置零 |

`w/o VAD contrast` 当前代码可以直接跑；`w/o VAD appraisal` 需要额外加参数开关。

### 可解释性分析

除 F1/Accuracy 外，建议额外统计：

1. 平均可靠性权重：text/audio/visual/LLM。
2. 残差强度 `alpha` 的均值与分布。
3. 不同情感类别下的可靠性权重分布。
4. 高质量 RAG 与低质量 RAG 样本上的性能差异。

这些分析能直接支撑 `10.docx` 中关于“检索证据、认知评价、可靠性分配、残差校正”的可解释性主张。

## 推荐论文表格顺序

论文主消融表建议按下面顺序组织：

```text
TEN
TEN + RAG-LLM cognitive fusion
+ LLM soft-label distillation
+ reliability supervision
+ VAD-aware contrastive learning
Full model
Full model w/o quality gate
Full model w/ weak residual
```

这个顺序与方法贡献逐项对应，能够清楚展示每个模块对最终性能的影响。
