# 其他实验设计与运行说明

本文档补充 `ablation_experiment_design.md` 之外的实验。主消融实验主要回答“每个模块有没有用”，这里的其他实验主要回答：

1. 模型是否稳定。
2. 参数是否敏感。
3. RAG、VAD、LLM 质量阈值是否合理。
4. 模型是否具有可解释性。
5. 增强模块带来的计算开销是否可接受。

所有结果建议统一保存到：

```bash
/data/zzb/BaseLine/ten/SDT/result
```

运行环境统一使用：

```bash
/home/zzb/anaconda3/envs/wxy/bin/python
```

## 一、重复运行稳定性实验

### 实验目的

验证模型在不同随机初始化和训练波动下是否稳定。论文中建议报告平均值、标准差和最好值，而不是只报告单次最好结果。

### 建议设置

| 数据集 | 运行次数 | epoch | 记录指标 |
| --- | --- | --- | --- |
| IEMOCAP | 5 / 10 / 100 | 150 | Acc, Macro-F1, Weighted-F1 |
| MELD | 5 / 10 / 100 | 150 | Acc, Macro-F1, Weighted-F1 |

### IEMOCAP 运行方式

先小规模测试：

```bash
cd /data/zzb/BaseLine/ten/SDT
RUNS=5 EPOCHS=150 GPU_ID=0 LOG_FILE=/data/zzb/BaseLine/ten/SDT/result/iemocap_stability_5runs.txt bash exec_iemocap_best_100.sh
```

正式运行：

```bash
cd /data/zzb/BaseLine/ten/SDT
RUNS=100 EPOCHS=150 GPU_ID=0 LOG_FILE=/data/zzb/BaseLine/ten/SDT/result/iemocap_stability_100runs.txt bash exec_iemocap_best_100.sh
```

### MELD 运行方式

先小规模测试：

```bash
cd /data/zzb/BaseLine/ten/SDT
RUNS=5 EPOCHS=150 GPU_ID=0 LOG_FILE=/data/zzb/BaseLine/ten/SDT/result/meld_stability_5runs.txt bash exec_meld_best_100.sh
```

正式运行：

```bash
cd /data/zzb/BaseLine/ten/SDT
RUNS=100 EPOCHS=150 GPU_ID=0 LOG_FILE=/data/zzb/BaseLine/ten/SDT/result/meld_stability_100runs.txt bash exec_meld_best_100.sh
```

### 论文中怎么写

可以写成：

```text
To reduce the effect of random initialization, each setting is repeated multiple times, and we report the mean and standard deviation of Acc, Macro-F1 and Weighted-F1.
```

## 二、LLM / RAG 质量阈值敏感性实验

### 实验目的

验证 `--llm-min-quality` 和 `--llm-min-confidence` 的选择是否合理。该实验能支撑“低质量 LLM 推理不应被强行融合”的设计。

### IEMOCAP 推荐网格

固定其他参数，只改变质量阈值：

| 实验名 | llm-min-quality | llm-min-confidence |
| --- | --- | --- |
| Q1 | 0.00 | 0.00 |
| Q2 | 0.40 | 0.70 |
| Q3 | 0.50 | 0.80 |
| Q4 | 0.60 | 0.85 |
| Q5 | 0.70 | 0.90 |

IEMOCAP 单次命令模板：

```bash
cd /data/zzb/BaseLine/ten/SDT/result
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u /data/zzb/BaseLine/ten/SDT/train.py \
  --Dataset IEMOCAP \
  --epochs 150 \
  --batch-size 16 \
  --temp 1 \
  --lr 0.0001 \
  --llm-cache /data/zzb/BaseLine/ten/SDT/data/iemocap_llm_reasoning_mmrag.jsonl \
  --use-llm-reasoning \
  --llm-loss-weight 0.00005 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0.00002 \
  --llm-residual-init 0.001 \
  --llm-min-quality 0.60 \
  --llm-min-confidence 0.85 \
  --grad-clip 0.0 \
  > /data/zzb/BaseLine/ten/SDT/result/iemocap_quality_q4.txt 2>&1
```

### MELD 推荐网格

| 实验名 | llm-min-quality | llm-min-confidence |
| --- | --- | --- |
| Q1 | 0.00 | 0.00 |
| Q2 | 0.50 | 0.80 |
| Q3 | 0.60 | 0.85 |
| Q4 | 0.65 | 0.90 |
| Q5 | 0.70 | 0.95 |

MELD 单次命令模板：

```bash
cd /data/zzb/BaseLine/ten/SDT/result
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u /data/zzb/BaseLine/ten/SDT/train_1.py \
  --Dataset MELD \
  --epochs 150 \
  --batch-size 16 \
  --temp 8 \
  --lr 0.000005 \
  --llm-cache /data/zzb/BaseLine/ten/SDT/data/meld_llm_reasoning_mmrag.jsonl \
  --use-llm-reasoning \
  --llm-loss-weight 0.00001 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0.00003 \
  --llm-residual-init 0.003 \
  --llm-min-quality 0.65 \
  --llm-min-confidence 0.90 \
  > /data/zzb/BaseLine/ten/SDT/result/meld_quality_q4.txt 2>&1
```

### 结果表建议

| Dataset | Quality | Confidence | Acc | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- | --- |
| IEMOCAP | 0.00 | 0.00 |  |  |  |
| IEMOCAP | 0.60 | 0.85 |  |  |  |
| MELD | 0.65 | 0.90 |  |  |  |

## 三、残差强度敏感性实验

### 实验目的

验证认知残差融合的强度是否需要被限制。如果 `--llm-residual-init` 太大，LLM/RAG 信息可能压过原始多模态表示；如果太小，增强信息可能发挥不足。

### 推荐设置

| 数据集 | residual-init 候选 |
| --- | --- |
| IEMOCAP | 0.0001, 0.001, 0.003, 0.01, 0.03 |
| MELD | 0.0003, 0.001, 0.003, 0.01, 0.03 |

### IEMOCAP 命令示例

```bash
cd /data/zzb/BaseLine/ten/SDT/result
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u /data/zzb/BaseLine/ten/SDT/train.py \
  --Dataset IEMOCAP \
  --epochs 150 \
  --batch-size 16 \
  --temp 1 \
  --lr 0.0001 \
  --llm-cache /data/zzb/BaseLine/ten/SDT/data/iemocap_llm_reasoning_mmrag.jsonl \
  --use-llm-reasoning \
  --llm-loss-weight 0.00005 \
  --llm-reliability-weight 0.00001 \
  --vad-contrast-weight 0.00002 \
  --llm-residual-init 0.001 \
  --llm-min-quality 0.60 \
  --llm-min-confidence 0.85 \
  --grad-clip 0.0 \
  > /data/zzb/BaseLine/ten/SDT/result/iemocap_residual_0.001.txt 2>&1
```

### 论文中怎么分析

如果中等残差强度最好，可以说明：

```text
The cognitive residual branch should provide corrective information rather than dominate the original multimodal representation.
```

## 四、损失权重敏感性实验

### 实验目的

验证三个增强损失的权重是否合理：

1. `--llm-loss-weight`：LLM soft-label 蒸馏。
2. `--llm-reliability-weight`：可靠性门控监督。
3. `--vad-contrast-weight`：VAD 引导对比学习。

### 建议先做单因素实验

每次只改一个权重，其他权重保持最佳设置。

#### IEMOCAP

| 参数 | 候选值 |
| --- | --- |
| llm-loss-weight | 0, 0.000005, 0.00001, 0.00005, 0.0001 |
| llm-reliability-weight | 0, 0.000001, 0.000005, 0.00001, 0.0001 |
| vad-contrast-weight | 0, 0.000002, 0.00001, 0.00002, 0.00005 |

#### MELD

| 参数 | 候选值 |
| --- | --- |
| llm-loss-weight | 0, 0.000001, 0.000005, 0.00001, 0.00005 |
| llm-reliability-weight | 0, 0.000001, 0.000005, 0.00001, 0.00005 |
| vad-contrast-weight | 0, 0.000003, 0.00001, 0.00003, 0.00005 |

### 结果图建议

建议画折线图：

```text
x-axis: loss weight
y-axis: Weighted-F1
```

论文中不用把所有数值都放主表，可以把最关键的一组放正文，其余放附录。

## 五、RAG 检索策略实验

### 实验目的

验证多模态 RAG 是否比 text-only RAG 更有效。该实验最能支撑本文“检索增强认知推理”的贡献。

### 当前代码支持情况

训练代码可以直接通过 `--llm-cache` 替换不同 cache 文件来跑；但不同 RAG 策略的 cache 需要先重新生成。

### 推荐 cache 类型

| Cache 名称 | 生成方式 | 目的 |
| --- | --- | --- |
| MM-RAG | 当前默认，多模态统计 + 文本检索 | 完整模型 |
| Text-only RAG | `--text-rag-weight 1.0` | 验证多模态检索是否有用 |
| AV-heavy RAG | `--text-rag-weight 0.3` | 验证音频/视觉统计占比提高后的影响 |
| No-RAG / LLM-only | 不拼接 retrieved examples | 需要小改生成脚本 |

### Text-only RAG cache 生成示例

IEMOCAP：

```bash
cd /data/zzb/BaseLine/ten/SDT
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u generate_llm_reasoning_rag.py \
  --Dataset IEMOCAP \
  --prompts data/iemocap_llm_prompts.jsonl \
  --output data/iemocap_llm_reasoning_textrag.jsonl \
  --model-path /data/LLM/Qwen2.5-7B-Instruct \
  --rag-k 5 \
  --context-window 3 \
  --text-rag-weight 1.0 \
  --dtype bf16 \
  --temperature 0.0
```

MELD：

```bash
cd /data/zzb/BaseLine/ten/SDT
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u generate_llm_reasoning_rag.py \
  --Dataset MELD \
  --prompts data/meld_llm_prompts.jsonl \
  --output data/meld_llm_reasoning_textrag.jsonl \
  --model-path /data/LLM/Qwen2.5-7B-Instruct \
  --rag-k 5 \
  --context-window 3 \
  --text-rag-weight 1.0 \
  --dtype bf16 \
  --temperature 0.0
```

### 使用不同 cache 训练

只需要替换 `--llm-cache`：

```bash
--llm-cache /data/zzb/BaseLine/ten/SDT/data/iemocap_llm_reasoning_textrag.jsonl
```

或：

```bash
--llm-cache /data/zzb/BaseLine/ten/SDT/data/meld_llm_reasoning_textrag.jsonl
```

### 结果表建议

| Dataset | RAG 类型 | Acc | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- |
| IEMOCAP | No RAG / LLM-only |  |  |  |
| IEMOCAP | Text-only RAG |  |  |  |
| IEMOCAP | MM-RAG |  |  |  |
| MELD | Text-only RAG |  |  |  |
| MELD | MM-RAG |  |  |  |

## 六、类别级分析实验

### 实验目的

观察模型主要提升哪些情感类别，尤其是容易混淆的类别：

| 数据集 | 重点类别 |
| --- | --- |
| IEMOCAP | happy / excited, angry / frustrated, neutral / sad |
| MELD | joy / surprise, anger / disgust, neutral / sadness |

### 怎么做

训练日志每 10 个 epoch 会输出 `classification_report` 和 `confusion_matrix`，最终也会输出 best epoch 的分类报告和混淆矩阵。建议从以下日志中提取：

```bash
/data/zzb/BaseLine/ten/SDT/result/iemocap_stability_100runs.txt
/data/zzb/BaseLine/ten/SDT/result/meld_stability_100runs.txt
```

如果只做论文分析，可以选择最好一次或平均表现最接近均值的一次作为可视化对象。

### 论文中建议展示

1. 每类 Precision / Recall / F1。
2. 混淆矩阵。
3. 错误最多的类别对。
4. Full model 与 TEN baseline 的类别级差异。

## 七、可解释性与可靠性分析

### 实验目的

验证可靠性门控是否符合直觉。例如文本清楚时 text 权重更高，语气强烈时 audio 权重更高，表情明显时 visual 权重更高，RAG 质量高时 LLM 分支权重更高。

### 当前代码支持情况

当前 `model.py` 中已有：

```text
last_reliability_weights
last_contrast_features
```

但 `train.py` 目前没有把每个样本的可靠性权重保存到文件。若要做这个实验，需要加一个分析脚本或在 eval 阶段保存：

1. dialogue id。
2. turn id。
3. gold label。
4. pred label。
5. text/audio/visual/LLM 四个可靠性权重。
6. LLM confidence。
7. RAG quality。

### 建议保存格式

```json
{"vid": "Ses01F_impro01", "turn_id": 3, "gold": "frustrated", "pred": "angry", "w_text": 0.31, "w_audio": 0.42, "w_visual": 0.11, "w_llm": 0.16, "confidence": 0.87, "rag_quality": 0.72}
```

### 结果表建议

| 类别 | text 权重 | audio 权重 | visual 权重 | LLM 权重 | RAG quality |
| --- | --- | --- | --- | --- | --- |
| happy / joy |  |  |  |  |  |
| angry / anger |  |  |  |  |  |
| sad / sadness |  |  |  |  |  |
| neutral |  |  |  |  |  |

### 论文中怎么写

可以写成：

```text
We further analyze the learned reliability weights. The model tends to assign larger audio weights to high-arousal emotions and larger LLM weights to utterances with high RAG quality, indicating that the cognitive residual branch acts as a selective correction source.
```

## 八、LLM cache 覆盖率与质量分析

### 实验目的

确认 LLM cache 是否覆盖训练集和测试集。如果 cache 覆盖率低，增强模块效果会不稳定。

### 怎么做

运行任意带 `--llm-cache` 的训练命令时，日志会输出：

```text
LLM reasoning cache coverage: train x/y, test x/y
```

建议把覆盖率记录进论文实验设置：

| Dataset | Train coverage | Test coverage | Cache |
| --- | --- | --- | --- |
| IEMOCAP |  |  | iemocap_llm_reasoning_mmrag.jsonl |
| MELD |  |  | meld_llm_reasoning_mmrag.jsonl |

### 质量分析

可以统计 cache 中 `reasoning_quality.quality` 和 `confidence` 的均值：

```bash
cd /data/zzb/BaseLine/ten/SDT
/home/zzb/anaconda3/envs/wxy/bin/python - <<'PY'
import json
for path in [
    'data/iemocap_llm_reasoning_mmrag.jsonl',
    'data/meld_llm_reasoning_mmrag.jsonl',
]:
    qs, cs = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            cs.append(float(item.get('confidence', 0)))
            rq = item.get('reasoning_quality', {})
            qs.append(float(rq.get('quality', 0) if isinstance(rq, dict) else rq))
    print(path, 'n=', len(qs), 'quality_mean=', sum(qs)/len(qs), 'confidence_mean=', sum(cs)/len(cs))
PY
```

## 九、效率与参数量实验

### 实验目的

说明增强模块带来的额外开销。训练开始时日志已经输出：

```text
total parameters
training parameters
```

每个 epoch 也会输出：

```text
time: xx sec
```

### 建议比较

| 设置 | 参数量 | 单 epoch 时间 | 最佳 Weighted-F1 |
| --- | --- | --- | --- |
| TEN baseline |  |  |  |
| Full model |  |  |  |

### TEN baseline 命令

IEMOCAP：

```bash
cd /data/zzb/BaseLine/ten/SDT/result
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u /data/zzb/BaseLine/ten/SDT/train.py \
  --Dataset IEMOCAP \
  --epochs 150 \
  --batch-size 16 \
  --temp 1 \
  --lr 0.0001 \
  > /data/zzb/BaseLine/ten/SDT/result/iemocap_ten_baseline_efficiency.txt 2>&1
```

MELD：

```bash
cd /data/zzb/BaseLine/ten/SDT/result
CUDA_VISIBLE_DEVICES=0 /home/zzb/anaconda3/envs/wxy/bin/python -u /data/zzb/BaseLine/ten/SDT/train_1.py \
  --Dataset MELD \
  --epochs 150 \
  --batch-size 16 \
  --temp 8 \
  --lr 0.000005 \
  > /data/zzb/BaseLine/ten/SDT/result/meld_ten_baseline_efficiency.txt 2>&1
```

## 十、推荐执行顺序

建议按下面顺序做，先做低成本、最能支撑论文的实验：

| 顺序 | 实验 | 是否必须 | 说明 |
| --- | --- | --- | --- |
| 1 | 重复运行稳定性 | 必须 | 支撑结果可靠性 |
| 2 | 主消融实验 | 必须 | 已在 `ablation_experiment_design.md` 中设计 |
| 3 | 质量阈值敏感性 | 推荐 | 支撑 RAG quality gate |
| 4 | 残差强度敏感性 | 推荐 | 支撑 cognitive residual fusion |
| 5 | 损失权重敏感性 | 推荐 | 支撑三个辅助目标 |
| 6 | 类别级分析 | 推荐 | 支撑模型实际提升点 |
| 7 | LLM cache 覆盖率与质量分析 | 推荐 | 支撑增强数据可靠性 |
| 8 | RAG 检索策略实验 | 可选但很有价值 | 需要生成额外 cache |
| 9 | 可解释性权重分析 | 可选但很有价值 | 需要额外保存权重 |
| 10 | 效率与参数量实验 | 推荐 | 回答额外开销问题 |

## 十一、最终论文中建议保留的表和图

| 类型 | 内容 |
| --- | --- |
| 主结果表 | 与最新 SOTA / baseline 对比 |
| 消融表 | TEN 到 Full model 的模块增量 |
| 稳定性表 | 多次运行 mean/std |
| 敏感性图 | quality threshold、residual-init、loss weight |
| 类别分析图 | confusion matrix 或 per-class F1 |
| 可解释性表 | 不同类别的 text/audio/visual/LLM 权重 |
| 效率表 | 参数量、epoch 时间、性能 |

