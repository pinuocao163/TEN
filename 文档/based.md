# 2026 年可参考会议/期刊基线

本文档从 `实验结果_总.xlsx` 已出现的会议/期刊中，筛选出已能查到 2026 年官方信息、且与多模态会话情感识别、自然语言处理、人工智能、多媒体理解或情感计算相关的 venue。可用于后续查找 2026 年最新对比基线论文。

## 筛选范围

`实验结果_总.xlsx` 中出现的 venue 包括：

| 类型 | Venue |
| --- | --- |
| NLP / AI 会议 | ACL, EMNLP, NAACL, AAAI, IJCAI, COLING |
| 多媒体 / 检索会议 | ACM MM, SIGIR |
| 情感计算期刊 | TAC / IEEE Transactions on Affective Computing |

## 推荐优先查找的 2026 venue

| Venue | 2026 信息 | 地点 | 与本项目相关性 | 官方/参考链接 |
| --- | --- | --- | --- | --- |
| ACL 2026 | Overall Conference: 2026-07-02 至 2026-07-07 | San Diego, CA, USA | NLP 顶会；适合查找 ERC、LLM reasoning、dialogue understanding、multimodal language understanding 相关最新工作 | https://2026.aclweb.org/ |
| EMNLP 2026 | Main Conference: 2026-10-24 至 2026-10-29 | Budapest, Hungary | NLP 顶会；适合查找对话情感识别、LLM 推理、RAG、情感解释生成等基线 | https://2026.emnlp.org/calls/main_conference_papers/ |
| AAAI 2026 | 2026-01-20 至 2026-01-27 | Singapore | AI 综合顶会；适合查找图建模、知识增强、对比学习、情感识别和多模态推理相关方法 | https://ojs.aaai.org/index.php/AAAI/index |
| IJCAI-ECAI 2026 | 2026-08-15 至 2026-08-21；主会 2026-08-18 至 2026-08-21 | Bremen, Germany | AI 综合顶会；适合查找可解释 AI、知识推理、多模态决策、可靠性建模相关方法 | https://2026.ijcai.org/ |
| ACM MM 2026 | 2026-11-10 至 2026-11-14 | Rio de Janeiro, Brazil | 多媒体顶会；最适合查找 multimodal emotion recognition、audio-visual-language fusion、missing modality、multimodal contrastive learning 等基线 | https://2026.acmmm.org/ |
| SIGIR 2026 | 2026-07-20 至 2026-07-24 | Melbourne / Naarm, Australia | 信息检索顶会；适合查找 RAG、retrieval-augmented reasoning、conversational search、LLM retrieval quality 等与本文 RAG 模块相关的最新工作 | https://sigir2026.org/ |
| IEEE Transactions on Affective Computing | 期刊持续征稿；2026 年仍有卷期记录 | Journal | 情感计算领域核心期刊；适合查找 multimodal affective computing、emotion recognition、emotion reasoning、trustworthy affective systems 等期刊基线 | https://www.computer.org/digital-library/journals/ta/tac-general-call-for-papers |

## 各 venue 查找建议

### ACL 2026

建议关键词：

- conversational emotion recognition
- emotion recognition in conversation
- dialogue emotion reasoning
- multimodal dialogue understanding
- affective reasoning with LLMs
- retrieval augmented dialogue understanding

ACL 更偏 NLP 和对话建模，如果有 2026 年相关论文，适合作为“最新 NLP/LLM 类基线”。

### EMNLP 2026

建议关键词：

- emotion recognition in conversations
- LLM for emotion recognition
- retrieval augmented generation for emotion
- affective dialogue understanding
- explainable emotion recognition

EMNLP 与 ACL 类似，但经验方法、数据驱动方法和 LLM 应用类工作较多。若出现 ERC 或 LLM-enhanced ERC 论文，优先放入对比表。

### AAAI 2026

建议关键词：

- multimodal emotion recognition
- emotion recognition in conversation
- graph neural network emotion recognition
- knowledge enhanced emotion recognition
- contrastive learning emotion recognition
- trustworthy multimodal learning

AAAI 适合作为 AI 综合类强基线来源，尤其是图网络、知识推理、可解释建模、对比学习方向。

### IJCAI-ECAI 2026

建议关键词：

- explainable multimodal learning
- emotion reasoning
- reliability-aware fusion
- cognitive reasoning
- multimodal decision making
- human-centered AI emotion

IJCAI-ECAI 适合查找解释性、可靠性、多源信息融合、认知推理相关方法，可作为本文“机制可解释性”方向的支撑基线。

### ACM MM 2026

建议关键词：

- multimodal emotion recognition
- multimodal sentiment analysis
- audio visual language fusion
- missing modality emotion recognition
- multimodal contrastive learning
- affective computing

ACM MM 是本项目最应该重点查的 2026 venue 之一，因为本文方法本质上是 multimodal emotion recognition / multimodal fusion。

### SIGIR 2026

建议关键词：

- retrieval augmented generation
- conversational retrieval
- retrieval quality
- LLM retrieval reasoning
- multimodal retrieval
- retrieval augmented emotion recognition

SIGIR 未必直接有 ERC 主任务论文，但适合作为 RAG 模块和 retrieval quality 设计的参考来源。

### IEEE Transactions on Affective Computing

建议关键词：

- multimodal affective computing
- emotion recognition in conversation
- affective reasoning
- interpretable affective computing
- trustworthy affective computing
- audio visual emotion recognition

TAC 是情感计算方向非常贴合的期刊。若需要补充期刊类对比基线，应优先检索 TAC 2025-2026 年已发表或 early access 的多模态情感识别论文。

## 建议优先级

| 优先级 | Venue | 原因 |
| --- | --- | --- |
| 高 | ACM MM 2026 | 与多模态情感识别最直接相关 |
| 高 | ACL 2026 / EMNLP 2026 | 与对话理解、LLM 推理、RAG 相关 |
| 高 | IEEE TAC | 情感计算领域期刊基线，适合补充 journal 对比 |
| 中 | AAAI 2026 / IJCAI-ECAI 2026 | AI 综合类强基线，适合查知识、图、可解释、多模态推理 |
| 中 | SIGIR 2026 | 不一定直接有 ERC，但适合支撑 RAG 和检索质量模块 |

## 备注

1. 当前表格中也包含 NAACL 和 COLING，但本次优先保留已能明确查到 2026 年官方会议信息、且与本项目高度相关的 venue。
2. 若后续要更新 `10v1.md` 的 2026 最新基线，建议先查 ACM MM 2026、ACL 2026、EMNLP 2026 和 TAC 2026 early access。
3. 对比基线选择时建议同时满足三个条件：年份新、venue 与表格已有范围一致、任务或方法与 MERC / ERC / RAG / multimodal fusion 相关。
