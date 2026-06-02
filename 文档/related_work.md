# 相关工作

本文档根据 `实验结果_总.xlsx` 中整理的 IEMOCAP 与 MELD 对比方法，并结合对话情感识别和多模态情感识别中的经典论文，撰写与本文方法相匹配的相关工作。文中引用采用 `[编号]` 标记，编号与文末参考文献一一对应。

## 2 相关工作

### 2.1 会话情感识别

会话情感识别旨在根据对话上下文、说话人交互关系以及话语语义，识别每个话语所表达的情感状态。与传统句子级情感分类不同，会话情感识别需要处理情绪随上下文动态变化、不同说话人之间相互影响、同一话语在不同语境下含义不同等问题。IEMOCAP 数据集为多模态情感分析提供了包含语音、文本、视觉和动作捕捉信息的双人交互数据，被广泛用于会话情感识别和语音情感识别研究 [1]。MELD 在 EmotionLines 的基础上扩展了文本、语音和视觉模态，并引入多方对话场景，成为多模态会话情感识别的重要基准数据集 [2]。

早期会话情感识别方法主要关注上下文建模和说话人状态建模。DialogueRNN 通过维护全局上下文状态、说话人状态和情感状态，显式跟踪不同参与者在对话过程中的情绪变化 [3]。随后，DialogueGCN 将对话建模为图结构，通过图卷积捕获说话人内部依赖和不同说话人之间的交互关系，缓解长距离上下文传播不足的问题 [4]。DAG-ERC 进一步使用有向无环图刻画对话的时间依赖和结构依赖，使历史话语能够以更符合对话流的方式影响当前情感判断 [5]。这些方法说明，ERC 的关键不只是识别当前话语的情感词，还需要建模对话历史、说话人身份和情绪转移过程。

近年来，研究者开始引入更复杂的图结构、认知线索和未来信息来提升 ERC 性能。DualGATs 同时建模 discourse-aware graph 和 speaker-aware graph，以捕获话语之间的篇章结构与说话人依赖 [6]。PFA-ERC 通过预测伪未来信息，为当前话语的动态情感识别提供更充分的上下文补充 [7]。这些方法在 `实验结果_总.xlsx` 中也被列为近期强基线，说明当前 ERC 研究已经从单纯的序列上下文建模，逐渐转向结构化、动态化和认知化的情感建模。

### 2.2 多模态会话情感识别

多模态会话情感识别进一步利用文本语义、语音韵律和视觉表情等信息进行联合判断。经典多模态学习方法如 TFN 通过张量外积显式建模 unimodal、bimodal 和 trimodal 交互 [8]，MFN 通过记忆机制追踪多视角序列中的跨模态动态交互 [9]，MulT 则利用跨模态 Transformer 在未对齐多模态序列之间进行注意力交互 [10]。这些工作为后续多模态情感识别中的跨模态交互建模奠定了基础。

在多模态 ERC 场景中，如何同时捕获会话上下文和跨模态互补信息是核心问题。MMGCN 将文本、语音和视觉信息统一建模到图卷积框架中，同时利用说话人关系和长距离上下文依赖提升多模态融合效果 [11]。SDT 使用模态内和模态间 Transformer 建模多模态交互，并通过层次化门控融合和自蒸馏增强各模态表示 [12]。本文以 SDT/TEN 类 Transformer 多路径结构作为基础骨干，其优势在于能够细粒度建模文本、语音和视觉之间的双向交互；但这类方法的融合权重通常仍主要来自隐空间学习，缺少可追溯的外部证据和可解释的认知中间变量。

针对多模态融合中的模态差异、模态噪声和类别混淆问题，近期方法提出了更细粒度的融合策略。MPT 将多模态提示信息注入 Transformer 注意力层，并结合对比学习提升多模态语义融合效果 [13]。MultiEMO 通过相关性感知的跨模态注意力融合文本、音频和视觉，并使用加权焦点对比损失缓解少数类和语义相近类别识别困难 [14]。DQ-Former 从认知对齐角度引入动态模态优先级，使模型能够根据样本状态选择更可靠的模态信息 [15]。VEGA 通过 CLIP 视觉原型引入类别级视觉语义先验，增强多模态表示的心理语义约束 [16]。这些方法与本文关注点相近，均试图解决多模态融合中的可靠性和语义对齐问题；不同的是，本文进一步把检索证据、LLM 推理、VAD 评价和模态可靠性统一到一个可解释的残差融合链条中。

### 2.3 知识增强、LLM 与检索增强推理

会话情感往往依赖隐含常识、事件原因和说话人心理状态，仅依靠端到端特征学习容易忽略深层语义原因。COSMIC 将常识知识引入会话情感识别，通过建模说话人的心理状态、事件和因果关系增强情感推断能力 [17]。SKIER 进一步结合篇章结构和符号知识图谱，说明外部知识对于理解多方对话中的情绪触发因素具有重要作用 [18]。UniMEEC 将多模态情感识别和情感原因分析统一到同一框架中，利用因果提示模板建模情感与原因之间的内在联系 [19]。MKE-IGN 通过多知识增强交互图网络融合外部知识和多模态上下文，为复杂情感类别提供更强的语义支撑 [20]。

随着大语言模型的发展，LLM 在会话理解、常识推理和情感解释方面展现出较强能力。LaERC-S 利用 LLM 探索说话人特征、心理状态和行为线索，用于改进会话情感识别 [21]。DialogueMMT 借助大视觉语言模型和多任务调优增强对话场景理解，从视觉场景和话语结构两个层面提升多模态 ERC 表现 [22]。这些工作表明，大模型可以为 ERC 提供更丰富的语义和认知线索。然而，现有 LLM 类方法通常直接利用大模型生成的隐式表示、解释或预测结果，仍存在两个问题：一是大模型推理可能与特定数据集的标签边界不完全一致；二是生成解释未必真正参与神经模型的内部决策过程。

检索增强生成为缓解上述问题提供了思路。RAG 通过从外部或任务相关语料中检索证据，再将检索内容提供给生成模型，从而增强生成结果的事实约束和任务适配性 [23]。本文借鉴 RAG 的思想，但并不是将检索增强用于开放域问答，而是从训练集中检索文本语义和音视频状态相似的情感样本，为 LLM 生成结构化认知变量提供数据集内部证据。与直接调用 LLM 进行情感分类不同，本文将 LLM 输出拆解为情感软分布、VAD 评价、模态提示、推理置信度和检索质量，使其成为可量化、可筛选、可参与训练的认知信号。

### 2.4 情感维度建模与可解释可靠融合

离散情感标签虽然便于分类，但不同情感之间并非完全独立。例如 happy 与 excited、angry 与 frustrated、neutral 与 sad 往往具有相近或连续的情感语义。心理学中的 PAD/VAD 情感空间使用 pleasure/valence、arousal 和 dominance 描述情绪状态，为离散类别之间的连续关系提供了理论基础 [24]；Russell 的 circumplex model 进一步强调 valence 与 arousal 是组织情感状态的重要维度 [25]。因此，将 VAD 信息引入 ERC 不仅可以辅助区分易混淆类别，也可以提升模型表示空间的情感语义一致性。

现有多模态融合方法已经开始关注模态可靠性和动态融合。例如 SDT 通过层次化门控学习不同模态权重 [12]，DQ-Former 通过动态模态优先级刻画不同样本中的模态贡献 [15]，MPT 和 MultiEMO 则通过提示、注意力和对比学习增强跨模态语义交互 [13][14]。但这些方法大多仍从神经特征内部学习模态权重，缺少可追溯的检索证据和显式认知变量。本文的核心区别在于：模型不是简单拼接 LLM 表示，也不是直接采用 LLM 分类结果，而是形成“检索证据—认知评价—可靠性分配—残差校正”的决策链条。具体而言，RAG 检索样本提供证据来源，LLM 生成结构化认知变量，VAD 评价约束情感语义空间，质量阈值和置信度控制低质量推理的影响，可靠性门控动态分配 text/audio/visual/LLM 四类信息源的贡献，认知残差则限制外部推理对 TEN 基础表示的修正幅度。

综上，已有 ERC/MERC 研究已经在上下文建模、图结构建模、多模态融合、知识增强和 LLM 推理方面取得了充分进展。然而，如何使大语言模型推理受到任务数据约束，并以可解释、可控、可训练的方式参与多模态情感识别，仍然没有得到充分解决。本文在 TEN 多模态 Transformer 骨干基础上，引入多模态 RAG 增强的 LLM 认知推理、VAD 引导的情感语义约束和可靠性感知认知残差融合，从而在保持多模态识别能力的同时，提高模型预测过程的证据可追溯性、模态可解释性和推理可信度。

## 参考文献

[1] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N. Chang, Sungbok Lee, and Shrikanth S. Narayanan. 2008. IEMOCAP: Interactive emotional dyadic motion capture database. Language Resources and Evaluation, 42(4):335-359.

[2] Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations. ACL 2019.

[3] Navonil Majumder, Soujanya Poria, Devamanyu Hazarika, Rada Mihalcea, Alexander Gelbukh, and Erik Cambria. 2019. DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. AAAI 2019.

[4] Deepanway Ghosal, Navonil Majumder, Soujanya Poria, Niyati Chhaya, and Alexander Gelbukh. 2019. DialogueGCN: A Graph Convolutional Neural Network for Emotion Recognition in Conversation. EMNLP-IJCNLP 2019.

[5] Weizhou Shen, Junqing Chen, Xiaojun Quan, and Zhixian Xie. 2021. Directed Acyclic Graph Network for Conversational Emotion Recognition. ACL/IJCNLP 2021.

[6] Duzhen Zhang, Feilong Chen, and Xiuyi Chen. 2023. DualGATs: Dual Graph Attention Networks for Emotion Recognition in Conversations. ACL 2023.

[7] Tanmay Khule, Rishabh Agrawal, and Apurva Narayan. 2024. PFA-ERC: Psuedo-Future Augmented Dynamic Emotion Recognition in Conversations. Findings of EMNLP 2024.

[8] Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2017. Tensor Fusion Network for Multimodal Sentiment Analysis. EMNLP 2017.

[9] Amir Zadeh, Paul Pu Liang, Navonil Mazumder, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018. Memory Fusion Network for Multi-view Sequential Learning. AAAI 2018.

[10] Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2019. Multimodal Transformer for Unaligned Multimodal Language Sequences. ACL 2019.

[11] Jingwen Hu, Yuchen Liu, Jinming Zhao, and Qin Jin. 2021. MMGCN: Multimodal Fusion via Deep Graph Convolution Network for Emotion Recognition in Conversation. ACL-IJCNLP 2021.

[12] Hui Ma, Jian Wang, Hongfei Lin, Bo Zhang, Yijia Zhang, and Bo Xu. 2024. A Transformer-Based Model With Self-Distillation for Multimodal Emotion Recognition in Conversations. IEEE Transactions on Multimedia, 26:776-788.

[13] Shihao Zou, Xianying Huang, and Xudong Shen. 2023. Multimodal Prompt Transformer with Hybrid Contrastive Learning for Emotion Recognition in Conversation. ACM MM 2023.

[14] Tao Shi and Shao-Lun Huang. 2023. MultiEMO: An Attention-Based Correlation-Aware Multimodal Fusion Framework for Emotion Recognition in Conversations. ACL 2023.

[15] Jing Ye and Xinpei Zhao. 2024. DQ-Former: Querying Transformer with Dynamic Modality Priority for Cognitive-aligned Multimodal Emotion Recognition in Conversation. ACM MM 2024.

[16] Guanyu Hu, Dimitrios Kollias, and Xinyu Yang. 2025. Grounding Emotion Recognition with Visual Prototypes: VEGA - Revisiting CLIP in MERC. ACM MM 2025.

[17] Deepanway Ghosal, Navonil Majumder, Alexander Gelbukh, Rada Mihalcea, and Soujanya Poria. 2020. COSMIC: COmmonSense knowledge for eMotion Identification in Conversations. Findings of EMNLP 2020.

[18] Wei Li, Luyao Zhu, Rui Mao, and Erik Cambria. 2023. SKIER: A Symbolic Knowledge Integrated Model for Conversational Emotion Recognition. AAAI 2023.

[19] Guimin Hu, Zhihong Zhu, Daniel Hershcovich, Lijie Hu, Hasti Seifi, and Jiayuan Xie. 2024. UniMEEC: Towards Unified Multimodal Emotion Recognition and Emotion Cause. Findings of EMNLP 2024.

[20] Geng Tu, Jun Wang, Zhenyu Li, Shiwei Chen, Bin Liang, Xi Zeng, Min Yang, and Ruifeng Xu. 2024. Multiple Knowledge-Enhanced Interactive Graph Network for Multimodal Conversational Emotion Recognition. Findings of EMNLP 2024.

[21] Yumeng Fu, Junjie Wu, Zhongjie Wang, Meishan Zhang, Lili Shan, Yulin Wu, and Bingquan Liu. 2025. LaERC-S: Improving LLM-based Emotion Recognition in Conversation with Speaker Characteristics. COLING 2025.

[22] ChenYuan He, Senbin Zhu, Hongde Liu, Fei Gao, Yuxiang Jia, Hongying Zan, and Min Peng. 2025. DialogueMMT: Dialogue Scenes Understanding Enhanced Multi-modal Multi-task Tuning for Emotion Recognition in Conversations. COLING 2025.

[23] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.

[24] Albert Mehrabian and James A. Russell. 1974. The Basic Emotional Impact of Environments. Perceptual and Motor Skills, 38(1):283-301.

[25] James A. Russell. 1980. A Circumplex Model of Affect. Journal of Personality and Social Psychology, 39(6):1161-1178.
