# 证据锚定认知校准：面向多模态会话情感识别的可追溯 RAG-LLM 融合框架

# 摘要

多模态会话情感识别需要在对话上下文中联合理解文本语义、语音韵律和视觉表情。现有 Transformer、图神经网络和提示学习方法能够提升跨模态建模能力，但多数方法仍将融合过程压缩为隐式高维表示：模型给出情感类别，却难以说明它参考了哪些相似案例、为何信任某一模态，以及外部知识在多大程度上改变了多模态判断。针对这一问题，本文提出一种证据锚定认知校准框架（Evidence-Anchored Cognitive Calibration, EACC），构建“检索证据—认知解析—可靠性分配—质量校准”的可追溯决策链。具体而言，本文首先从训练集中检索与当前话语在文本上下文和音视频状态上相似的样本，以数据集内部证据约束大语言模型推理；随后将大语言模型输出解析为结构化认知变量，包括情感软分布、VAD 评价、模态提示、推理置信度和检索质量；进一步地，模型基于这些认知变量动态估计文本、语音、视觉和认知推理四类信息源的可靠性，并通过质量感知校准门控控制认知信息对多模态表示的修正幅度。与此同时，本文利用 VAD 空间构造情感语义约束，使表示学习兼顾离散类别判别和连续情感结构。实验在 IEMOCAP 和 MELD 数据集上进行，结果表明，本文方法能够在保持有竞争力识别性能的同时，提供更清晰的证据来源、模态贡献和推理注入强度，从而提升多模态会话情感识别的机制可解释性与可信度。

# 引言

多模态会话情感识别旨在根据对话中的文本内容、语音韵律、视觉表情和上下文交互关系，识别每个话语所表达的情感状态。该任务是情感计算、自然语言处理和可信人机交互中的重要问题，在智能客服、社交媒体理解、心理健康分析和具身智能交互等场景中具有广泛应用价值。与单句情感分类不同，会话情感识别通常面临三类困难：第一，情感表达具有强上下文依赖，同一句话在不同对话历史和说话人状态下可能呈现不同情绪；第二，文本、语音和视觉并非始终一致，讽刺、沉默、遮挡、噪声和个体表达差异都会造成模态冲突；第三，情感类别之间存在连续语义关系，例如 happy 与 excited、angry 与 frustrated、neutral 与 sad 往往只在唤醒度、控制感或情绪强度上存在细粒度差别。已有研究从说话人关系、情绪转移、图结构上下文和多模态交互等角度提升会话情感识别能力 [1][22][34][53]，但复杂场景下的“为什么这样预测”仍然难以回答。

近年来，基于 Transformer、图神经网络和多模态提示学习的方法显著推动了 MERC 发展。MPT 将多模态提示和混合对比学习引入 Transformer 框架 [70]，DQ-Former 通过动态模态优先级刻画样本级模态贡献差异 [76]，VEGA 利用视觉原型强化情感语义对齐 [47]，超图、关系图和跨模态语义图方法也进一步增强了会话结构建模能力 [39][43][45][51]。然而，这些方法大多仍依赖端到端学习得到的隐式融合权重。即使模型内部存在注意力或门控机制，其权重也往往缺少可验证的外部证据支撑。当模型面对低质量视觉、语音噪声或文本歧义时，研究者很难判断模型究竟是依据相似情感案例、某一模态线索，还是仅仅依赖数据偏置完成预测。这种机制不透明性限制了模型在高可信应用中的使用。

大语言模型展现出的上下文理解、情感解释和常识推理能力为解决上述问题提供了新机会。近期工作已经尝试利用 LLM 建模说话人特征、语音细节、心理状态和对话场景信息，以增强 ERC/MERC 中的高层语义理解能力 [16][31][41][46]。知识增强方法也表明，情绪常识、人格信息、情绪原因和情绪转移知识能够为复杂会话中的情感判断提供重要补充 [6][7][20][58]。然而，直接把 LLM 输出作为情感预测或后验解释并不充分。一方面，LLM 可能过度依赖通用知识，与特定数据集的标签边界和标注风格不一致；另一方面，后验生成的解释并不一定真实参与神经模型决策，仍无法解释模型内部融合过程；此外，当输入证据不足或检索样本质量较低时，LLM 推理可能引入新的噪声。因此，关键问题不是简单“接入 LLM”，而是如何让 LLM 推理受到任务数据约束，并以可度量、可筛选、可控的方式参与多模态决策。

本文围绕这一问题提出证据锚定认知校准框架 EACC。核心思想是将 LLM 从外部预测器转化为可解释的认知信号提供者：模型首先通过多模态 RAG 从训练集中检索相似样本，使 LLM 推理对齐数据集内部标签边界；然后将 LLM 输出解析为情感软分布、VAD 评价、模态提示、置信度和检索质量；最后将这些认知变量注入可靠性融合模块，由模型动态决定文本、语音、视觉和 LLM 认知分支的贡献，并通过校准强度控制认知推理对多模态表示的修正程度。这样，最终预测不再只是一个黑箱分类结果，而是可以沿着检索样本、认知变量、可靠性权重和校准门控逐层追溯。

本文的主要贡献如下。

1. 提出一种证据约束的多模态 RAG-LLM 认知推理机制。该机制联合文本上下文相似度和音视频统计相似度检索训练集样本，使 LLM 推理受到数据集内部案例约束，从而降低其偏离任务标签体系的风险。
2. 构建结构化认知变量，将 LLM 输出从自然语言解释转化为可计算信号。认知变量包含情感软分布、VAD appraisal、模态提示、置信度、检索质量和 VAD 指导向量，既能参与模型训练，也能支撑解释分析。
3. 设计证据锚定认知校准模块。模型根据认知变量动态分配文本、语音、视觉和 LLM 认知分支的可靠性，并通过质量感知校准强度控制认知表示对基础多模态表示的插值式修正，避免低质量推理无条件覆盖多模态判断。
4. 引入 VAD 引导的情感语义约束。本文将离散情感标签与 valence-arousal-dominance 连续空间结合，使模型在区分类别的同时保留情感之间的强度、极性和控制感结构，尤其适用于 happy/excited、angry/frustrated、neutral/sad 等易混淆类别。

# 相关工作

## 2.1 上下文与图结构会话情感识别

会话情感识别的核心在于建模上下文依赖、说话人交互和情绪状态变化。DualGATs 通过双图注意力网络分别建模 discourse-aware graph 和 speaker-aware graph [1]，Curriculum Learning Meets DAG 将有向无环图引入多模态情感识别并结合课程学习 [22]，Adaptive Graph Learning 从数据中自适应学习多模态会话图结构 [34]。近期研究进一步关注高阶关系和复杂交互，Graph Spectrum 从图频谱角度重新审视 MERC 中的上下文传播机制 [38]，Dynamic Interactive Bimodal Hypergraph Networks 使用超图建模双模态高阶关系 [39]，Hybrid Relational Graphs 和 MATCH 分别从情感语义对齐和模态校准超图角度增强多模态融合 [43][45]。这些方法说明结构化上下文对 ERC/MERC 至关重要，但多数方法仍主要在神经特征或图拓扑内部学习关系，缺少可追溯外部证据。

## 2.2 多模态融合与可靠性建模

文本、语音和视觉在不同话语中的可靠性并不相同。CMCF-SRNet 通过跨模态上下文融合和语义细化提升模态交互质量 [3]，MultiEMO 利用相关性感知注意力框架建模多模态关系 [4]，Joyful 将联合模态融合与图对比学习结合起来 [17]。近期研究进一步关注模态平衡、动态优先级和可信输出，MERC Calibration 关注多模态情感识别中的校准问题 [73]，Ada2I 缓解模态不平衡 [74]，DQ-Former 根据样本动态查询不同模态优先级 [76]，DRKF 通过表示解耦和知识融合增强多模态情感识别 [77]。与这些方法相比，本文不仅学习隐式模态权重，还将 LLM 生成的模态提示、置信度、VAD 评价和检索质量显式纳入可靠性分配，使融合权重具备可解释来源。

## 2.3 知识增强、提示学习与 LLM 情感推理

会话情感往往依赖隐含常识、情绪原因、说话人特征和情绪转移规律。ECoK 挖掘情绪常识知识图谱 [6]，EmoTransKG 利用情绪知识图谱揭示情绪转移 [7]，SKIER 将符号知识融入会话情感识别 [32]，MKE-IGN 通过多知识增强交互图网络融合外部知识和多模态上下文 [21]，UniMEEC 将情感识别与情感原因分析统一建模 [20]。随着 LLM 发展，提示学习和 LLM 适配方法也被引入 MERC。MSE-Adapter 设计轻量插件赋予 LLM 多模态情感分析能力 [41]，LaERC-S 利用 LLM 挖掘说话人特征 [31]，Beyond Silent Letters 强调语音细节对 LLM 情感理解的重要性 [16]，DialogueMMT 利用对话场景理解增强多模态多任务调优 [11]。这些方法证明 LLM 可以提供高层语义补充；本文进一步强调，LLM 推理需要被训练集证据约束，并以结构化认知变量的形式真实进入模型内部计算链条。具体而言，本文不是将 LLM 作为独立分类器或后验解释器，而是把 RAG 检索证据、VAD 评价、模态可靠性提示和软标签分布编码为可学习认知特征，使其参与神经模型的前向融合、蒸馏监督和可靠性约束。

# 方法

## 3.1 任务定义与总体框架

给定包含 $N$ 个话语的多模态对话 $D=\{u_1,\ldots,u_N\}$，每个话语包含文本、语音和视觉特征 $u_i=\{x_i^t,x_i^a,x_i^v\}$。模型目标是在上下文中预测话语级情感标签 $\hat{y}_i\in\{1,\ldots,C\}$。传统方法通常直接学习 $(x_i^t,x_i^a,x_i^v)\rightarrow h_i^{base}\rightarrow\hat{y}_i$，其中 $h_i^{base}$ 是隐式多模态表示。本文将其扩展为证据约束的认知决策链：

$$
\mathcal{R}_i\rightarrow c_i,\quad
(h_i^{base},c_i)\rightarrow h_i^{final}\rightarrow \hat{p}_i.
$$

其中，$\mathcal{R}_i$ 是检索证据集合，$c_i$ 是由 RAG-LLM 分支生成的结构化认知变量，$h_i^{final}$ 是 EACC 校准后的最终表示，$\hat{p}_i$ 是情感概率分布。EACC 包含四个阶段：三模态交互编码、多模态 RAG 证据构建、LLM 认知变量解析，以及质量受控的认知校准。LLM 不直接替代分类器，而是被转化为可计算、可筛选、可解释的认知信号。

## 3.2 三模态交互编码器

本文使用预处理好的文本、语音和视觉特征。为消除模态维度差异，模型首先通过一维卷积将三类模态投影到统一隐空间：

$$
h_i^m=\phi_m(x_i^m),\quad m\in\{t,a,v\}.
$$

随后加入位置编码和说话人嵌入：

$$
\tilde{h}_i^m=h_i^m+p_i+s_i.
$$

为了同时建模模态内上下文和跨模态交互，本文构建九条 Transformer 分支：

$$
z_i^{m\leftarrow n}=\operatorname{Transformer}_{m\leftarrow n}(\tilde{H}^n,\tilde{H}^m),
\quad m,n\in\{t,a,v\}.
$$

当 $m=n$ 时，该分支执行模态内上下文建模；当 $m\neq n$ 时，该分支执行跨模态注意力交互。对于每个目标模态，模型拼接来自三个源模态的交互结果，并降维得到目标模态增强表示：

$$
r_i^m=W_m[z_i^{m\leftarrow t};z_i^{m\leftarrow a};z_i^{m\leftarrow v}]+b_m.
$$

基础多模态表示由门控融合得到：

$$
h_i^{base}=\operatorname{Gate}(r_i^t,r_i^a,r_i^v).
$$

该表示是后续 EACC 校准的起点，而不是被 LLM 分支替代。

## 3.3 多模态 RAG 证据构建

LLM 的通用情感知识并不天然等同于 IEMOCAP 或 MELD 的标签边界。为使 LLM 推理对齐目标数据集，本文从训练集中检索相似标注样本作为证据。对目标话语 $u_i$ 和候选训练话语 $u_j$，检索得分定义为：

$$
s(u_i,u_j)=\lambda s_{text}(u_i,u_j)+(1-\lambda)s_{av}(u_i,u_j).
$$

其中，$s_{text}$ 为基于上下文 TF-IDF 的文本相似度，$s_{av}$ 为基于语音能量、变化幅度和视觉动态统计的音视频状态相似度，$\lambda=0.7$。该检索方式同时考虑“说了什么”和“如何表达”，避免只按文本表面相似度选取证据。

最终选取得分最高的 $K$ 个训练样本作为检索证据：

$$
\mathcal{R}_i=\operatorname{TopK}_{u_j\in\mathcal{D}_{train}}s(u_i,u_j).
$$

为了衡量检索证据是否可靠，本文定义检索质量：

$$
\rho_i=0.35s_i^{top}+0.20s_i^{mean}+0.20\max(s_i^{mm},0)+0.25a_i^{label}.
$$

其中，$s_i^{top}$、$s_i^{mean}$、$s_i^{mm}$ 和 $a_i^{label}$ 分别表示最高综合相似度、平均综合相似度、平均音视频相似度以及检索样本标签与 LLM 预测标签的一致比例。$\rho_i$ 越高，说明检索证据越集中、越可信。

## 3.4 结构化认知变量

LLM 接收当前话语、上下文、音视频统计和检索证据后，返回包含 `label`、`confidence`、`rationale`、`appraisal` 和 `modality_hint` 的 JSON 对象。与自由文本解释相比，结构化输出可以被稳定解析为向量，并参与模型训练。

认知特征定义为：

$$
c_i=[p_i^{llm};a_i;q_i;\gamma_i;\rho_i;g_i].
$$

其中，$p_i^{llm}$ 是 LLM 情感软分布，$a_i=[v_i,\alpha_i,d_i,z_i]$ 是由 valence、arousal、dominance 和 certainty 组成的 VAD appraisal，$\alpha_i$ 表示 arousal，$q_i=[q_i^t,q_i^a,q_i^v]$ 是模态提示，$\gamma_i$ 是 LLM 置信度，$\rho_i$ 是检索质量，$g_i$ 是由 VAD appraisal 派生的指导向量。

VAD guidance 由 appraisal 派生：

$$
g_i=[g_i^{pos},g_i^{ctrl},g_i^{low}],
$$

$$
g_i^{pos}=\max(v_i,0)\cdot \alpha_i,
$$

$$
g_i^{ctrl}=\max(-v_i,0)\cdot\max(\alpha_i,0)\cdot d_i,
$$

$$
g_i^{low}=\max(-v_i,0)\cdot\max(-\alpha_i,0).
$$

三维 guidance 分别描述正向高激活、负向高唤醒且带控制感、负向低唤醒三类连续情感因素，用于区分 happy/excited、angry/frustrated 和 neutral/sad 等易混淆情绪。认知特征维度为：

$$
\dim(c_i)=C+4+3+1+1+3=C+12.
$$

IEMOCAP 中 $\dim(c_i)=18$，MELD 中 $\dim(c_i)=19$。

## 3.5 证据锚定认知校准

证据锚定认知校准（EACC）是本文的核心融合模块。该模块不把 LLM 输出直接作为最终预测，而是把结构化认知变量转化为两个可解释控制量：一是文本、语音、视觉和认知分支之间的可靠性分布，二是认知分支对基础多模态表示的校准强度。前者回答模型“信任哪些信息源”，后者回答模型“接受多少外部认知修正”。

首先将认知特征投影到与多模态表示一致的隐空间：

$$
h_i^c=\psi(c_i).
$$

其中，$h_i^c\in\mathbb{R}^{d}$ 表示认知分支隐表示；$\psi(\cdot)$ 表示认知投影网络，由线性层、ReLU 和 LayerNorm 组成；$d=1024$ 是统一隐空间维度。

模型随后计算四类信息源的基础可靠性 logits：

$$
o_i=W_r[r_i^t;r_i^a;r_i^v;h_i^c]+b_r.
$$

其中，$o_i\in\mathbb{R}^{4}$ 表示四个信息源的未归一化可靠性分数；$W_r\in\mathbb{R}^{4\times 4d}$ 和 $b_r\in\mathbb{R}^{4}$ 是可学习参数；四个信息源依次为文本、语音、视觉和 LLM 认知分支。

为使可靠性门控受到认知变量显式影响，本文构造模态提示偏置：

$$
b_i^{hint}=W_h[q_i^t,q_i^a,q_i^v,\tau_i],
$$

$$
\tau_i=\gamma_i(0.5+0.5\rho_i).
$$

其中，$b_i^{hint}\in\mathbb{R}^{4}$ 表示由模态提示产生的门控偏置；$W_h\in\mathbb{R}^{4\times4}$ 是可学习参数；$\tau_i$ 表示 reasoning trust，即 LLM 认知分支可信度；$\gamma_i$ 越高且 $\rho_i$ 越高，$\tau_i$ 越大。

同时构造 VAD 偏置：

$$
b_i^{vad}=W_v[a_i;g_i].
$$

其中，$b_i^{vad}\in\mathbb{R}^{4}$ 表示由 VAD appraisal 和 VAD guidance 产生的门控偏置；$W_v\in\mathbb{R}^{4\times7}$ 是可学习参数；$[a_i;g_i]\in\mathbb{R}^{7}$ 由 4 维 appraisal 和 3 维 guidance 拼接得到。

最终可靠性权重为：

$$
\beta_i=\operatorname{softmax}(o_i+b_i^{hint}+b_i^{vad}).
$$

其中，$\beta_i=[\beta_i^t,\beta_i^a,\beta_i^v,\beta_i^c]\in\mathbb{R}^{4}$；$\beta_i^t$、$\beta_i^a$、$\beta_i^v$ 和 $\beta_i^c$ 分别表示文本、语音、视觉和认知分支的可靠性权重；$\operatorname{softmax}$ 保证四个权重非负且和为 1。

若某个话语不存在 LLM cache，或其检索质量和置信度未通过阈值，认知分支会被 mask。记 mask 为：

$$
M_i^{llm}=\mathbb{I}(\rho_i\ge\delta_q)\cdot\mathbb{I}(\gamma_i\ge\delta_c).
$$

其中，$M_i^{llm}\in\{0,1\}$ 表示 LLM 认知分支是否有效；$\mathbb{I}(\cdot)$ 是指示函数；$\delta_q$ 是检索质量阈值；$\delta_c$ 是置信度阈值。IEMOCAP 使用 $\delta_q=0.60,\delta_c=0.85$，MELD 使用 $\delta_q=0.65,\delta_c=0.90$。

可靠性感知融合表示为：

$$
h_i^{rel}=\beta_i^t r_i^t+\beta_i^a r_i^a+\beta_i^v r_i^v+\beta_i^c h_i^c.
$$

其中，$h_i^{rel}\in\mathbb{R}^{d}$ 表示可靠性感知融合表示；$\beta_i^c h_i^c$ 是认知分支的贡献项；当 $M_i^{llm}=0$ 时，模型抑制 $\beta_i^c$，避免低质量认知信息进入融合。

为了避免 LLM 推理直接覆盖多模态判断，本文采用质量感知的插值校准：

$$
h_i^{final}=h_i^{base}+\eta_i(h_i^{rel}-h_i^{base}).
$$

其中，$h_i^{final}$ 表示最终用于分类的表示；$h_i^{rel}-h_i^{base}$ 表示可靠性融合结果相对于基础多模态表示的校准方向；$\eta_i\in[0,1]$ 表示校准强度，决定模型接受多少认知修正。

校准强度定义为：

$$
\eta_i=\sigma(\theta_\eta)\cdot(0.5+\sigma(W_\eta[\gamma_i;\rho_i]+b_\eta))\cdot \gamma_i(0.5+0.5\rho_i)\cdot M_i^{llm}.
$$

其中，$\sigma(\cdot)$ 表示 sigmoid 函数；$\theta_\eta$ 是可学习全局校准参数；$W_\eta\in\mathbb{R}^{1\times2}$ 和 $b_\eta\in\mathbb{R}$ 是校准控制器参数；$[\gamma_i;\rho_i]$ 是置信度和检索质量拼接向量；$M_i^{llm}$ 保证未通过质量门控的认知信息不会产生表示校准。实现中 `llm-residual-init` 会被限制在 $[10^{-4},0.99]$，从而避免初始校准幅度过大或完全为零。

最终分类概率为：

$$
\hat{p}_i=\operatorname{softmax}(W_o h_i^{final}+b_o).
$$

其中，$W_o\in\mathbb{R}^{C\times d}$ 和 $b_o\in\mathbb{R}^{C}$ 是分类器参数；$\hat{p}_i=[\hat{p}_{i1},\ldots,\hat{p}_{iC}]$ 是最终情感概率分布；$\hat{p}_{ik}$ 表示话语 $u_i$ 属于第 $k$ 类情感的概率。

该模块的创新性在于，LLM 认知信息同时影响“信任谁”和“改多少”：可靠性权重 $\beta_i$ 决定文本、语音、视觉和认知分支的相对贡献，校准强度 $\eta_i$ 决定认知融合结果对基础多模态表示的修正幅度。因此，模型既能吸收 LLM 的认知推理，又能在推理质量不足时回退到多模态编码器判断。

## 3.6 训练目标

设 $\Omega$ 为 mini-batch 中所有有效话语集合。本文将训练目标组织为监督判别、认知对齐和情感几何三部分：

$$
\mathcal{L}
=\mathcal{L}_{sup}
+\lambda_{cog}\mathcal{L}_{cog}
+\lambda_{aff}\mathcal{L}_{aff}.
$$

监督判别目标继承基础多模态编码器的分类监督、单模态监督和自蒸馏约束：

$$
\mathcal{L}_{sup}
=\mathcal{L}_{cls}
+\lambda_{uni}\mathcal{L}_{uni}
+\lambda_{sd}\mathcal{L}_{sd}.
$$

其中，$\mathcal{L}_{cls}$ 是最终融合分支的交叉熵损失，$\mathcal{L}_{uni}$ 约束文本、语音和视觉单模态分支保持判别能力，$\mathcal{L}_{sd}$ 使用融合分支的 softened distribution 蒸馏单模态分支。

认知对齐目标由 LLM 分布蒸馏和可靠性监督组成：

$$
\mathcal{L}_{cog}
=\lambda_{llm}\mathcal{L}_{llm}
+\lambda_{rel}\mathcal{L}_{rel},
$$

$$
\mathcal{L}_{llm}
=\frac{\sum_{i\in\Omega} w_i^{llm}
\operatorname{KL}(p_i^{llm}\Vert \hat{p}_i)}
{\max(\sum_{i\in\Omega} w_i^{llm},1)}.
$$

其中，$p_i^{llm}$ 是 LLM 情感软分布，$w_i^{llm}=M_i^{cache}M_i^{llm}\rho_i\gamma_i$。该权重使模型只从存在 cache、通过质量门控且置信度较高的认知样本中学习。可靠性监督 $\mathcal{L}_{rel}$ 则利用 LLM 模态提示与 VAD 模态先验构造目标可靠性分布，约束模型的 $\beta_i$ 与认知变量一致。

情感几何目标利用 VAD 空间补充离散标签监督。模型首先拉近同类别且 VAD 接近的样本，再对 IEMOCAP 中 happy/excited、angry/frustrated 和 neutral/sad 等易混淆类别施加间隔约束：

$$
\mathcal{L}_{aff}=\mathcal{L}_{vad}
=\mathcal{L}_{pos}+\mathcal{L}_{neg}.
$$

其中，$\mathcal{L}_{pos}$ 强化同标签且 VAD 相近样本的聚合，$\mathcal{L}_{neg}$ 拉开易混淆情感对，使表示空间同时符合离散类别判别和连续情感语义。

# 实验

## 4.1 数据集与评价指标

本文在 IEMOCAP 和 MELD 两个公开数据集上进行实验。IEMOCAP 包含 happy、sad、neutral、angry、excited 和 frustrated 六类情感，测试集包含 1623 个有效话语。MELD 包含 neutral、surprise、fear、sadness、joy、disgust 和 anger 七类情感，测试集包含 2610 个有效话语。评价指标采用 Accuracy、Macro-F1 和 Weighted-F1。考虑到两个数据集均存在类别不均衡，本文主要以 Weighted-F1 作为整体性能指标，同时报告 Accuracy 和 Macro-F1。

表 1 数据集与评价设置

| 数据集 | 情感类别数 | 测试话语数 | 主要评价指标 |
| --- | --- | --- | --- |
| IEMOCAP | 6 | 1623 | Weighted-F1 / Accuracy / Macro-F1 |
| MELD | 7 | 2610 | Weighted-F1 / Accuracy / Macro-F1 |

## 4.2 实现细节

基础编码器采用三模态 Transformer 交互结构，隐藏维度为 1024，dropout 为 0.5，注意力头数为 8，权重衰减为 1e-5，训练轮数为 150。认知 cache 离线生成，默认 LLM 为 Qwen2.5-7B-Instruct；RAG 使用 $K=5$，上下文窗口为 3，文本检索权重为 0.7，生成温度为 0。训练阶段不加载 LLM，只读取 jsonl cache。

实验中，IEMOCAP 使用 `train.py`，batch size 为 16，学习率为 1e-4，自蒸馏温度为 1；MELD 使用 `train_1.py`，batch size 为 8，学习率为 5e-6，自蒸馏温度为 8。完整模型超参数如表 2 所示。

表 2 完整模型主要超参数

| 数据集 | 训练入口 | lr | temp | batch | LLM loss | Reliability loss | VAD contrast | Calibration init | 质量阈值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | train.py | 1e-4 | 1 | 16 | 5e-6 | 1e-6 | 2e-6 | 1e-6 | quality >= 0.60, confidence >= 0.85 |
| MELD | train_1.py | 5e-6 | 8 | 8 | 1e-6 | 1e-6 | 3e-6 | 0.0003 | quality >= 0.65, confidence >= 0.90 |

实现中会将初始校准系数限制在 $[10^{-4},0.99]$ 区间内。因此，当给定初始值小于 $10^{-4}$ 时，模型内部有效初始校准强度为 $10^{-4}$。质量阈值和置信度阈值同时影响认知分支 mask、LLM 蒸馏、可靠性监督和 VAD 对比损失。

## 4.3 主实验结果

表 3 给出完整模型结果。IEMOCAP 上模型取得 74.99% Weighted-F1 和 74.86% Accuracy；MELD 上取得 67.62% Weighted-F1 和 68.58% Accuracy。结果表明，认知推理信息在质量门控和 EACC 校准下能够有效补充多模态表示，同时避免低质量 LLM 推理无条件覆盖基础多模态判断。

表 3 完整模型结果

| 数据集 | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- |
| IEMOCAP | 74.86 | 74.48 | 74.99 |
| MELD | 68.58 | 51.90 | 67.62 |

表 4 选取近年来具有代表性的 ERC/MERC 方法作为对比基线。为保证比较口径一致，主表仅保留同时报告 IEMOCAP F1 和 MELD F1 的方法；若原论文未报告 Accuracy，则以 “-” 标记。所选基线优先满足三个条件：一是属于 ERC/MERC 中被广泛比较的经典方向，二是覆盖 ACL、EMNLP、NAACL、AAAI、IJCAI、ACM MM、COLING 等主流会议中的较新工作，三是评价指标能够与本文的 Weighted-F1 结果直接对照。表中方法按发表年份排序。实验结果表明，本文方法在 IEMOCAP 上优于 MPT、DialogueMMT、MCGDiff、Graph Spectrum、Dynamic Interactive Bimodal Hypergraph Networks 和 LaERC-S 等方法，并接近 Hybrid Relational Graphs、VEGA 等 2025 年强基线。MELD 上本文方法低于 DialogueMMT、LaERC-S、Graph Spectrum 等专门优化的最新方法，但仍保持有竞争力的整体性能。此外，本文方法额外提供检索证据、可靠性权重和认知校准强度，使性能结果具备更好的机制解释支撑。

表 4 与现有方法对比

| 方法 | 会议/期刊 | 年份 | IEMOCAP Acc. | IEMOCAP F1 | MELD Acc. | MELD F1 |
| --- | --- | --- | --- | --- | --- | --- |
| MultiEMO [4] | ACL | 2023 | - | 72.84 | - | 66.74 |
| Joyful [17] | EMNLP | 2023 | 70.55 | 71.03 | 62.53 | 61.77 |
| MPT [70] | ACM MM | 2023 | 72.83 | 72.51 | 65.86 | 65.02 |
| Disentanglement and Fusion [71] | ACM MM | 2023 | 71.84 | 71.75 | 68.28 | 67.03 |
| PFA-ERC [19] | EMNLP | 2024 | - | 77.34 | - | 68.63 |
| UniMEEC [20] | EMNLP | 2024 | - | 74.83 | - | 68.96 |
| MKE-IGN [21] | EMNLP | 2024 | 72.03 | 71.93 | 67.78 | 66.56 |
| MERC Calibration [73] | ACM MM | 2024 | - | 71.98 | - | 66.85 |
| DQ-Former [76] | ACM MM | 2024 | 71.68 | 71.76 | 64.88 | 64.70 |
| CEPT [25] | COLING | 2024 | - | 70.53 | - | 67.51 |
| DialogueMMT [11] | ACL | 2025 | 72.58 | 72.71 | 71.19 | 70.66 |
| MCGDiff [14] | NAACL | 2025 | 72.77 | 73.19 | 67.94 | 68.78 |
| Graph Spectrum [38] | AAAI | 2025 | - | 73.90 | - | 69.00 |
| Dynamic Interactive Bimodal Hypergraph Networks [39] | AAAI | 2025 | 72.58 | 72.46 | 68.01 | 66.61 |
| Fine-grained Dynamic Multimodal Analysis [46] | ACM MM | 2025 | - | 74.02 | - | 68.71 |
| Hybrid Relational Graphs [43] | IJCAI | 2025 | 75.48 | 75.47 | 68.05 | 66.83 |
| VEGA [47] | ACM MM | 2025 | 76.02 | 75.58 | 69.72 | 68.54 |
| LaERC-S [31] | COLING | 2025 | - | 72.40 | - | 69.27 |
| 本文方法 | - | 2026 | 74.86 | 74.99 | 68.58 | 67.62 |

## 4.4 消融与敏感性分析

为了分析 EACC 中认知信号强度对模型性能的影响，本文比较 Light EACC 与 Strong EACC 两组设置。Light EACC 使用较小的 LLM 蒸馏、可靠性监督和 VAD 对比损失权重；Strong EACC 将三类认知相关权重同步放大。两组实验均使用相同的数据划分、训练轮数、RAG cache 和评价流程。表 5 按多次运行中每次最佳 Weighted-F1 统计最高值、均值和标准差。

表 5 EACC 权重敏感性结果

| 数据集 | 设置 | LLM loss | Rel. loss | VAD loss | 运行数 | Best Acc. | Best W-F1 | Mean Best W-F1 | Std Best W-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | Light EACC | 5e-6 | 1e-6 | 2e-6 | 72 | 74.49 | 74.72 | 73.99 | 0.33 |
| IEMOCAP | Strong EACC | 5e-5 | 1e-5 | 2e-5 | 73 | 74.43 | 74.55 | 73.85 | 0.71 |
| MELD | Light EACC | 1e-6 | 1e-6 | 3e-6 | 15 | 67.85 | 67.35 | 67.17 | 0.12 |
| MELD | Strong EACC | 1e-5 | 1e-5 | 3e-5 | 15 | 68.24 | 67.33 | 67.13 | 0.14 |

结果显示，Light EACC 在两个数据集上均取得略高且更稳定的平均最佳 Weighted-F1，Strong EACC 的方差更大。这说明 LLM 认知信息更适合作为质量受控的辅助校准信号，而不应以过大的认知损失强行覆盖基础多模态编码器。

## 4.5 LLM Cache 质量分析

为进一步分析 EACC 是否依赖特定 LLM cache，本文比较 Qwen2.5 系列不同规模模型生成的结构化认知 cache。表 6 统计 cache 覆盖数、平均置信度、通过置信度阈值的比例、平均 RAG 质量和标签一致性。该表衡量的是当前 prompt、RAG 检索和结构化解析规则下的 cache 可用性，而不是 LLM 通用能力排名。

表 6 Qwen2.5 系列 cache 诊断统计

| 数据集 | LLM cache | 样本数 | 平均置信度 | Conf. >= 0.85 | Conf. >= 0.90 | 平均 RAG 质量 | RAG >= 0.60 | RAG >= 0.65 | 标签一致性 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEMOCAP | Qwen2.5-1.5B | 7433 | 0.8500 | 100.00 | 0.00 | 0.5555 | 34.97 | 21.36 | 0.4666 |
| IEMOCAP | Qwen2.5-3B | 7433 | 0.8938 | 100.00 | 49.91 | 0.5813 | 43.39 | 26.03 | 0.5697 |
| IEMOCAP | Qwen2.5-7B | 7433 | 0.8380 | 73.05 | 2.46 | 0.5732 | 40.84 | 24.19 | 0.5375 |
| IEMOCAP | Qwen2.5-14B | 7433 | 0.8648 | 90.08 | 38.68 | 0.5787 | 42.47 | 25.25 | 0.5595 |
| MELD | Qwen2.5-1.5B | 13708 | 0.8501 | 100.00 | 0.05 | 0.5729 | 38.63 | 21.97 | 0.4560 |
| MELD | Qwen2.5-3B | 13708 | 0.9027 | 100.00 | 60.66 | 0.5703 | 39.22 | 22.29 | 0.4457 |
| MELD | Qwen2.5-7B | 13708 | 0.8380 | 72.43 | 3.23 | 0.5689 | 38.54 | 21.95 | 0.4400 |
| MELD | Qwen2.5-14B | 4046 | 0.8550 | 87.39 | 22.89 | 0.5926 | 48.42 | 28.45 | 0.4619 |

表 6 中 Qwen2.5-3B 在若干 cache 诊断指标上高于 Qwen2.5-7B。主要原因是 EACC 会通过置信度和检索质量筛选认知信息，当前 7B cache 的置信度更保守，导致通过高置信门控的样本比例较低；而 3B 在当前生成设置下具有更集中的置信度分布和更高的标签一致性。该差异反映的是 cache 可用性和置信度校准差别，并不意味着 3B 的通用推理能力强于 7B。最终 LLM backbone 选择仍应以下游 Accuracy、Macro-F1 和 Weighted-F1 为准。

## 4.6 可解释性分析

EACC 的解释性来自模型内部计算链条，而不是预测后的自然语言解释。对于任一话语，模型可以输出如下证据链：

$$
\mathcal{R}_i\rightarrow c_i\rightarrow \beta_i\rightarrow \eta_i\rightarrow \hat{p}_i.
$$

其中，$\mathcal{R}_i$ 表示检索到的训练集相似样本，$c_i$ 表示 LLM 结构化认知变量，$\beta_i$ 表示文本、语音、视觉和认知分支的可靠性权重，$\eta_i$ 表示认知校准强度，$\hat{p}_i$ 表示最终情感概率分布。该链条可以从三个层面解释模型行为。第一，证据层展示模型参考了哪些训练样本，从而判断 LLM 推理是否受到合理案例约束。第二，认知层展示 LLM label、confidence、VAD appraisal 和 modality hint，便于分析外部认知信号是否与当前话语一致。第三，融合层展示 $\beta_i$ 和 $\eta_i$，用于判断模型是否真正采纳了 LLM 认知信息，以及采纳程度是否与检索质量和推理置信度一致。

从机制上看，高 RAG 质量和高置信度样本应对应更大的认知分支权重和校准强度；当语音韵律对情感判断更关键时，$\beta_i^a$ 应上升；当文本语义更稳定时，$\beta_i^t$ 应占主导。对于 IEMOCAP 中 happy/excited、angry/frustrated 和 neutral/sad 等易混淆类别，VAD appraisal 与 VAD 对比约束可以为模型提供连续情感空间中的极性、唤醒度和控制感信息，从而使错误分析不再停留在最终类别层面，而能进一步追踪到证据质量、模态可靠性和认知校准强度。

# 结论

本文针对多模态会话情感识别中融合过程难解释、模态贡献难追踪和 LLM 推理难控制的问题，提出证据锚定认知校准框架 EACC。该框架通过多模态 RAG 检索训练集内部相似案例，为 LLM 情感推理提供证据约束；同时将 LLM 输出解析为情感软分布、VAD 评价、模态提示、置信度和检索质量等结构化认知变量，并通过可靠性门控和质量感知校准机制注入多模态表示。实验结果表明，本文方法在 IEMOCAP 和 MELD 上保持有竞争力的识别性能，同时能够输出可追溯的证据链、模态可靠性和认知校准强度。未来工作可进一步扩展到更多 LLM backbone、缺失模态场景以及基于可靠性权重的交互式情感解释。

# 参考文献

[1] DualGATs: Dual Graph Attention Networks for Emotion Recognition in Conversation. ACL 2023.

[3] CMCF-SRNet: A Cross-Modality Context Fusion and Semantic Refinement Network for Emotion Recognition in Conversation. ACL 2023.

[4] MultiEMO: An Attention-Based Correlation-Aware Multimodal Fusion Framework for Emotion Recognition in Conversations. ACL 2023.

[6] ECoK: Emotional Commonsense Knowledge Graph for Mining Emotional Gold. ACL 2024.

[7] EmoTransKG: An Innovative Emotion Knowledge Graph to Reveal Emotion Transformation. ACL 2024.

[8] Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition. ACL 2024.

[11] DialogueMMT: Dialogue Scenes Understanding Enhanced Multi-modal Multi-task Tuning for Emotion Recognition in Conversations. ACL 2025.

[12] TelME: Teacher-leading Multimodal Fusion Network for Emotion Recognition in Conversation. NAACL 2024.

[13] Emotion-Anchored Contrastive Learning Framework for Emotion Recognition in Conversation. NAACL 2024.

[14] Multi-Condition Guided Diffusion Network for Multimodal Emotion Recognition in Conversation. NAACL 2025.

[15] PEMV: Improving Spatial Distribution for Emotion Recognition in Conversation Using Proximal Emotion Mean Vectors. NAACL 2025.

[16] Beyond Silent Letters: Amplifying LLMs in Emotion Recognition with Vocal Nuances. NAACL 2025.

[17] Joyful: Joint Modality Fusion and Graph Contrastive Learning for Multimodal Emotion Recognition. EMNLP 2023.

[19] PFA-ERC: Pseudo-Future Augmented Dynamic Emotion Recognition in Conversations. EMNLP 2024.

[20] UniMEEC: Towards Unified Multimodal Emotion Recognition and Emotion Cause. EMNLP 2024.

[21] Multiple Knowledge-Enhanced Interactive Graph Network for Multimodal Conversational Emotion Recognition. EMNLP 2024.

[22] Curriculum Learning Meets Directed Acyclic Graph for Multimodal Emotion Recognition. COLING 2024.

[25] CEPT: a Contrast-Enhanced Prompt-Tuning Framework for Emotion Recognition in Conversation. COLING 2024.

[30] A Dual Contrastive Learning Framework for Enhanced Multimodal Conversational Emotion Recognition. COLING 2025.

[31] LaERC-S: Improving LLM-based Emotion Recognition in Conversation with Speaker Characteristics. COLING 2025.

[32] SKIER: A Symbolic Knowledge Integrated Model for Conversational Emotion Recognition. AAAI 2023.

[34] Adaptive Graph Learning for Multimodal Conversational Emotion Detection. AAAI 2024.

[38] Revisiting Multimodal Emotion Recognition in Conversation from the Perspective of Graph Spectrum. AAAI 2025.

[39] Dynamic Interactive Bimodal Hypergraph Networks for Emotion Recognition in Conversations. AAAI 2025.

[41] MSE-Adapter: A Lightweight Plugin Endowing LLMs with the Capability to Perform Multimodal Sentiment Analysis and Emotion Recognition. AAAI 2025.

[43] Hybrid Relational Graphs with Sentiment-laden Semantic Alignment for Multimodal Emotion Recognition in Conversation. IJCAI 2025.

[45] MATCH: Modality-Calibrated Hypergraph Fusion Network for Conversational Emotion Recognition. IJCAI 2025.

[46] From Subtle Hints to Grand Expressions - Mastering Fine-grained Emotions with Dynamic Multimodal Analysis. ACM MM 2025.

[47] Grounding Emotion Recognition with Visual Prototypes: VEGA - Revisiting CLIP in MERC. ACM MM 2025.

[48] Hardness-Aware Dynamic Curriculum Learning for Robust Multimodal Emotion Recognition with Missing Modalities. ACM MM 2025.

[51] A Multi-Level Alignment and Cross-Modal Unified Semantic Graph Refinement Network for Conversational Emotion Recognition. TAC 2024.

[53] Multi-Party Conversation Modeling for Emotion Recognition. TAC 2024.

[57] Contrastive Learning Based Modality-Invariant Feature Acquisition for Robust Multimodal Emotion Recognition With Missing Modalities. TAC 2024.

[58] PIRNet: Personality-Enhanced Iterative Refinement Network for Emotion Recognition in Conversation. TAC 2024.

[61] Generalizing to Unseen Speakers: Multimodal Emotion Recognition in Conversations With Speaker Generalization. TAC 2025.

[62] LineConGraphs: Line Conversation Graphs for Effective Emotion Recognition Using Graph Neural Networks. TAC 2025.

[64] Towards Speaker-Unknown Emotion Recognition in Conversation via Progressive Contrastive Deep Supervision. TAC 2025.

[66] Dynamic Causal Disentanglement Model for Dialogue Emotion Detection. TAC 2025.

[70] Multimodal Prompt Transformer with Hybrid Contrastive Learning for Emotion Recognition in Conversation. ACM MM 2023.

[71] Revisiting Disentanglement and Fusion on Modality and Context in Conversational Multimodal Emotion Recognition. ACM MM 2023.

[73] Multimodal Emotion Recognition Calibration in Conversations. ACM MM 2024.

[74] Ada2I: Enhancing Modality Balance for Multimodal Conversational Emotion Recognition. ACM MM 2024.

[75] Multimodal Fusion via Hypergraph Autoencoder and Contrastive Learning for Emotion Recognition in Conversation. ACM MM 2024.

[76] DQ-Former: Querying Transformer with Dynamic Modality Priority for Cognitive-aligned Multimodal Emotion Recognition in Conversation. ACM MM 2024.

[77] DRKF: Decoupled Representations with Knowledge Fusion for Multimodal Emotion Recognition. ACM MM 2025.
