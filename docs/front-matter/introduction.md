# 引言 {#introduction}

<aside class="page-attribution" aria-label="本页署名">
作者：Haggai Roitman
</aside>

## 宏观图景 {#the-big-picture}

<p>本指南将带你<strong>从第一原理走到生产系统</strong>。它面向想要理解并构建现代 AI 完整技术栈的从业者——研究者、工程师与应用科学家：从 Transformer 架构与运行它们的硬件，到使模型与人类意图对齐并教会它们推理的训练算法，再到把模型作为自主系统部署出去的智能体架构。</p>

<p>核心论点很简单：<em>构建优秀的 AI 系统需要理解整条流水线，而不仅是某一层</em>。调试训练过程的工程师需要理解 GPU 显存层级与优化器动力学。做微调的从业者要知道何时 LoRA 已足够、何时全参数训练值得这个成本。Agent 开发者需要理解底层模型是如何训练出来的。评估框架的技术负责人需要理解每个框架做出了怎样的权衡。本指南提供这幅完整的图景。</p>

## 通往智能体 AI 之路：一段简史 {#the-road-to-agentic-ai-a-brief-history}

<p>今天的智能体 AI 系统并非凭空出现。它们站立在数十年间一系列里程碑式系统之上——每一个都解决了一个更窄的问题，但它们共同积累起的技术、硬件与雄心，最终让自主智能体成为可能。</p>

<ol>
<li><p><strong>Deep Blue（1997）</strong> <a class="book-citation" id="cite-campbell2002deep-1" href="../references.md#ref-campbell2002deep" aria-label="参考文献 37">[37]</a>——IBM 的象棋引擎以暴力搜索（每秒 2 亿个局面）配合手工评估函数，击败了世界冠军 Garry Kasparov。它证明了机器在边界清晰的对抗性领域可以超越人类，但无法泛化到其他任何任务。</p></li>
<li><p><strong>IBM Watson —— Jeopardy!（2011）</strong> <a class="book-citation" id="cite-ferrucci2010building-1" href="../references.md#ref-ferrucci2010building" aria-label="参考文献 90">[90]</a>——Watson 结合了信息检索、NLP 与大规模并行，在开放领域问答中战胜人类冠军。它表明 AI 能够在规模上处理非结构化文本，但需要多年针对领域的工程，且如果没有大量人力，便无法学习新领域。</p></li>
<li><p><strong>AlexNet 与深度学习革命（2012）</strong> <a class="book-citation" id="cite-krizhevsky2012imagenet-1" href="../references.md#ref-krizhevsky2012imagenet" aria-label="参考文献 192">[192]</a>——Krizhevsky 等人的 CNN 在 ImageNet 上以惊人的领先优势夺冠，证明了在 GPU 上训练的深度神经网络可以从原始数据中学到表征。仅此一个结果便引爆了现代深度学习时代，以及最终让 LLM 成为可能的硬件投入。</p></li>
<li><p><strong>AlphaGo（2016）</strong> <a class="book-citation" id="cite-silver2016mastering-1" href="../references.md#ref-silver2016mastering" aria-label="参考文献 331">[331]</a>——DeepMind 的系统用深度 RL（policy 网络 + value 网络 + 蒙特卡洛树搜索）击败了围棋世界冠军 Lee Sedol。与 Deep Blue 的暴力搜索不同，AlphaGo 是<em>学会</em>下棋的——证明了 RL 能够攻克仅靠搜索无法处理的领域（<span class="arithmatex">\(10^{170}\)</span> 个棋盘状态）。随后 AlphaGo Zero（2017） <a class="book-citation" id="cite-silver2017mastering-1" href="../references.md#ref-silver2017mastering" aria-label="参考文献 332">[332]</a> 完全通过自我对弈学习，根本不需要任何人类棋谱。</p></li>
<li><p><strong>GPT-2/GPT-3（2019–2020）</strong> <a class="book-citation" id="cite-brown2020language-1" href="../references.md#ref-brown2020language" aria-label="参考文献 32">[32]</a>——OpenAI 表明，把 decoder-only Transformer 扩展到数十亿参数会涌现出 few-shot learning 能力。GPT-3（175B 参数）仅凭上下文示例，就能完成它从未被显式训练过的任务——翻译、算术、代码生成。基础模型的时代由此开始。</p></li>
<li><p><strong>AlphaFold（2020）</strong> <a class="book-citation" id="cite-jumper2021alphafold-1" href="../references.md#ref-jumper2021alphafold" aria-label="参考文献 174">[174]</a>——DeepMind 解决了悬而未决 50 年的蛋白质折叠问题，以原子级精度预测蛋白质 3D 结构。AlphaFold 证明了深度学习能够攻克此前被认为距离我们还有数十年的根本性科学难题，同时也展示了“架构创新（在残基对上做 attention）+ 大规模算力”组合的威力。</p></li>
<li><p><strong>ChatGPT 与 RLHF（2022）</strong> <a class="book-citation" id="cite-ouyang2022training-1" href="../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a>——InstructGPT/ChatGPT 证明：一个有能力的 base 模型，在经过 RLHF 对齐后，会变成真正有用的助手。这是一个拐点：AI 从研究工具变成了被亿万用户使用的消费产品。其对齐技术（reward 模型、PPO）成为此后所有 LLM 后训练的模板。</p></li>
<li><p><strong>GPT-4 与多模态模型（2023）</strong> <a class="book-citation" id="cite-openai2023gpt4-1" href="../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a>——多模态能力（视觉 + 语言）、更长的上下文与更强的推理能力，把 LLM 推向通用认知。工具使用（代码解释器、浏览网页）已经初见智能体能力的端倪。</p></li>
<li><p><strong>推理模型（2024）</strong> <a class="book-citation" id="cite-deepseek2025r1-1" href="../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a>——OpenAI 的 o1 与 DeepSeek-R1 表明，仅依靠 reward 信号，RL 就能教会模型<em>推理</em>：思维链、回溯与自我验证自发地涌现。模型开始攻克竞赛级数学题与复杂编码任务。</p></li>
<li><p><strong>智能体 AI（2025 至今）</strong>——汇合点到来：具备推理能力的 LLM，配上标准化工具访问（MCP）、智能体间通信（A2A）、持久化记忆与成熟的编排框架。Agent 现在能够自主编写代码、开展研究、管理工作流，并与其他 Agent 协作——正是本指南的主题。</p></li>
</ol>

<div class="admonition tip book-intuition">
<p>每一个里程碑都共享一条共同的轨迹：<strong>架构创新</strong> <span class="arithmatex">\(+\)</span> <strong>规模</strong> <span class="arithmatex">\(+\)</span> <strong>学习信号</strong> <span class="arithmatex">\(=\)</span> <strong>突破</strong>。Deep Blue 用的是手工搜索；AlphaGo 从自我对弈中学习；GPT-3 从互联网文本中学习；今天的智能体系统则从人类反馈、可验证奖励与环境交互中学习。学习信号已经从博弈胜负扩展到开放式的人类偏好——架构也随之成长以与之匹配。</p>
</div>

<p>本指南从基础模型时代接续叙事，并把它推进到对齐、推理与自主智能体阶段。</p>

## 你将看到什么 {#what-you-should-expect}

<p><strong>第一部分：基础</strong>（第 1–3 章）搭建本指南其余部分所依赖的基础知识。我们从 LLM 在内部如何工作开始——决定能力的架构决策——然后覆盖让训练与推理成为可能的硬件与系统，最后从第一原理引入强化学习。</p>

<ul>
<li><p><strong>第 1 章——LLM 架构与优化</strong>：Transformer 内部机制（self-attention、multi-head attention、RoPE、GQA）、FlashAttention、优化方法（AdamW、学习率调度、梯度裁剪）、混合精度、LoRA/QLoRA、量化、知识蒸馏与 Mixture of Experts。</p></li>
<li><p><strong>第 2 章——系统基础</strong>：GPU 架构（A100/H100/B200）、内存层级、NVLink/NVSwitch、分布式训练（FSDP、DeepSpeed ZeRO、张量/流水线并行），以及用于高吞吐推理的 vLLM。</p></li>
<li><p><strong>第 3 章——RL 简介</strong>：MDP、Bellman 方程、TD 学习、Q-learning、policy gradient（REINFORCE）、actor-critic 方法、GAE——支撑第二部分的算法工具箱。</p></li>
</ul>

<p><strong>第二部分：面向 LLM 的 RL 方法</strong>（第 4–12 章）是训练与对齐的核心。在这里你会学到如何对齐、改进与微调语言模型——从完整的数学推导到可运行的代码。</p>

<ul>
<li><p><strong>第 4–8 章</strong>：每一个主流 RL / 偏好算法都配有数学推导、直觉解释与 TRL 代码——PPO、DPO、GRPO，以及偏好优化变体（Online DPO、KTO、IPO、ORPO、SimPO、Best-of-N）。</p></li>
<li><p><strong>第 9–10 章</strong>：Reward 模型训练（Bradley-Terry、scaling laws、reward hacking）与 SFT 最佳实践（数据质量、课程学习、格式化）。</p></li>
<li><p><strong>第 11–12 章</strong>：大规模系统架构（解耦训练、容错、GPU 分配）与 LLM 智能体训练——如何用轨迹级 RL 端到端训练 Agent。</p></li>
</ul>

<p><strong>第三部分：推理</strong>（第 13 章）覆盖模型能力的前沿——教 LLM 解决多步问题。</p>

<ul>
<li><p><strong>第 13 章——大型推理模型的 RL</strong>：DeepSeek-R1、OpenAI o1/o3/o4-mini、QwQ——RL 如何发现思维链、MCTS、过程奖励模型与测试时算力扩展。</p></li>
</ul>

<p><strong>第四部分：评估</strong>（第 14 章）提供测量“上面这些是否真的有效”的方法论。</p>

<ul>
<li><p><strong>第 14 章——LLM 评估</strong>：指标（perplexity、pass@k、ELO）、LLM-as-Judge 模式、污染检测、基准套件，以及智能体评估方法论。</p></li>
</ul>

<p><strong>第五部分：智能体 AI</strong>（第 15–27 章）带你从一个训练好的模型走向一个被部署出去的自主系统。这是篇幅最大的部分，覆盖了一个 Agent 在真实世界中运作所需的一切。</p>

<ul>
<li><p><strong>第 15 章——智能体 AI 简介</strong>：是什么让一个系统具备“智能体性”，从聊天机器人到自主 Agent 的光谱，以及第五部分余下章节所需的基础概念。</p></li>
<li><p><strong>第 16 章——RAG</strong>：检索方法、分块、Embedding 模型、混合检索、重排，以及生产架构。</p></li>
<li><p><strong>第 17 章——记忆系统</strong>：用于 Agent 持久知识的工作记忆、情景记忆、语义记忆与程序性记忆。</p></li>
<li><p><strong>第 18 章——编排</strong>：ReAct、Plan-and-Execute、LLM Compiler、reflexion 模式、上下文管理与 harness 设计。</p></li>
<li><p><strong>第 19 章——循环工程（Loop Engineering）</strong>：推理时强化学习、上下文工程（Context Engineering）、自适应编排循环，以及“循环即策略”（loop-as-policy）抽象。</p></li>
<li><p><strong>第 20 章——设计模式</strong>：prompt chaining、路由、并行化、评估驱动的编排，以及“简洁性原则”。</p></li>
<li><p><strong>第 21 章——环境与基准</strong>：WebArena、SWE-bench、OSWorld、GAIA——评估智能体能力的环境。</p></li>
<li><p><strong>第 22 章——Model Context Protocol (MCP)</strong>：架构、传输层，tool/resource/prompt 原语、安全与部署。</p></li>
<li><p><strong>第 23 章——Agent Skills</strong>：技能库、工具组合与能力抽象。</p></li>
<li><p><strong>第 24 章——A2A 通信</strong>：Google 的 Agent-to-Agent 协议——Agent Cards、任务生命周期、流式通信与企业级模式。</p></li>
<li><p><strong>第 25 章——多智能体系统</strong>：层级式、辩论式、市场式与群体式架构——大规模协作。</p></li>
<li><p><strong>第 26 章——开发框架</strong>：LangGraph、CrewAI、AutoGen、OpenAI Agents SDK、NVIDIA OO Agents——附代码的对比分析。</p></li>
<li><p><strong>第 27 章——智能体 UI</strong>：流式界面、生成式 UI、canvas 范式、工具可视化与人机协同（Human-in-the-Loop）模式。</p></li>
</ul>

<p><strong>第六部分：自测与参考</strong>（第 28–30 章）整合自测材料，并为从业者提供快速查阅的参考。</p>

<ul>
<li><p><strong>第 28 章——测验题与详细解答</strong>：覆盖全部主题——架构、RL、推理、智能体系统——的 108 道题，并配有完整的推导式解答。</p></li>
<li><p><strong>第 29 章——速查手册</strong>：以紧凑的查阅格式呈现关键公式、API 参考、超参数表与失败模式诊断。</p></li>
<li><p><strong>第 30 章——总结与未来方向</strong>：领域将走向何方——开放问题、新兴范式与前路。</p></li>
</ul>

## 现代 AI 流水线 {#the-modern-ai-pipeline}

<p>从 base 模型到部署 Agent 的完整流水线：</p>

<figure id="fig:pipeline">
<a href="../assets/figures/fig_001_pipeline.png" target="_blank" rel="noopener" aria-label="查看现代 AI 流水线原图"><img src="../assets/figures/fig_001_pipeline.png" alt="从预训练基础模型经对齐、推理到智能体系统的现代 AI 流水线" width="1505" height="354" loading="lazy"></a>
<figcaption><span class="figure-number">图 1　</span>现代 LLM 开发流水线：从预训练 base 模型，经由对齐与推理，走向自主的智能体能力（The modern LLM development pipeline: from pre-trained base model through alignment and reasoning to autonomous agentic capability）。每一阶段对应本指南的一个部分。</figcaption>
</figure>
