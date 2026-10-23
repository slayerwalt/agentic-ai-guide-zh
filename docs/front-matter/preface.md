# 前言 {#preface}

<aside class="page-attribution" aria-label="本页署名">
作者：Haggai Roitman
</aside>

## 为什么要写这本指南 {#why-this-guide-exists}

<p>在 2026 年构建智能 AI 系统，需要掌握极其广阔的知识——从 Transformer 在内部如何处理语言，到使训练成为可能的硬件与系统，再到让训练高效的优化技术、教会模型推理并与人类意图对齐的强化学习算法，乃至大规模协调自主系统的多智能体架构。</p>

<p>这些知识散落在数百篇论文、博客文章、GitHub 仓库，以及少数实验室内部的“部落知识”之中。本指南之所以存在，是因为<strong>从业者需要一份覆盖整套技术栈的统一参考</strong>——不仅是理论，更是让事情真正跑起来的实现细节。</p>

## 我个人走向智能体 AI 的旅程 {#a-personal-journey-to-agentic-ai}

<p>我对智能体（agent）的迷恋始于二十年前，那时我还在攻读信息系统工程的本科学位。我选修了面向智能体的软件工程（Agent-Oriented Software Engineering, AOSE） <a class="book-citation" id="cite-wooldridge2000gaia-1" href="../references.md#ref-wooldridge2000gaia" aria-label="参考文献 382">[382]</a> 课程，并学习如何用 JADE <a class="book-citation" id="cite-bellifemine2007jade-1" href="../references.md#ref-bellifemine2007jade" aria-label="参考文献 22">[22]</a>（Java Agent DEvelopment Framework）构建多智能体系统——这是一个遵循 FIPA <a class="book-citation" id="cite-fipa2002acl-1" href="../references.md#ref-fipa2002acl" aria-label="参考文献 93">[93]</a> 标准的平台，智能体之间通过结构化协议通信、就共享资源协商，并自主协作。与此同时，Berners-Lee、Hendler 与 Lassila 发表的奠基性论文《The Semantic Web》 <a class="book-citation" id="cite-bernerslee2001semantic-1" href="../references.md#ref-bernerslee2001semantic" aria-label="参考文献 24">[24]</a> 描绘了一种机器可读知识的愿景，让智能体能够在其上推理。这两条线索——自主智能体架构与语义知识表示——种下了一颗种子，此后一直引导着我的职业生涯。一个早期项目让这一愿景具体化：在我未来尊敬的学术导师 Prof. Avigdor Gal 的指导下，我们尝试用 OntoBuilder <a class="book-citation" id="cite-gal2006ontobuilder-1" href="../references.md#ref-gal2006ontobuilder" aria-label="参考文献 103">[103]</a> 构建一个<em>购物智能体</em>——一个能够在不同异构网站上自动填写商品搜索查询与订单的系统，通过本体匹配与映射来理解它们各异的 schema。Semantic Web 承诺，在一个结构化、机器可读的世界里，这类智能体将大放异彩。然而在实践中，手工本体的脆弱性、真实商品数据的混乱，以及缺乏稳健的自然语言理解，让这个愿景永远停留在“还有五年”。</p>

<p>在随后的岁月里，我亲历了 AI 进步的每一次浪潮：用于组合优化的神经网络与启发式搜索；深度学习与表征学习；大规模信息检索与个性化；以及最近的大语言模型革命。每一波浪潮都带来强大的新工具，但梦想始终未变：构建能在复杂环境中自主<em>理解</em>、<em>推理</em>与<em>行动</em>的系统。</p>

<p>2024–2026 年之所以非凡，是因为这些线索终于汇合在一起。LLM 提供语言理解与生成能力；强化学习教会它们推理并与人类意图对齐；工具使用协议（Model Context Protocol, MCP）赋予它们在世界中行动的“手”；智能体编排框架提供了二十年前 JADE 曾设想过的协作层——只是现在由基础模型驱动，而非手工编写的本体。在很多意义上，本指南正是我希望自己在那段旅程的每一步都能拥有的参考。</p>

## 2026 年的全景 {#the-landscape-in-2026}

<p>通向今天智能体 AI 系统的旅程，横跨三十年间在架构、训练与部署上不断累积的突破：</p>

<ol>
<li><p><strong>架构基础（2017–2020）</strong>：Transformer <a class="book-citation" id="cite-vaswani2017attention-1" href="../references.md#ref-vaswani2017attention" aria-label="参考文献 357">[357]</a> 把 self-attention 引入为一种通用的序列处理原语。Scaling Laws 揭示：用更多数据训练的更大模型会稳定地变好。GPT-2 与 GPT-3 证明，仅 decoder 结构的 Transformer 在足够规模化之后，会成为有能力的 few-shot learner。</p></li>
<li><p><strong>系统与效率（2020–2023）</strong>：FlashAttention <a class="book-citation" id="cite-dao2022flashattention-1" href="../references.md#ref-dao2022flashattention" aria-label="参考文献 65">[65]</a> 通过消除显存瓶颈让训练加速 2–4<span class="arithmatex">\(\times\)</span>。LoRA <a class="book-citation" id="cite-hu2022lora-1" href="../references.md#ref-hu2022lora" aria-label="参考文献 147">[147]</a> 让在单个节点上微调 70B+ 模型成为可能。专家混合（Mixture-of-Experts, MoE）将模型容量与算力成本解耦。vLLM 这类推理引擎把吞吐量带到了实时应用可触及的范围。</p></li>
<li><p><strong>基于 RL 的对齐（2022–2024）</strong>：RLHF <a class="book-citation" id="cite-ouyang2022training-1" href="../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a> 把“有能力但不实用”的 base 模型变成了真正有用的助手——这正是 ChatGPT 背后的配方。DPO <a class="book-citation" id="cite-rafailov2023direct-1" href="../references.md#ref-rafailov2023direct" aria-label="参考文献 302">[302]</a> 把 reward 模型与 RL 循环压缩成单一的监督 loss，降低了对齐的门槛。变体随之涌现：KTO <a class="book-citation" id="cite-ethayarajh2024kto-1" href="../references.md#ref-ethayarajh2024kto" aria-label="参考文献 87">[87]</a>、IPO <a class="book-citation" id="cite-azar2024general-1" href="../references.md#ref-azar2024general" aria-label="参考文献 14">[14]</a>、ORPO <a class="book-citation" id="cite-hong2024orpo-1" href="../references.md#ref-hong2024orpo" aria-label="参考文献 141">[141]</a>、GRPO <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a>。</p></li>
<li><p><strong>推理与自主（2024–2026）</strong>：DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 与 OpenAI 的 o1/o3 证明：RL 可以教会模型<em>推理本身</em>——模型自发地发现思维链（Chain-of-Thought，CoT）、回溯与自我验证。与此同时，Model Context Protocol (MCP) 统一了工具访问、智能体到智能体（Agent-to-Agent, A2A）使智能体间通信成为可能，生产级编排框架也走向成熟。</p></li>
</ol>

## 本指南的目标读者 {#who-this-guide-is-for}

<p>本文档是为<strong>真正动手做东西的从业者</strong>而写：</p>

<ul>
<li><p><strong>ML 工程师</strong>：需要理解 Transformer 内部机制、训练基础设施、优化方法，以及为何训练会发散。</p></li>
<li><p><strong>应用研究者</strong>：为其特定领域评估架构、微调策略与 RL 方法。</p></li>
<li><p><strong>Agent 开发者</strong>：构建生产系统，需要编排模式、记忆架构、工具集成（MCP）以及多智能体协作（A2A）。</p></li>
<li><p><strong>系统工程师</strong>：负责训练基础设施、GPU 集群、分布式训练与推理部署。</p></li>
<li><p><strong>技术负责人</strong>：需要在全栈层面做出架构与资源决策。</p></li>
</ul>

<p>我们假设读者熟悉神经网络与基础概率论。<strong>不要求事先具备 LLM、RL 或系统方面的知识</strong>——本指南从第一原理构建。</p>

## 你将获得什么 {#what-you-will-gain}

<p>读完本指南后，你将能够：</p>

<ul>
<li><p><strong>理解 LLM 内部机制</strong>——注意力机制、位置编码、MoE 路由、FlashAttention，以及架构选择为何会影响下游能力。</p></li>
<li><p><strong>对系统进行推理</strong>——GPU 显存预算、分布式训练策略（FSDP、张量/流水线并行）、推理优化，以及用 vLLM 进行生产部署。</p></li>
<li><p><strong>高效训练与微调</strong>——LoRA/QLoRA、量化、知识蒸馏、优化器选择以及学习率调度。</p></li>
<li><p><strong>将模型与人类偏好对齐</strong>——实现 RLHF/DPO/GRPO/KTO 流水线、调试 reward hacking 与 mode collapse，并在 20+ 种方法中选出合适的算法。</p></li>
<li><p><strong>构建推理模型</strong>——理解 DeepSeek-R1、o1/o3 与 QwQ 如何在没有显式示范的情况下，通过 RL 发现思维链。</p></li>
<li><p><strong>构建智能体系统的架构</strong>——选择编排模式、设计记忆、通过 MCP 集成工具、通过 A2A 协调智能体，并以生产级基准进行评估。</p></li>
<li><p><strong>严谨地做评估</strong>——对模型质量与智能体能力应用合适的指标、基准与 LLM-as-Judge 模式。</p></li>
</ul>

## 本指南的组织方式 {#how-this-guide-is-organized}

<p>本指南共 30 章，按六个部分组织：</p>

<ol>
<li><p><strong>第一部分——基础</strong>（第 1–3 章）：LLM 架构与优化（Transformer、attention、位置编码、FlashAttention、LoRA、MoE）、系统基础（GPU 层级、分布式训练、vLLM），以及经典 RL 理论（MDP、policy gradient、actor-critic）。</p></li>
<li><p><strong>第二部分——面向 LLM 的 RL 方法</strong>（第 4–12 章）：完整的“面向 LLM 的 RL”工具箱。先介绍语言模型的 RL 基础，再完整地数学化处理 PPO、DPO、GRPO 与偏好优化变体（Online DPO、KTO、IPO、ORPO、SimPO）、reward 模型训练、SFT 最佳实践、大规模系统架构，以及基于轨迹级 RL 的智能体训练。</p></li>
<li><p><strong>第三部分——推理</strong>（第 13 章）：大型推理模型——DeepSeek-R1、OpenAI o1/o3/o4-mini、QwQ——RL 如何发现思维链、MCTS、过程奖励模型（Process Reward Model, PRM）与测试时算力扩展（Test-Time Scaling）。</p></li>
<li><p><strong>第四部分——评估</strong>（第 14 章）：全面的 LLM 评估方法论——指标、LLM-as-Judge、人工标注、基准套件、污染检测，以及智能体评估。</p></li>
<li><p><strong>第五部分——智能体 AI</strong>（第 15–27 章）：完整的智能体技术栈——智能体 AI 简介、检索增强生成（Retrieval-Augmented Generation, RAG）与检索、记忆系统、编排与上下文管理、循环工程（Loop Engineering）、设计模式、智能体环境与基准、Model Context Protocol (MCP)、Agent Skills、Agent-to-Agent (A2A) 通信、多智能体系统、开发框架以及智能体 UI。</p></li>
<li><p><strong>第六部分——自测与参考</strong>（第 28–30 章）：覆盖全部主题的 108 道详细测验题与完整解答；一章快速参考，整合关键公式、API 参考与失败模式诊断；以及关于未来方向的总结。</p></li>
</ol>

<p>本指南包含 100+ 道详细测验题及覆盖所有主题的完整解答，并附带一章整合关键公式、API 参考与失败模式诊断的快速参考。</p>

## 设计哲学 {#design-philosophy}

<p>三条原则贯穿本文档：</p>

<ol>
<li><p><strong>先直觉、后形式化</strong>。每一个方程之前，都会先用通俗语言解释它的含义与重要性。</p></li>
<li><p><strong>面向实现</strong>。不知道如何让理论跑起来，理论就毫无用处。我们贯穿始终地提供代码、超参表、显存预算、架构图与调试策略。</p></li>
<li><p><strong>对“什么有效”诚实相告</strong>。我们会明确指出哪些方法经过生产验证，哪些仍处于研究探索阶段。</p></li>
</ol>

## 范围与有意省略 {#scope-and-deliberate-omissions}

<p>本指南聚焦于<strong>文本输入、文本输出的语言模型</strong>，以及围绕它们构建的 RL、系统与智能体基础设施。有几个重要领域被有意排除：</p>

<ul>
<li><p><strong>多模态模型</strong>（视觉–语言、音频、视频）。多模态架构引入了独特的训练流水线（对比预训练、跨模态对齐、模态专用编码器）、数据精选挑战与评估协议，每一项都值得用一本书的篇幅来讨论。将其纳入只会让范围翻倍，却无法加深贯穿本指南的 RL 与智能体核心。</p></li>
<li><p><strong>特定领域部署</strong>（医疗、法律、金融、科学发现）。领域适配带来监管约束、专门评估与数据访问问题，这些都与本文呈现的通用方法正交。我们覆盖的算法与架构<em>正是</em>从业者向这些领域适配时所使用的构建模块，但适配细节交由专门的参考文献会更合适。</p></li>
<li><p><strong>个性化与推荐系统</strong>。个性化依赖用户建模、协同过滤与交互历史架构，构成了一条平行的研究传统。尽管 LLM 越来越多地被<em>用在</em>推荐系统中，但其核心技术（序列模型、基于 bandit 的探索、冷启动处理）足够独特，值得另行讨论。</p></li>
</ul>

<p>通过守住这条边界，我们得以保持一条连贯的主线——<em>从架构基础与系统基础设施，到产出对齐与推理模型的训练算法，再到自主智能体的编排与部署</em>——而不至于因穿插不同模态与垂直行业而让叙事破碎。</p>

<p>— <em>Haggai Roitman，2026</em></p>

## 1.3 版新增内容 {#whats-new-v13}

<p>1.3 版新增了十九个主题，反映截至 2026 年中的最新进展：</p>

<dl class="book-definitions">
<dt>第 1 章——LLM 架构与优化：</dt><dd><ul>
<li><p><strong>Muon 优化器</strong>——用 Newton–Schulz 正交化作为 AdamW 的继任者（已被 GLM-5、Kimi K2、DeepSeek-V4 采用）。</p></li>
<li><p><strong>多头潜在注意力（Multi-head Latent Attention, MLA）</strong>——DeepSeek 通过低秩潜投影压缩 KV 缓存。</p></li>
<li><p><strong>FP8/FP4 训练</strong>——采用逐 tile 微缩放的下一代混合精度。</p></li>
<li><p><strong>无辅助损失 MoE（Auxiliary-Loss-Free MoE）</strong>——用基于自适应偏置的负载均衡取代昂贵的辅助损失。</p></li>
<li><p><strong>中期训练（Mid-training）</strong>——介于预训练与 SFT 之间、为模型进入 RL 做准备的新兴阶段。</p></li>
</ul></dd>
<dt>第 2 章——系统基础：</dt><dd><ul>
<li><p><strong>Dynamo</strong>——NVIDIA 的数据中心级推理编排框架。</p></li>
</ul></dd>
<dt>第 11 章——大规模系统架构：</dt><dd><ul>
<li><p><strong>解耦 DiLoCo（Decoupled DiLoCo）</strong>——Google 的地理分布式训练，带宽需求降低 236<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p><strong>Miles</strong>——PyTorch 原生的 RL 后训练引擎。</p></li>
</ul></dd>
<dt>第 12 章——LLM 智能体训练：</dt><dd><ul>
<li><p><strong>交互式 RL 环境（Interactive RL Environments）</strong>——用于智能体后训练的 NeMo Gym、RLFactory 与 MOSAIC。</p></li>
<li><p><strong>在真实 Agent 轨迹上训练</strong>——把生产环境中的轨迹用作 RL 训练信号。</p></li>
</ul></dd>
<dt>第 13 章——大型推理模型的 RL：</dt><dd><ul>
<li><p><strong>推理时算力扩展定律（Inference-Time Compute Scaling Laws）</strong>——对数线性关系与预算感知提示（budget-aware prompting）。</p></li>
<li><p><strong>潜空间推理（Latent Reasoning）</strong>——coconut 式的连续思维，无需 token 级思维链。</p></li>
<li><p><strong>同策略自蒸馏（On-Policy Self-Distillation）</strong>——从模型自身的特权上下文获取密集的 token 级监督，以低 10–100<span class="arithmatex">\(\times\)</span> 的算力达到 RL 级推理。</p></li>
</ul></dd>
<dt>第 17 章——智能体记忆系统：</dt><dd><ul>
<li><p><strong>主动记忆架构（Proactive Memory Architectures）</strong>——Meta AI 的行为状态衰减（behavioral-state decay）与预判式检索。</p></li>
</ul></dd>
<dt>第 19 章——循环工程：</dt><dd><ul>
<li><p><strong>上下文工程</strong>——优化“用什么填满上下文窗口”的学科（Lütke、Karpathy、Anthropic）。</p></li>
</ul></dd>
<dt>第 21 章——智能体环境：</dt><dd><ul>
<li><p><strong>UniClawBench 与 Terminal-Bench</strong>——面向机器人操作与长时程终端任务的 2026 年基准。</p></li>
</ul></dd>
<dt>第 25 章——多智能体系统：</dt><dd><ul>
<li><p><strong>BDI-LLM 自进化 Agent</strong>——带可学习计划库的信念—愿望—意图（Belief–Desire–Intention, BDI）架构。</p></li>
</ul></dd>
<dt>第 26 章——Agent 开发框架：</dt><dd><ul>
<li><p><strong>NVIDIA OO Agents (NOOA)</strong>——面向对象框架：Agent 即 Python 对象，方法即工具，<code>...</code> 方法体成为由 LLM 驱动的循环。</p></li>
</ul></dd>
</dl>
