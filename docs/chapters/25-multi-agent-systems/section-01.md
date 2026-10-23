# 25.1　动机：为什么需要多个 Agent？ {#subsec:mas-motivation}

<p>人工智能的历史在很多方面就是一部规模演进史。早期 AI 系统是单体式的：单一程序、单一知识库、单一推理引擎。随着问题日益复杂，研究者发现，没有任何单一 Agent——无论能力多强——能够高效地处理一个丰富、开放式任务的方方面面。这一洞见在分布式 AI 与多智能体系统（Multi-Agent System, MAS）研究中早有定论 <a class="book-citation" id="cite-weiss1999multiagent-1" href="../../references.md#ref-weiss1999multiagent" aria-label="参考文献 375">[375]</a><a class="book-citation" id="cite-wooldridge2009introduction-1" href="../../references.md#ref-wooldridge2009introduction" aria-label="参考文献 381">[381]</a>，而在大语言模型时代又被赋予了新的紧迫性。</p>

<div class="admonition tip"><p class="admonition-title">核心直觉</p>
<p>单个 LLM 无论多大，本质上都是一个通才。一支由专门化 LLM 组成的团队，每个 LLM 聚焦于一个狭窄的子问题并交流结果，在复杂的多维度任务上能够胜过通才——正如一支人类专家团队在复杂工程项目上胜过单个通才。</p>
</div>

<p>从单体 Agent 转向<em>Agent 社会</em>（agent societies）的根本动机有四个：</p>

#### 专门化（Specialization）。 {#specialization.}

<p>不同的子任务受益于不同的能力、Prompt 策略，乃至不同的基础模型。代码生成 Agent 可以在编程语料上微调；事实核查 Agent 可以借助检索工具锚定事实；创意写作 Agent 可以通过 Prompt 激发风格多样性。强迫单个 Agent 在所有这些方面同时出色，既低效，往往也不可能。</p>

#### 并行化（Parallelism）。 {#parallelism.}

<p>许多现实世界的任务可以分解为可并发执行的独立子任务。一个需要文献综述、数据分析和报告撰写的研究流水线可以让这三者并行进行，从而显著降低墙钟时间。串行的单 Agent 处理是一个瓶颈，而多 Agent 并行化正好消除了它。</p>

#### 鲁棒性（Robustness）。 {#robustness.}

<p>单个 Agent 是单点故障。如果它产生幻觉、陷入循环或给出细微错误的答案，将无人核验。多 Agent 系统引入了冗余：第二个 Agent 可以验证、批判，或独立地重新推导结果。在输出被信任之前，对抗性 Agent 可以主动探测其弱点。</p>

#### 涌现能力（Emergent Capabilities）。 {#emergent-capabilities.}

<p>也许最引人入胜的是，Agent 群体可以展现出任何单一 Agent 都不具备的能力。通过辩论、协商和迭代精化，多 Agent 系统能够得到超越任一单独 Agent 所能产出的解决方案——这是一种社会性生物涌现智能的计算类比。</p>

<div class="admonition note"><p class="admonition-title">历史脉络</p>
<p>多智能体系统研究可以追溯到 1980 年代，其奠基性工作包括分布式问题求解 <a class="book-citation" id="cite-durfee1989distributed-1" href="../../references.md#ref-durfee1989distributed" aria-label="参考文献 80">[80]</a>、合同网协议（Contract Net Protocol） <a class="book-citation" id="cite-smith1980contract-1" href="../../references.md#ref-smith1980contract" aria-label="参考文献 334">[334]</a> 以及 FIPA Agent 通信标准 <a class="book-citation" id="cite-fipa2002acl-1" href="../../references.md#ref-fipa2002acl" aria-label="参考文献 93">[93]</a>。向基于 LLM 的 Agent 转变让这些经典思想在新的载体上重获新生：以前是手工编码、做符号推理的 Agent，现在则是其“认知”从学习到的神经表示中涌现出来的 Agent。核心的架构模式——层级、市场、黑板、消息传递——依然惊人地相关。</p>
</div>

<p>从单体 Agent 到 Agent 社会的转变映射了复杂系统中一个更广泛的模式：随着问题空间的增长，分布式、模块化架构始终优于集中式、单体式架构。问题不再是<em>是否</em>使用多个 Agent，而是<em>如何</em>组织它们。</p>
