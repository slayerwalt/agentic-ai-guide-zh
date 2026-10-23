# 17.11　智能体记忆的最新进展 {#sec:memory-recent}

<p>上文描述的记忆系统奠定了基础模式。近期若干工作进一步推进了边界：</p>

## 17.11.1　CoALA：面向语言 Agent 的认知架构 {#coala-cognitive-architectures-for-language-agents}

<p>Sumers 等人 <a class="book-citation" id="cite-sumers2023coala-1" href="../../references.md#ref-sumers2023coala" aria-label="参考文献 344">[344]</a> 提出了 <em>面向语言 Agent 的认知架构</em>（Cognitive Architectures for Language Agents, CoALA），这是一个借用认知科学与符号 AI 原则、用于组织日益庞杂的 LLM Agent 生态的统一框架。CoALA 将一个语言 Agent 拆解为：</p>

<ul>
<li><p><strong>模块化记忆</strong>：工作记忆（上下文窗口）、情景记忆（过往经验）、语义记忆（世界知识）以及程序记忆（动作 schema）——与 <a href="section-02.md#sec:memory-taxonomy">17.2</a> 节的分类体系相对应。</p></li>
<li><p><strong>结构化的动作空间</strong>：内部动作（推理、检索、记忆写入）与外部动作（工具调用 Tool Calling、与环境交互）。</p></li>
<li><p><strong>决策循环</strong>：一个广义的“感知—规划—行动”循环，其中带有显式的检索与写入步骤。</p></li>
</ul>

<p>CoALA 的贡献与其说是一个新系统，不如说是一种<em>设计语言</em>：它提供了一种系统化的方式来分析已有 Agent 并识别缺失能力，对从业者而言是一个有用的参考架构。</p>

## 17.11.2　Mem0：生产级记忆层 {#mem0-production-scale-memory-layer}

<p>Mem0 <a class="book-citation" id="cite-chhikara2025mem0-1" href="../../references.md#ref-chhikara2025mem0" aria-label="参考文献 53">[53]</a> 致力于弥合研究型记忆系统与生产部署之间的鸿沟。核心思路：</p>

<ul>
<li><p><strong>自动抽取</strong>：Mem0 并不依赖 LLM 显式发出记忆写入命令，而是自动从对话轮次中抽取显著事实，并将其巩固到持久化存储中。</p></li>
<li><p><strong>基于图的记忆</strong>：除了扁平的向量存储之外，Mem0 还在抽取出的实体与事实之上维护一张<em>关系图</em>，以支持多跳记忆查询（“用户在项目 Y 的语境下，对话题 X 说过什么？”）。</p></li>
<li><p><strong>记忆压缩</strong>：冗余或被取代的事实会自动合并，使记忆存储保持紧凑与最新。</p></li>
</ul>

<p>在 LOCOMO 基准上，Mem0 相对 OpenAI 的基线记忆取得了 26% 的相对提升，与全上下文方案相比，p95 延迟降低 91%，Token 成本降低 <span class="arithmatex">\(&gt;\)</span>90%。</p>

## 17.11.3　睡眠时计算（Sleep-Time Compute）：离线记忆处理 {#sleep-time-compute-offline-memory-processing}

<p>Lin 等人 <a class="book-citation" id="cite-lin2025sleeptime-1" href="../../references.md#ref-lin2025sleeptime" aria-label="参考文献 218">[218]</a> 提出了 <em>sleep-time compute</em>（睡眠时计算）范式，让 Agent 在用户交互的<em>间隔之间</em>处理并巩固记忆，而不是只在查询时才计算。其类比是生物的睡眠——大脑在睡眠中巩固记忆并预先建立有用的联想。</p>

#### 工作机制。 {#how-it-works.}

<p>在空闲时段（“睡眠”）中，Agent 会：</p>

<ol>
<li><p>基于当前上下文预测未来可能出现的查询。</p></li>
<li><p>预先计算推理链、摘要与结构化表示。</p></li>
<li><p>把这些预计算产物存储下来，以便测试时（test-time）推理可检索并复用。</p></li>
</ol>

#### 效果。 {#results.-dup4}

<p>在推理类基准上，sleep-time compute 把达成等价准确率所需的测试时计算降低约 <span class="arithmatex">\(5\times\)</span>。如果在同一上下文的多个相关查询上进行摊销，平均每次查询的成本下降 <span class="arithmatex">\(2.5\times\)</span>。当用户查询是<em>可预测</em>的——即上下文强约束了可能被问到的问题时——该方法最为有效。</p>

<div class="admonition tip"><p class="admonition-title">把记忆巩固视作离线 RL</p>
<p>Sleep-time compute 可以看作<em>离线 Policy 改进</em>：在空闲时间，Agent 利用已经收集到的数据（过往交互）改进其记忆表示（Policy），无需与环境进行新的交互。这与离线 RL 方法（见第 <a href="../08-preference-optimization/index.md#ch:po-variants">8</a> 章）相通——后者从一个静态的轨迹数据集学习。</p>
</div>

## 17.11.4　A-MEM：受 Zettelkasten 启发的智能体记忆 {#a-mem-zettelkasten-inspired-agentic-memory}

<p>A-MEM <a class="book-citation" id="cite-xu2025amem-1" href="../../references.md#ref-xu2025amem" aria-label="参考文献 400">[400]</a> 引入了一种借鉴 <em>Zettelkasten</em> 方法的记忆系统——该方法是一种基于密集互联原子笔记的笔记系统——为 LLM Agent 实现动态、自组织的记忆。</p>

#### 关键设计原则。 {#key-design-principles.}

<ul>
<li><p><strong>结构化笔记。</strong> 每条记忆条目都不是原始文本块，而是一条带有多种结构化属性的<em>笔记</em>：上下文描述、关键词、标签，以及指向相关笔记的显式链接。这些元数据使检索比单纯的 Embedding 相似度更丰富。</p></li>
<li><p><strong>动态链接。</strong> 当一条新记忆被加入时，系统会分析已有记忆，识别语义上有意义的关联并建立双向链接。结果是一张<em>知识网络</em>，而不是扁平的列表。</p></li>
<li><p><strong>记忆演化。</strong> 关键之处在于，新增笔记可以<em>触发对已有笔记的更新</em>——随着 Agent 理解的加深，对它们的上下文表示与属性进行细化。这让记忆成为一个随时间不断演进的活体结构，而非静态归档。</p></li>
<li><p><strong>Agent 驱动的组织。</strong> 与固定 schema 的记忆系统不同，A-MEM 让 LLM 自身决定如何组织、链接与更新记忆——使组织结构能够自适应任务领域。</p></li>
</ul>

#### 效果。 {#results.-1-dup3}

<p>在六个基础模型、多种多会话推理任务上，A-MEM 稳定优于扁平向量存储、基于摘要的记忆与图数据库方案，说明记忆<em>如何</em>组织与<em>存什么</em>同等重要。</p>
