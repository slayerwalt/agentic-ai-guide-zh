# 19.1　上下文工程：循环之下的那一层 {#sec:context-engineering}

<p>在循环运行之前，必须有人决定模型在每一步看到什么。这门学科就是<strong>上下文工程</strong>：在推理期间策展并维护上下文窗口中最优 token 集合的正式实践。它是循环之下的那一层——循环状态被翻译为模型输入的机制。</p>

<p>该术语由 Tobi Lütke（Shopify CEO）在 2025 年 6 月推广开来 <a class="book-citation" id="cite-lutke2025context-1" href="../../references.md#ref-lutke2025context" aria-label="参考文献 244">[244]</a>，他将其描述为与 AI Agent 协作的新兴核心技能。Anthropic 在 2025 年 9 月将其形式化为“在推理期间策展并维护最优 token 集合”。Andrej Karpathy 认同这一框架，将其描述为“在下一步之前，用恰到好处的信息填满上下文窗口的精妙艺术与科学” <a class="book-citation" id="cite-karpathy2025context-1" href="../../references.md#ref-karpathy2025context" aria-label="参考文献 178">[178]</a>。这些定义的汇聚反映了一个现实：当上下文窗口从 4K 增长到 128K 再到 1M token 时，“<em>往里面放什么</em>”变得与“<em>问什么</em>”同等重要。</p>

<p>上下文工程不是提示工程。提示工程优化的是<em>指令</em>——引导模型行为的措辞。上下文工程优化的是<em>信息环境</em>——模型据以推理的文档、工具输出、历史与状态的全集。在单轮交互中，两者的区别很小。在多步 Agent 循环中，它决定了模型是拥有正确行动所需的信息，还是在盲目飞行。</p>

#### 关键技术。 {#heading-19-3}

<p>上下文工程借助几种互补的方法：</p>

<ul>
<li><p><strong>动态上下文组装</strong>：在每一轮循环迭代中从多个来源（检索到的文档、工具输出、记忆存储）选择并组合上下文，而不是使用固定模板。</p></li>
<li><p><strong>RAG 集成</strong>：检索任务相关文档或代码片段并注入上下文窗口，用精确、即时的信息取代过时或泛泛的背景知识。</p></li>
<li><p><strong>工具输出摘要化</strong>：在把冗长的工具输出（如长文件列表、测试运行器日志）插入上下文之前先压缩，在保留信号的同时节省 token 预算。</p></li>
<li><p><strong>对话历史管理</strong>：决定哪些先前的轮次原样保留、哪些摘要化、哪些彻底丢弃——一个滚动压缩问题，在长时程 Agent 运行中变得至关重要。</p></li>
<li><p><strong>Token 预算分配</strong>：在相互竞争的信息来源之间显式划分上下文窗口，确保模型始终有空间生成有用的回答。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">上下文窗口预算分配（典型 Agent 循环）</p>
<span id="table-19-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>组件</strong></th><th scope="col"><strong>预算</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>系统提示</td><td>10–15%</td><td>稳定；定义 Agent 角色、工具与约束</td></tr>
<tr><td>检索文档 / RAG</td><td>20–30%</td><td>动态；每轮迭代刷新</td></tr>
<tr><td>对话 / 动作历史</td><td>30–40%</td><td>随运行变长而压缩</td></tr>
<tr><td>生成预算</td><td>20–30%</td><td>为模型输出预留；绝不可压缩低于此线</td></tr></tbody>
</table>
</div>

<p>这些是指导原则，而非硬性规则。长时程编码 Agent 可以给历史更多预算；单步问答 Agent 可以给检索文档更多预算。关键不变量：始终<em>先</em>预留生成预算，再填充其余部分。</p>
</div>

#### 循环内的上下文工程。 {#heading-19-4}

<p>在循环中，上下文工程在每一轮迭代都发挥作用。在每一步 <span class="arithmatex">\(t\)</span>，循环控制器必须决定：保留哪些先前的观测 <span class="arithmatex">\(o_{&lt;t}\)</span>、摘要哪些、哪些检索文档仍然相关，以及如何从这些组件组合出新的上下文 <span class="arithmatex">\(s_t\)</span>。这不是一次性的设计决策——而是一条与 Agent 策略 <span class="arithmatex">\(\pi_\theta\)</span> 并行的动态策略。设计糟糕的上下文策略会让模型丢失目标、重复已做过的动作，或无法使用两步之前检索到的信息。设计良好的策略则能让模型的“工作记忆”在整个运行期间始终与任务当前状态对齐。</p>

<div class="admonition tip"><p class="admonition-title">上下文工程 vs. 循环工程</p>
<p>上下文工程回答：<em>模型在第 <span class="arithmatex">\(t\)</span> 步看到了什么？</em>循环工程回答：<em>模型在第 <span class="arithmatex">\(t\)</span> 步行动之后发生了什么？</em>两者互补。循环决定动作与观测的序列；上下文工程决定当模型采取下一个动作时这些观测如何被表示。做不好上下文工程就不可能做好循环工程——循环的状态只有被正确编码进模型上下文时才有用。</p>
</div>
