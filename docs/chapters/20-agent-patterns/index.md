# 第 20 章 · Agent 设计模式 {#sec:agent-design-patterns}

<p>构建有效的 Agent 不仅需要强大的模型和一组工具。<em>架构</em>——即如何编排 LLM、如何分解任务、控制流如何在各组件间流动——决定了 Agent 是否可靠、可调试且具备成本效益。本章介绍从 Anthropic、OpenAI、Google 以及开源社区的生产部署中沉淀出的经典设计模式。</p>

<div class="admonition tip"><p class="admonition-title">何时使用 Agent 而非工作流</p>
<p>并非每个任务都需要一个自主 Agent。关键区别在于：</p>

<ul>
<li><p><strong>工作流（Workflow）</strong>：预定义的控制流，在特定步骤调用 LLM。可预测、可测试、成本更低。当任务结构已知时使用。</p></li>
<li><p><strong>Agent</strong>：LLM 动态决定下一步做什么。灵活、可应对新情况。当任务需要自适应决策时使用。</p></li>
</ul>

<p><strong>从工作流开始。</strong> 只有当任务确实需要动态路由或开放式探索时，才升级到 Agent。</p>
</div>

## 20.1　工作流模式 {#workflow-patterns}

<p>以下模式改编自 Anthropic 对 Agent 构建模块的分类 <a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a>，在<em>预定义</em>的控制流中使用 LLM。由系统（而非模型）决定执行顺序。</p>

### 20.1.1　Prompt 链（Prompt Chaining） {#prompt-chaining}

<p>最简单的模式：将复杂任务拆分为固定序列的 LLM 调用，将一个调用的结果作为上下文传入下一个调用。步骤之间的校验门可在错误向下游传播之前及早捕获它们。</p>

<figure id="figure-20-1"><a href="../../assets/figures/fig_060_fig60.png" target="_blank" rel="noopener" aria-label="查看图 20.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_060_fig60.png" alt="带质量门的 Prompt 链。每一步都是独立的 LLM 调用。质量门可以基于 LLM，也可以基于程序逻辑。" loading="lazy"></a><figcaption><span class="figure-number">图 20.1　</span>带质量门的 Prompt 链。每一步都是独立的 LLM 调用。质量门可以基于 LLM，也可以基于程序逻辑。</figcaption></figure>

<p><strong>适用场景</strong>：天然顺序化的任务——内容生成、数据转换、多阶段分析。</p>

<p><strong>关键优势</strong>：每一步可以使用不同的 Prompt、模型或温度。中间结果可检查、可调试。</p>

### 20.1.2　路由（Routing） {#routing}

<p>由分类器（LLM 或传统方法）检查输入，并分派到专门的处理器。</p>

<figure id="figure-20-2"><a href="../../assets/figures/fig_061_fig61.png" target="_blank" rel="noopener" aria-label="查看图 20.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_061_fig61.png" alt="路由模式：输入被分类一次，然后由专门的处理器处理。" loading="lazy"></a><figcaption><span class="figure-number">图 20.2　</span>路由模式：输入被分类一次，然后由专门的处理器处理。</figcaption></figure>

<p><strong>适用场景</strong>：不同任务类型对应不同的最佳 Prompt、工具或模型。如客服分诊、多模态输入处理。</p>

### 20.1.3　并行化（Parallelization） {#parallelization}

<p>多个 LLM 调用并发运行，由一个程序层合并它们的输出。可分为两个子模式：</p>

<ul>
<li><p><strong>分段（Sectioning，fan-out）</strong>：将输入划分为互不相交的块并独立处理——例如对一个代码库同时运行安全、性能和风格检查。</p></li>
<li><p><strong>投票（Voting，冗余）</strong>：用不同随机种子或温度对同一 Prompt 发起 <span class="arithmatex">\(N\)</span> 次调用，然后通过多数投票 <a class="book-citation" id="cite-wang2022selfconsistency-1" href="../../references.md#ref-wang2022selfconsistency" aria-label="参考文献 368">[368]</a>、奖励模型打分或 LLM-as-judge 选出最佳结果。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">并行化示例：代码评审</p>
<ol>
<li><p><strong>并行调用</strong>：安全评审 <span class="arithmatex">\(\|\)</span> 性能评审 <span class="arithmatex">\(\|\)</span> 风格评审</p></li>
<li><p><strong>聚合</strong>：合并所有发现，去重，按严重程度排序</p></li>
</ol>

<p>延迟为 <span class="arithmatex">\(\max\)</span>（各调用），而非 <span class="arithmatex">\(\sum\)</span>（各调用）。</p>
</div>

### 20.1.4　Orchestrator-Workers {#orchestrator-workers}

<p>在该模式下，由 LLM 自身决定如何拆分工作。一个 Orchestrator 模型分析任务、产出子任务计划、将每个子任务分派给 Worker LLM（可能使用不同的 Prompt 或工具），最终将它们的输出合并为一致的结果。与并行化的关键区别在于：分解逻辑是由模型生成的，而非硬编码的。</p>

<figure id="figure-20-3"><a href="../../assets/figures/fig_062_fig62.png" target="_blank" rel="noopener" aria-label="查看图 20.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_062_fig62.png" alt="Orchestrator-workers：LLM 决定如何分解任务，并综合各 Worker 的结果。" loading="lazy"></a><figcaption><span class="figure-number">图 20.3　</span>Orchestrator-workers：LLM 决定如何分解任务，并综合各 Worker 的结果。</figcaption></figure>

<p><strong>适用场景</strong>：开放式问题，子任务的数量和性质无法在设计时穷举——例如“重构这个代码库”需要先理解依赖图，再决定要修改哪些文件。</p>

### 20.1.5　Evaluator-Optimizer {#evaluator-optimizer}

<p>一个双模型反馈循环 <a class="book-citation" id="cite-madaan2023selfrefine-1" href="../../references.md#ref-madaan2023selfrefine" aria-label="参考文献 246">[246]</a>：生成器产出候选输出，由独立的评估器依据显式标准对其打分。若得分低于阈值，则将评估器的评论追加到生成器的上下文中，循环往复，直到达到质量标准或耗尽重试预算。</p>

<figure id="figure-20-4"><a href="../../assets/figures/fig_063_fig63.png" target="_blank" rel="noopener" aria-label="查看图 20.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_063_fig63.png" alt="Evaluator-optimizer：无需训练的迭代式精化。" loading="lazy"></a><figcaption><span class="figure-number">图 20.4　</span>Evaluator-optimizer：无需训练的迭代式精化。</figcaption></figure>

<p><strong>适用场景</strong>：具有明确质量标准的任务——必须通过测试的代码、必须保留语义的翻译、必须符合风格指南的写作。</p>

## 20.2　自主 Agent 模式 {#autonomous-agent-patterns}

<p>这些模式把执行流的控制权交给 LLM 本身。</p>

### 20.2.1　ReAct（Reason + Act） {#react-reason-act}

<p>最基础的 Agent 模式 <a class="book-citation" id="cite-yao2023react-1" href="../../references.md#ref-yao2023react" aria-label="参考文献 408">[408]</a>。LLM 在思考（内部推理）、行动（工具调用）和观察（处理结果）之间循环交替，直到产出最终答案。</p>

<div class="admonition note"><p class="admonition-title">ReAct 实现要点</p>
<ul>
<li><p><strong>草稿本（Scratchpad）</strong>：“思考”步骤被记录但不向用户展示。</p></li>
<li><p><strong>工具解析</strong>：脚手架（harness）从模型输出中抽取结构化的工具调用。</p></li>
<li><p><strong>最大迭代次数</strong>：始终对循环设上限（通常 10–25 次迭代）。</p></li>
<li><p><strong>终止条件</strong>：模型输出特殊动作（如 <code>final_answer</code>），或检测不到任何工具调用。</p></li>
</ul>
</div>

### 20.2.2　规划型 Agent（Planning Agents） {#planning-agents}

<p>Agent 在执行前生成显式计划，并可在执行过程中修订计划 <a class="book-citation" id="cite-wang2023planandsolve-1" href="../../references.md#ref-wang2023planandsolve" aria-label="参考文献 364">[364]</a>。</p>

<p class="table-caption" id="table-20-1"><span class="table-number">表 20.1　</span>规划策略对比</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>再规划时机</strong></th><th scope="col"><strong>特征</strong></th></tr></thead>
<tbody><tr><td>Plan-then-Execute</td><td>从不</td><td>简单；对意外结果脆弱</td></tr>
<tr><td>Adaptive</td><td>失败时</td><td>仅在某步失败时再规划；成本适中</td></tr>
<tr><td>Continuous</td><td>每一步</td><td>每次观察后完整重估；昂贵但鲁棒</td></tr>
<tr><td>Hierarchical</td><td>子计划完成时</td><td>高层计划固定；子计划动态生成</td></tr></tbody>
</table>
</div>

<div class="admonition example"><p class="admonition-title">规划型 Agent：研究报告生成</p>
<p><strong>用户请求</strong>：“写一份 2 页的报告，比较用于时间序列预测的 transformer 架构。”</p>

<p><strong>步骤 1 — 计划生成</strong>（单次 LLM 调用）：</p>

<div class="highlight"><pre><span></span><code><span class="n">plan</span> <span class="o">=</span> <span class="p">[</span>
    <span class="p">{</span><span class="s2">"id"</span><span class="p">:</span> <span class="mi">1</span><span class="p">,</span> <span class="s2">"task"</span><span class="p">:</span> <span class="s2">"Search for recent transformer-based "</span>
                      <span class="s2">"time-series models (2023-2025)"</span><span class="p">,</span>
     <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"search_web"</span><span class="p">,</span> <span class="s2">"deps"</span><span class="p">:</span> <span class="p">[]},</span>
    <span class="p">{</span><span class="s2">"id"</span><span class="p">:</span> <span class="mi">2</span><span class="p">,</span> <span class="s2">"task"</span><span class="p">:</span> <span class="s2">"Read top 5 papers, extract key methods"</span><span class="p">,</span>
     <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"read_papers"</span><span class="p">,</span> <span class="s2">"deps"</span><span class="p">:</span> <span class="p">[</span><span class="mi">1</span><span class="p">]},</span>
    <span class="p">{</span><span class="s2">"id"</span><span class="p">:</span> <span class="mi">3</span><span class="p">,</span> <span class="s2">"task"</span><span class="p">:</span> <span class="s2">"Build comparison table (architecture, "</span>
                      <span class="s2">"dataset, metrics)"</span><span class="p">,</span>
     <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"none"</span><span class="p">,</span> <span class="s2">"deps"</span><span class="p">:</span> <span class="p">[</span><span class="mi">2</span><span class="p">]},</span>
    <span class="p">{</span><span class="s2">"id"</span><span class="p">:</span> <span class="mi">4</span><span class="p">,</span> <span class="s2">"task"</span><span class="p">:</span> <span class="s2">"Write introduction + methodology section"</span><span class="p">,</span>
     <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"none"</span><span class="p">,</span> <span class="s2">"deps"</span><span class="p">:</span> <span class="p">[</span><span class="mi">2</span><span class="p">]},</span>
    <span class="p">{</span><span class="s2">"id"</span><span class="p">:</span> <span class="mi">5</span><span class="p">,</span> <span class="s2">"task"</span><span class="p">:</span> <span class="s2">"Write results + conclusion"</span><span class="p">,</span>
     <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"none"</span><span class="p">,</span> <span class="s2">"deps"</span><span class="p">:</span> <span class="p">[</span><span class="mi">3</span><span class="p">,</span> <span class="mi">4</span><span class="p">]},</span>
    <span class="p">{</span><span class="s2">"id"</span><span class="p">:</span> <span class="mi">6</span><span class="p">,</span> <span class="s2">"task"</span><span class="p">:</span> <span class="s2">"Review and polish final report"</span><span class="p">,</span>
     <span class="s2">"tool"</span><span class="p">:</span> <span class="s2">"none"</span><span class="p">,</span> <span class="s2">"deps"</span><span class="p">:</span> <span class="p">[</span><span class="mi">5</span><span class="p">]},</span>
<span class="p">]</span>
</code></pre></div>

<p><strong>步骤 2 — 带自适应再规划的执行</strong>：Agent 按依赖顺序执行各步骤。第 1 步后，搜索仅返回 3 篇相关论文。Agent 进行<em>再规划</em>：新增一个子步骤，将搜索范围扩展到相邻领域（如 PatchTST、iTransformer）。修订后的计划基于扩充后的语料从第 2 步继续。</p>

<p><strong>关键洞见</strong>：计划是一份<em>活文档</em>——它提供结构，但会随观察而调整。harness 将依赖关系跟踪为 DAG，仅执行其前驱已完成的步骤。</p>
</div>

### 20.2.3　反思与自我批判（Reflection and Self-Critique） {#reflection-and-self-critique}

<p>Agent 暂停以评估自身轨迹并纠正方向：</p>

<ol>
<li><p><strong>输出校验</strong>：“这正确吗？我有遗漏吗？”</p></li>
<li><p><strong>轨迹回顾</strong>：回顾最近 <span class="arithmatex">\(k\)</span> 步，识别错误或低效之处。</p></li>
<li><p><strong>策略修订</strong>：重新考虑整体方法（“我在解决正确的问题吗？”）。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">Reflexion：从失败中学习</p>
<p><strong>Reflexion</strong> 模式 <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 维护一个持久化的“反思记忆”。每次失败后，Agent 写下一段自然语言反思（“我失败是因为没有检查边界情况”）。在下一次尝试中，这些反思被纳入 Prompt——从而实现跨 Episode 的学习而无需更新权重。</p>
</div>

### 20.2.4　工具调用模式（Tool-Use Patterns） {#tool-use-patterns}

<p>Agent 调用工具的方式显著影响其可靠性、延迟和成本。已涌现出五种经典模式 <a class="book-citation" id="cite-schick2023toolformer-1" href="../../references.md#ref-schick2023toolformer" aria-label="参考文献 315">[315]</a>：</p>

<p class="table-caption" id="table-20-2"><span class="table-number">表 20.2　</span>工具调用模式</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>描述</strong></th><th scope="col"><strong>示例</strong></th></tr></thead>
<tbody><tr><td>单轮（Single-turn）</td><td>每次 LLM 回复一次工具调用</td><td>带搜索的简单问答</td></tr>
<tr><td>多工具（Multi-tool）</td><td>单次回复中并行多个工具调用</td><td>搜索 + 计算 + 格式化</td></tr>
<tr><td>顺序（Sequential）</td><td>工具输出馈入下一次工具调用</td><td>搜索 <span class="arithmatex">\(\to\)</span> 阅读 <span class="arithmatex">\(\to\)</span> 抽取</td></tr>
<tr><td>嵌套（Nested）</td><td>工具调用触发另一个 Agent</td><td>代码 Agent 调用测试运行器</td></tr>
<tr><td>回退（Fallback）</td><td>首选工具失败时尝试备选方案</td><td>API <span class="arithmatex">\(\to\)</span> 爬取 <span class="arithmatex">\(\to\)</span> 缓存</td></tr></tbody>
</table>
</div>

##### 单轮工具使用。 {#single-turn-tool-use.}

<p>最简模式：模型发出一次工具调用、接收结果、产出最终答案。足以应对事实查询、单位换算或单次 API 查询。harness 恰好进行两次 LLM 调用（一次决定使用哪个工具，一次综合结果）。</p>

##### 多工具（并行）。 {#multi-tool-parallel.}

<p>现代 API（OpenAI、Anthropic）允许模型在单次回复中请求多个工具调用。harness 并发执行它们并一起返回所有结果。对于需要从多个来源获取独立信息的任务（如同时获取股价、天气和日历），这能极大降低延迟。关键约束：这些工具必须<em>相互独立</em>（任何工具的输出都不作为另一工具的输入）。</p>

##### 顺序（流水线）。 {#sequential-pipeline.}

<p>每个工具的输出馈入下一个工具的输入，形成数据流水线。模型基于前一步结果决定下一个工具。在研究工作流中很常见：<code>search</code> <span class="arithmatex">\(\to\)</span> <code>fetch_page</code> <span class="arithmatex">\(\to\)</span> <code>extract_data</code> <span class="arithmatex">\(\to\)</span> <code>analyze</code>。harness 必须跟踪不断增长的上下文，并可能需要对中间结果进行摘要以保持在预算之内。</p>

##### 嵌套（Agent-as-Tool）。 {#nested-agent-as-tool.}

<p>一次工具调用会唤起一个完全独立的 Agent——拥有自己的 Prompt、工具和上下文。父 Agent 将子 Agent 视为黑盒函数。这实现了专业化：研究 Agent 将代码执行委派给编码 Agent，后者能访问沙箱和测试运行器。Swarm 模式 <a class="book-citation" id="cite-openai2024swarm-1" href="../../references.md#ref-openai2024swarm" aria-label="参考文献 276">[276]</a> 通过专业化 Agent 之间的交接（handoff）泛化了这一思路。</p>

##### 回退（优雅降级）。 {#fallback-graceful-degradation.}

<p>harness 按优先级顺序尝试工具：若首选工具失败（超时、限流、API 错误），则自动回退到备选工具。模型无需感知回退逻辑——harness 透明处理。示例：主搜索 API <span class="arithmatex">\(\to\)</span> 备用搜索 <span class="arithmatex">\(\to\)</span> 缓存结果 <span class="arithmatex">\(\to\)</span> 告知模型搜索不可用。</p>

## 20.3　设计原则 {#design-principles}

<p>以下原则提炼自 Anthropic 的《构建有效 Agent》指南 <a class="book-citation" id="cite-anthropic2024buildingagents-1--section-03" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a>，适用于所有模式：</p>

<ol>
<li><p><strong>保持简单。</strong> 使用能跑通的最简架构。仅在确有必要时增加复杂度。一个能解决问题的 Prompt 链，永远胜过一个“也许能解决”的多 Agent 系统。</p></li>
<li><p><strong>透明优于聪明。</strong> 每一步都应可检查。避免隐藏状态或隐式推理。当 Agent 失败时，你需要理解<em>原因</em>——不透明的架构使调试无从下手。</p></li>
<li><p><strong>提供良好工具。</strong> 文档完善、类型清晰、错误信息明确的工具是效能放大器。描述含糊的工具会被误用；具备精确 Schema 和使用指南的工具会被正确选择。</p></li>
<li><p><strong>为失败做规划。</strong> 每次工具调用都可能失败。在 harness 层构建重试逻辑、回退方案和优雅降级，使模型无需推理基础设施故障。</p></li>
<li><p><strong>使用结构化输出。</strong> 受约束的生成（JSON Schema、函数调用）防止解析失败。一个产出需要正则解析的自由文本的 Agent 是脆弱的；一个产出经校验 JSON 的 Agent 是鲁棒的。</p></li>
<li><p><strong>使用多样化输入测试。</strong> Agent 行为比单轮对话更易变。同一 Prompt 在不同运行中可能产生不同的工具调用序列。进行对抗性测试，覆盖边界情况、模糊请求和畸形输入。</p></li>
</ol>

## 20.4　模式选择指南 {#pattern-selection-guide}

<p>选择正确的模式取决于三个因素：(1) 任务结构的可预测性如何，(2) 延迟和成本上能承担多少次 LLM 调用，(3) 质量是否需要迭代。可将下表作为决策矩阵——从顶部（最简模式）开始，仅当较简单的模式确实失败时才向下移动。</p>

<p class="table-caption" id="table-20-3"><span class="table-number">表 20.3　</span>各 Agent 设计模式的适用场景</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>复杂度</strong></th><th scope="col"><strong>LLM 调用次数</strong></th><th scope="col"><strong>最适用于</strong></th></tr></thead>
<tbody><tr><td>Prompt 链</td><td>低</td><td><span class="arithmatex">\(N\)</span>（固定）</td><td>顺序任务、内容流水线</td></tr>
<tr><td>路由</td><td>低</td><td>1 + 1</td><td>多类型输入、分诊</td></tr>
<tr><td>并行化</td><td>低</td><td><span class="arithmatex">\(N\)</span>（并行）</td><td>独立子任务、投票</td></tr>
<tr><td>Orchestrator-workers</td><td>中</td><td>可变</td><td>未知的分解结构</td></tr>
<tr><td>Evaluator-optimizer</td><td>中</td><td>2–10（循环）</td><td>对质量要求高的输出</td></tr>
<tr><td>ReAct</td><td>中</td><td>3–25（循环）</td><td>通用工具使用、探索</td></tr>
<tr><td>规划型 Agent</td><td>高</td><td>5–50+</td><td>长时程、多步任务</td></tr>
<tr><td>反思</td><td>高</td><td>+50% 开销</td><td>首次尝试常失败的任务</td></tr>
<tr><td>多 Agent</td><td>高</td><td>多次</td><td>复杂领域、专业化</td></tr></tbody>
</table>
</div>

<p>这些模式是可组合的：一个规划型 Agent 可以在各步骤内使用 Prompt 链，在评审阶段使用 Evaluator-optimizer，并使用路由将子任务分派给专家。艺术在于知道何时停止增加层次。</p>
