# 20.2　自主 Agent 模式 {#autonomous-agent-patterns}

<p>这些模式把执行流的控制权交给 LLM 本身。</p>

## 20.2.1　ReAct（Reason + Act） {#react-reason-act}

<p>最基础的 Agent 模式 <a class="book-citation" id="cite-yao2023react-1" href="../../references.md#ref-yao2023react" aria-label="参考文献 408">[408]</a>。LLM 在思考（内部推理）、行动（工具调用）和观察（处理结果）之间循环交替，直到产出最终答案。</p>

<div class="admonition note"><p class="admonition-title">ReAct 实现要点</p>
<ul>
<li><p><strong>草稿本（Scratchpad）</strong>：“思考”步骤被记录但不向用户展示。</p></li>
<li><p><strong>工具解析</strong>：脚手架（harness）从模型输出中抽取结构化的工具调用。</p></li>
<li><p><strong>最大迭代次数</strong>：始终对循环设上限（通常 10–25 次迭代）。</p></li>
<li><p><strong>终止条件</strong>：模型输出特殊动作（如 <code>final_answer</code>），或检测不到任何工具调用。</p></li>
</ul>
</div>

## 20.2.2　规划型 Agent（Planning Agents） {#planning-agents}

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

## 20.2.3　反思与自我批判（Reflection and Self-Critique） {#reflection-and-self-critique}

<p>Agent 暂停以评估自身轨迹并纠正方向：</p>

<ol>
<li><p><strong>输出校验</strong>：“这正确吗？我有遗漏吗？”</p></li>
<li><p><strong>轨迹回顾</strong>：回顾最近 <span class="arithmatex">\(k\)</span> 步，识别错误或低效之处。</p></li>
<li><p><strong>策略修订</strong>：重新考虑整体方法（“我在解决正确的问题吗？”）。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">Reflexion：从失败中学习</p>
<p><strong>Reflexion</strong> 模式 <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 维护一个持久化的“反思记忆”。每次失败后，Agent 写下一段自然语言反思（“我失败是因为没有检查边界情况”）。在下一次尝试中，这些反思被纳入 Prompt——从而实现跨 Episode 的学习而无需更新权重。</p>
</div>

## 20.2.4　工具调用模式（Tool-Use Patterns） {#tool-use-patterns}

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

#### 单轮工具使用。 {#single-turn-tool-use.}

<p>最简模式：模型发出一次工具调用、接收结果、产出最终答案。足以应对事实查询、单位换算或单次 API 查询。harness 恰好进行两次 LLM 调用（一次决定使用哪个工具，一次综合结果）。</p>

#### 多工具（并行）。 {#multi-tool-parallel.}

<p>现代 API（OpenAI、Anthropic）允许模型在单次回复中请求多个工具调用。harness 并发执行它们并一起返回所有结果。对于需要从多个来源获取独立信息的任务（如同时获取股价、天气和日历），这能极大降低延迟。关键约束：这些工具必须<em>相互独立</em>（任何工具的输出都不作为另一工具的输入）。</p>

#### 顺序（流水线）。 {#sequential-pipeline.}

<p>每个工具的输出馈入下一个工具的输入，形成数据流水线。模型基于前一步结果决定下一个工具。在研究工作流中很常见：<code>search</code> <span class="arithmatex">\(\to\)</span> <code>fetch_page</code> <span class="arithmatex">\(\to\)</span> <code>extract_data</code> <span class="arithmatex">\(\to\)</span> <code>analyze</code>。harness 必须跟踪不断增长的上下文，并可能需要对中间结果进行摘要以保持在预算之内。</p>

#### 嵌套（Agent-as-Tool）。 {#nested-agent-as-tool.}

<p>一次工具调用会唤起一个完全独立的 Agent——拥有自己的 Prompt、工具和上下文。父 Agent 将子 Agent 视为黑盒函数。这实现了专业化：研究 Agent 将代码执行委派给编码 Agent，后者能访问沙箱和测试运行器。Swarm 模式 <a class="book-citation" id="cite-openai2024swarm-1" href="../../references.md#ref-openai2024swarm" aria-label="参考文献 276">[276]</a> 通过专业化 Agent 之间的交接（handoff）泛化了这一思路。</p>

#### 回退（优雅降级）。 {#fallback-graceful-degradation.}

<p>harness 按优先级顺序尝试工具：若首选工具失败（超时、限流、API 错误），则自动回退到备选工具。模型无需感知回退逻辑——harness 透明处理。示例：主搜索 API <span class="arithmatex">\(\to\)</span> 备用搜索 <span class="arithmatex">\(\to\)</span> 缓存结果 <span class="arithmatex">\(\to\)</span> 告知模型搜索不可用。</p>
