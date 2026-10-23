# 23.3　案例研究：Anthropic 的 Agent 设计 {#case-study-anthropics-agent-design}

<p>Anthropic 对 Agent 架构的方法 <a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a> 是关于生产环境中基于 skill 的 Agent 设计最清晰的阐述之一。其设计哲学强调<strong>简单优于复杂</strong>、<strong>可组合的构建块优于单体框架</strong>。（这些模式也会在第 <a href="../20-agent-patterns/index.md#sec:agent-design-patterns">20</a> 章从编排视角进行讨论。）</p>

## 23.3.1　核心原则 {#core-principles}

<ol>
<li><p><strong>从最简单的方案起步。</strong>在尝试更简单的方案（单次 LLM 调用、检索 + 生成）并确认其不足之前，不要急于使用 Agent 化模式。</p></li>
<li><p><strong>Workflow 与 Agent 的区分。</strong>Anthropic 将以下两者加以区分：</p>

<ul>
<li><p><strong>Workflows</strong>：对 LLM 调用进行预先编排——确定性的控制流，在特定节点包含 LLM 步骤。更可预测、更易调试。</p></li>
<li><p><strong>Agents</strong>：由 LLM 动态决定下一步该做什么——工具选择、迭代次数与停止条件都由模型驱动。更灵活，但更难控制。</p></li>
</ul></li>
<li><p><strong>以增强型 LLM 作为原子单元。</strong>原语从来不是裸模型——它总是与其检索源、可调用工具和持久化记忆捆绑在一起的模型。这一组合单元在实践中就是“配备了 skill 的模型”。</p></li>
</ol>

## 23.3.2　构建块模式 {#building-block-patterns}

<p>Anthropic 总结了五种可组合的 workflow 模式，可作为 skill 模板使用：</p>

<p class="table-caption" id="table-23-1"><span class="table-number">表 23.1　</span>Anthropic 提出的可组合 Agent 模式。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>机制</strong></th><th scope="col"><strong>适用场景</strong></th></tr></thead>
<tbody><tr><td><strong>Prompt Chaining</strong></td><td>顺序的 LLM 调用，每一步的输出作为下一步的输入。步骤之间设有关卡，校验中间结果。</td><td>具有清晰分解的多步骤转换任务</td></tr>
<tr><td><strong>Routing</strong></td><td>一个分类器或 LLM 根据任务类型将输入分发给专门的处理者（skill）。</td><td>需要不同领域专长的多个独立任务类别</td></tr>
<tr><td><strong>Parallelization</strong></td><td>多个 LLM 调用同时进行——可以是分段（拆分任务），也可以是投票（同一任务、汇总结果）。</td><td>相互独立的子任务；或通过共识获得置信度</td></tr>
<tr><td><strong>Orchestrator–Workers</strong></td><td>一个中心 LLM 将任务拆分为子任务，分派给 worker LLM，再综合结果。</td><td>子任务难以事先预测的复杂任务</td></tr>
<tr><td><strong>Evaluator–Optimizer</strong></td><td>一个 LLM 生成，另一个 LLM 评估；反复迭代直至达到质量阈值。</td><td>具有明确质量标准的任务（代码、写作）</td></tr></tbody>
</table>
</div>

## 23.3.3　增强型 LLM {#the-augmented-llm}

<p>在 Anthropic 的框架中，基本单元不是裸模型，而是<strong>增强型 LLM</strong>：</p>



<div class="arithmatex">\[\text{Augmented LLM} = \text{Model} + \text{Retrieval} + \text{Tools} + \text{Memory}\]</div>

<p>这与 skill 概念直接对应：每个 skill 都为一个特定任务配置该模型可访问的检索源、tool 与记忆存储。Skill 边界定义了模型在某次特定调用中<em>能看到什么、能做什么</em>。</p>

## 23.3.4　实际影响 {#practical-implications-dup2}

<div class="admonition tip"><p class="admonition-title">Anthropic 的关键洞见</p>
<p>最有效的 Agent 并不是最复杂的，而是<strong>配有好用工具的简单循环</strong>：</p>

<div class="highlight"><pre><span></span><code><span class="k">while</span> <span class="ow">not</span> <span class="n">done</span><span class="p">:</span>
    <span class="n">action</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">decide</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">tools</span><span class="p">)</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">execute</span><span class="p">(</span><span class="n">action</span><span class="p">)</span>
    <span class="n">context</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">result</span><span class="p">)</span>
    <span class="n">done</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">should_stop</span><span class="p">(</span><span class="n">context</span><span class="p">)</span>
</code></pre></div>

<p>智能来自（1）模型自身的能力、（2）工具描述的质量、（3）任务表述的清晰度——而非花哨的编排逻辑。Skill 为（2）和（3）提供了结构。</p>
</div>

#### 源自 Anthropic 方法的设计建议： {#design-recommendations-from-anthropics-approach}

<ul>
<li><p><strong>保持 Agent 循环简单</strong>：避免过度工程化控制流，让模型自己决策。</p></li>
<li><p><strong>在工具质量上投入</strong>：详尽、无歧义的工具描述比复杂的路由逻辑更有价值。</p></li>
<li><p><strong>使用结构化输出</strong>：强制模型以可解析的格式（JSON、函数调用）输出决策——可减少 skill 执行错误。</p></li>
<li><p><strong>内建恢复机制</strong>：Skill 应优雅地处理错误——用不同参数重试、请求澄清，或升级到由人接管。</p></li>
<li><p><strong>限制单个 skill 的范围</strong>：试图无所不能的 skill 什么也做不好。狭窄、定义明确的 skill 比宽泛的 skill 更易组合。</p></li>
</ul>
