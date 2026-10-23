# 18.2　Context Window 管理 {#subsec:context-management}

<p>Context window 是 Agent 的工作记忆。窗口中的每一个 Token 都意味着金钱与延迟的开销；而<em>不在</em>窗口中的每一个 Token 对模型而言都是不可见的。管理这一有限资源是 Agent 设计中最具影响力的工程决策之一。</p>

## 18.2.1　Context 预算问题 {#the-context-budget-problem}

<p>设 <span class="arithmatex">\(C\)</span> 为模型支持的最大 Context 长度（以 Token 计）。Context 被划分为若干互相竞争的部分：</p>

<span id="eq:context-budget"></span>

<div class="arithmatex">\[C \geq \underbrace{S}_{\text{system prompt}} + \underbrace{M}_{\text{memory/RAG}} + \underbrace{T}_{\text{tool defs}} + \underbrace{H}_{\text{history}} + \underbrace{R}_{\text{reserved output}}
\tag{18.1}\]</div>

<p>随着对话增长，<span class="arithmatex">\(H\)</span> 会无限扩张而 <span class="arithmatex">\(C\)</span> 保持不变。工具输出可能非常大（例如一个网页、一次代码执行结果），导致 <span class="arithmatex">\(T + H\)</span> 出现突然峰值。harness 必须持续强制执行式 <a href="#eq:context-budget">18.1</a>。</p>

<div class="admonition warning"><p class="admonition-title">静默截断陷阱</p>
<p>许多 LLM API 在输入超出 Context 限制时会静默截断，从 Prompt 的<em>中间</em>或<em>开头</em>丢弃 Token。这会导致模型丢失 system prompt、忘记早期指令，或基于不完整 Context 产生幻觉——而且不会有任何错误信号。务必在发送<em>之前</em>计算 Token 数，并显式处理溢出。</p>
</div>

## 18.2.2　Context 分配策略 {#context-allocation-strategies}

#### 固定预算分配。 {#fixed-budget-allocation.}

<p>为每个组件分配硬性 Token 上限：</p>

<span id="eq:fixed-budget"></span>

<div class="arithmatex">\[\begin{aligned}
S &amp;\leq \alpha \cdot C, \quad \alpha \approx 0.10 \\
M &amp;\leq \beta \cdot C,  \quad \beta \approx 0.20 \\
T &amp;\leq \gamma \cdot C, \quad \gamma \approx 0.10 \\
H &amp;\leq \delta \cdot C, \quad \delta \approx 0.50 \\
R &amp;\leq \epsilon \cdot C, \quad \epsilon \approx 0.10
\end{aligned}
\tag{18.2}\]</div>

<p>固定分配简单且可预测，但当某些组件较小时会浪费容量。</p>

#### 动态分配。 {#dynamic-allocation.-dup2}

<p>在每一轮求解一个带约束的优化问题：</p>



<div class="arithmatex">\[\max_{S, M, T, H, R} \; \text{Utility}(S, M, T, H, R) \quad \text{s.t.} \quad S + M + T + H + R \leq C
\tag{18.3}\]</div>

<p>其中 <span class="arithmatex">\(\text{Utility}\)</span> 是一个任务相关的评分函数（例如相关性分数的加权和）。在实践中，动态分配通常用贪心方式近似：先填充最高优先级的组件，再对低优先级的组件进行压缩或截断。</p>

## 18.2.3　Context 压缩 {#context-compression}

<p>当 <span class="arithmatex">\(H\)</span> 超出预算时，harness 必须在不丢失关键信息的前提下压缩历史。</p>

#### 对旧轮次进行摘要。 {#summarization-of-old-turns.}

<p>用 LLM 生成的摘要替换最旧的 <span class="arithmatex">\(k\)</span> 轮 <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a>：</p>



<div class="arithmatex">\[H&#x27; = \text{Summarize}(H_{1:k}) \;\|\; H_{k+1:n}
\tag{18.4}\]</div>

<p>摘要通常比原文短 5–10<span class="arithmatex">\(\times\)</span>。可以使用一个专用的&quot;summarizer&quot;模型（更小更便宜）来执行这一步骤。</p>

#### 选择性保留。 {#selective-retention.}

<p>根据每条消息与当前查询 <span class="arithmatex">\(q\)</span> 的相关性打分：</p>



<div class="arithmatex">\[\text{score}(m_i) = \text{sim}(e(m_i),\, e(q)) + \lambda \cdot \text{recency}(i)
\tag{18.5}\]</div>

<p>其中 <span class="arithmatex">\(e(\cdot)\)</span> 是 Embedding 函数，<span class="arithmatex">\(\text{recency}(i) = i/n\)</span>。按分数保留 top-<span class="arithmatex">\(k\)</span> 条消息。</p>

#### 重要性加权截断。 {#importance-weighted-truncation.}

<p>为每一轮分配重要性权重 <span class="arithmatex">\(w_i\)</span>（例如包含工具结果或用户更正的轮次权重更高）。优先截断权重最低的轮次：</p>



<div class="arithmatex">\[\min_{S \subseteq [n]} \sum_{i \notin S} w_i \quad \text{s.t.} \quad \sum_{i \in S} |m_i| \leq B_H
\tag{18.6}\]</div>

<p>这是 0/1 背包问题的一个变体，可以通过按 <span class="arithmatex">\(w_i / |m_i|\)</span> 排序进行贪心求解。</p>

## 18.2.4　滑动窗口方法 {#sliding-window-approaches}

<ul>
<li><p><strong>FIFO（先进先出）：</strong> 当窗口填满时丢弃最旧的消息。简单但会丢失早期 Context（例如最初的任务描述）。</p></li>
<li><p><strong>按重要性排序保留：</strong> 将 system prompt 和首条用户消息固定（pinned），对其余消息应用重要性评分。</p></li>
<li><p><strong>分层摘要（Hierarchical Summarization）：</strong> 维护一个多层摘要金字塔——最近的轮次原文保留，较旧的轮次以段落摘要保留，最旧的轮次合并为一个抽象摘要。</p></li>
</ul>

<figure id="figure-18-2"><a href="../../assets/figures/fig_054_sliding-window.png" target="_blank" rel="noopener" aria-label="查看图 18.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_054_sliding-window.png" alt="三种滑动窗口策略。红色 = 固定保留，灰色 = 丢弃，蓝色 = 原文保留，黄色 = 摘要，绿色 = 新消息。" loading="lazy"></a><figcaption><span class="figure-number">图 18.2　</span>三种滑动窗口策略。红色 = 固定保留，灰色 = 丢弃，蓝色 = 原文保留，黄色 = 摘要，绿色 = 新消息。</figcaption></figure>

## 18.2.5　递归 Context 分解 {#recursive-context-decomposition}

<p>上述策略——摘要、选择性保留、滑动窗口——都接受一个基本约束：<em>所有内容都必须装入单个 context window</em>。一种更激进的方法完全摒弃这一约束：让模型<strong>递归调用自身</strong>（或子模型）处理 Context 的分区，并跨调用聚合结果 <a class="book-citation" id="cite-zhang2025rlm-1" href="../../references.md#ref-zhang2025rlm" aria-label="参考文献 420">[420]</a>。</p>

<div class="admonition note"><p class="admonition-title">递归语言模型（Recursive Language Model, RLM）</p>
<p><strong>递归语言模型</strong>用递归分解替代单一的整体 LLM 调用 <span class="arithmatex">\(M(q, C)\)</span>：</p>



<div class="arithmatex">\[\text{RLM}(q, C) = M\!\left(q,\; \text{RLM}(q_1, C_1),\; \text{RLM}(q_2, C_2),\; \ldots\right)
\tag{18.7}\]</div>

<p>其中根模型将 Context <span class="arithmatex">\(C\)</span> 划分为若干块 <span class="arithmatex">\(\{C_i\}\)</span>，构造子查询 <span class="arithmatex">\(\{q_i\}\)</span>，派生递归调用以处理每一块，然后将结果综合为最终答案。任何单次调用都看不到完整 Context——模型在每个递归层级自行决定要查看什么。</p>
</div>

#### 为什么递归有效。 {#why-recursion-helps.}

<p>Context rot（上下文腐烂）——即模型准确率随 Context 长度增长而经验性地下降的现象——意味着即便拥有大 context window（128k+）的模型在长输入上的表现也会更差。通过让每次单独调用保持简短和聚焦，递归分解完全规避了这种退化。Zhang 等 <a class="book-citation" id="cite-zhang2025rlm-2" href="../../references.md#ref-zhang2025rlm" aria-label="参考文献 420">[420]</a> 证明，递归的 GPT-5-mini 在困难的长 Context 基准上<em>超越</em>了非递归的 GPT-5，同时每次查询成本更低。</p>

#### 实现模式。 {#implementation-pattern.}

<p>一个实用的 RLM harness 为模型提供一个 REPL 环境，将 Context 作为一个变量暴露其中。模型可以：</p>

<ol>
<li><p><strong>以编程方式检查（Inspect）</strong>Context（正则、切片、长度检查）。</p></li>
<li><p>根据结构或相关性将其<strong>划分（Partition）</strong>为可处理的块。</p></li>
<li><p>通过在每一块上派生递归 LLM 调用来进行<strong>子查询（Sub-query）</strong>。</p></li>
<li><p>将子结果<strong>聚合（Aggregate）</strong>为最终答案。</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">对大型代码库的递归摘要</p>
<div class="highlight"><pre><span></span><code><span class="k">def</span><span class="w"> </span><span class="nf">recursive_summarize</span><span class="p">(</span><span class="n">context</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">query</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                         <span class="n">model</span><span class="p">:</span> <span class="n">LLM</span><span class="p">,</span> <span class="n">max_tokens</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">8000</span><span class="p">):</span>
    <span class="c1"># 对超出窗口的 context 进行递归摘要。</span>
<span class="w">    </span><span class="sd">"""Recursively summarize context that exceeds window."""</span>
    <span class="k">if</span> <span class="n">count_tokens</span><span class="p">(</span><span class="n">context</span><span class="p">)</span> <span class="o">&lt;=</span> <span class="n">max_tokens</span><span class="p">:</span>
        <span class="c1"># 基础情况：context 可在一次调用中容纳</span>
        <span class="k">return</span> <span class="n">model</span><span class="o">.</span><span class="n">call</span><span class="p">(</span><span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">Context:</span><span class="se">\n</span><span class="si">{</span><span class="n">context</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

    <span class="c1"># 递归情况：分块并发起子查询</span>
    <span class="n">chunks</span> <span class="o">=</span> <span class="n">split_by_structure</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">max_tokens</span> <span class="o">//</span> <span class="mi">2</span><span class="p">)</span>
    <span class="n">sub_results</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">chunk</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">chunks</span><span class="p">):</span>
        <span class="n">sub_q</span> <span class="o">=</span> <span class="sa">f</span><span class="s2">"Summarize this section relevant to: </span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">"</span>
        <span class="n">sub_results</span><span class="o">.</span><span class="n">append</span><span class="p">(</span>
            <span class="n">recursive_summarize</span><span class="p">(</span><span class="n">chunk</span><span class="p">,</span> <span class="n">sub_q</span><span class="p">,</span> <span class="n">model</span><span class="p">,</span> <span class="n">max_tokens</span><span class="p">)</span>
        <span class="p">)</span>

    <span class="c1"># 聚合：综合子结果</span>
    <span class="n">combined</span> <span class="o">=</span> <span class="s2">"</span><span class="se">\n</span><span class="s2">---</span><span class="se">\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">sub_results</span><span class="p">)</span>
    <span class="k">return</span> <span class="n">model</span><span class="o">.</span><span class="n">call</span><span class="p">(</span>
        <span class="sa">f</span><span class="s2">"Given these partial summaries, answer: </span><span class="si">{</span><span class="n">query</span><span class="si">}</span><span class="s2">"</span>
        <span class="sa">f</span><span class="s2">"</span><span class="se">\n\n</span><span class="s2">Summaries:</span><span class="se">\n</span><span class="si">{</span><span class="n">combined</span><span class="si">}</span><span class="s2">"</span>
    <span class="p">)</span>
</code></pre></div>
</div>

<p>该模式可推广至摘要之外的场景：递归搜索（在数百万 Token 中查找针式信息）、递归分析（审计大型代码库）、递归抽取（解析文档语料库），都遵循相同的“分解—递归—聚合”结构。</p>

<figure id="fig:rlm"><a href="../../assets/figures/fig_055_rlm.png" target="_blank" rel="noopener" aria-label="查看图 18.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_055_rlm.png" alt="递归语言模型（RLM）。根模型将 Context 划分为若干块，在深度 1 派生子 LLM 调用，子调用可能进一步递归（深度 2）。结果向上回流（绿色虚线箭头）并聚合为最终答案。任何单次调用都不会处理完整 Context。" loading="lazy"></a><figcaption><span class="figure-number">图 18.3　</span>递归语言模型（RLM）。根模型将 Context 划分为若干块，在深度 1 派生子 LLM 调用，子调用可能进一步递归（深度 2）。结果向上回流（绿色虚线箭头）并聚合为最终答案。任何单次调用都不会处理完整 Context。</figcaption></figure>

## 18.2.6　Token 计数与预算监控 {#token-counting-and-budget-monitoring}

<div class="admonition note"><p class="admonition-title">预飞行（Pre-Flight）Token 检查</p>
<p>在每次 LLM 调用之前，harness 必须：</p>

<ol>
<li><p>统计组装后 Prompt 的 Token 数（使用模型的 tokenizer，而非字数近似）。</p></li>
<li><p>与 <span class="arithmatex">\(C - R\)</span> 进行比较（Context 上限减去预留输出 Token 数）。</p></li>
<li><p>如果超出预算：触发压缩、截断，或抛出显式错误。</p></li>
<li><p>按组件记录 Token 分解，用于可观测性。</p></li>
</ol>
</div>

<p>Token 计数应使用模型<em>精确</em>的 tokenizer（例如 OpenAI 模型用 <code>tiktoken</code>，开源模型用 <code>transformers</code> tokenizer）。经验法则的近似（“每个 Token 约 4 字符”）在代码、JSON 或非英文文本上可能偏差 20–40%。</p>
