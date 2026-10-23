# 17.1　动机：为什么 Agent 需要记忆 {#sec:memory-motivation}

<p>大语言模型本质上是无状态的函数近似器：给定 Prompt <span class="arithmatex">\(x\)</span>，它们产生续写分布 <span class="arithmatex">\(p_\theta(y \mid x)\)</span>。每一次推理调用都从零开始。<em>上下文窗口</em>（context window）——模型可以关注的有限 Token 序列——是生成时唯一可用的信息。对短小、自包含的任务而言这已足够；但对长周期的智能体任务而言，这是一个根本性瓶颈。</p>

<div class="admonition note"><p class="admonition-title">上下文窗口瓶颈</p>
<p>设 <span class="arithmatex">\(L\)</span> 为最大上下文长度（例如 GPT-4o 的 <span class="arithmatex">\(L = 128{,}000\)</span> 个 Token）。单个 Token 大约编码 4 个字符；一本典型书籍约含 <span class="arithmatex">\(\sim\!500{,}000\)</span> 词 <span class="arithmatex">\(\approx 670{,}000\)</span> 个 Token。即便不计成本，一个运行数日的自主 Agent 所累积的观察、工具输出和推理轨迹<em>不可能</em>塞进任何固定窗口。记忆系统正是对这一物理约束的工程回应。</p>
</div>

<p>当 Agent 缺乏持久化记忆时，会出现三种截然不同的失败模式：</p>

<ol>
<li><p><strong>上下文的灾难性遗忘（Catastrophic Forgetting）。</strong> 一旦事件滚出上下文窗口便不可挽回地丢失。Agent 无法回溯一万个 Token 前所做的决策。</p></li>
<li><p><strong>无法从经验中学习。</strong> 没有情景式（episodic）存储，每个 Episode 对 Agent 都像是第一次。成功的策略无法复用，错误则反复发生。</p></li>
<li><p><strong>缺乏个性化。</strong> 用户偏好、领域事实、关系历史必须在每次会话中重新建立，导致用户体验与效率双双下降。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">记忆作为认知架构</p>
<p>认知科学在生物智能体中区分出多种记忆系统 <a class="book-citation" id="cite-tulving1985memory-1" href="../../references.md#ref-tulving1985memory" aria-label="参考文献 356">[356]</a><a class="book-citation" id="cite-squire1992declarative-1" href="../../references.md#ref-squire1992declarative" aria-label="参考文献 337">[337]</a>：<em>工作记忆</em>（对信息的主动操控）、<em>情景记忆</em>（自传式事件）、<em>语义记忆</em>（世界知识）以及<em>程序记忆</em>（技能与习惯）。有效的智能体 AI 系统从类似的划分中获益——并非因为我们要模拟神经科学，而是因为这些类别真实地反映了截然不同的<em>访问模式</em>、<em>更新频率</em>和<em>检索机制</em>。</p>
</div>

<p>形式化地，我们把 Agent 建模为一个元组 <span class="arithmatex">\(\mathcal{A} = (\pi_\theta, \mathcal{M}, \mathcal{R}, \mathcal{W})\)</span>，其中 <span class="arithmatex">\(\pi_\theta\)</span> 是 Policy（LLM），<span class="arithmatex">\(\mathcal{M}\)</span> 是记忆存储，<span class="arithmatex">\(\mathcal{R}: \mathcal{Q} \times \mathcal{M} \to \mathcal{D}\)</span> 是把查询映射到检索结果的检索函数，<span class="arithmatex">\(\mathcal{W}: \mathcal{M} \times \mathcal{E} \to \mathcal{M}\)</span> 是用新经验 <span class="arithmatex">\(\mathcal{E}\)</span> 更新记忆的写入函数。在每一步 <span class="arithmatex">\(t\)</span>，Agent 观察到 <span class="arithmatex">\(o_t\)</span>，检索相关上下文 <span class="arithmatex">\(c_t = \mathcal{R}(o_t, \mathcal{M})\)</span>，并采取行动：</p>



<div class="arithmatex">\[a_t \sim \pi_\theta\!\left(\cdot \;\middle|\; [s_t;\, c_t;\, h_t]\right),\]</div>

<p>其中 <span class="arithmatex">\(s_t\)</span> 是当前的系统 Prompt，<span class="arithmatex">\(c_t\)</span> 是检索到的记忆，<span class="arithmatex">\(h_t\)</span> 是近期的上下文历史。行动之后，Agent 可写入新信息：<span class="arithmatex">\(\mathcal{M} \leftarrow \mathcal{W}(\mathcal{M},\, (o_t, a_t, r_t))\)</span>。</p>
