# 17.4　记忆操作 {#sec:memory-operations}

## 17.4.1　写入：把信息提交到记忆 {#write-committing-to-memory}

<p>并非每条观察都应被存储。写入决策本质上是一个过滤问题：</p>



<div class="arithmatex">\[\text{Write}(e) = \mathbf{1}\!\left[\text{importance}(e) &gt; \tau\right],\]</div>

<p>其中 <span class="arithmatex">\(\tau\)</span> 为阈值，<span class="arithmatex">\(\text{importance}(e)\)</span> 可以是：</p>

<ul>
<li><p><strong>惊讶度（Surprise）：</strong> <span class="arithmatex">\(-\log p_\theta(e \mid \text{context})\)</span>——出乎意料的事件信息量更大。</p></li>
<li><p><strong>Reward 信号：</strong> 与高 <span class="arithmatex">\(|r_t|\)</span>（正或负）相关联的事件值得记住。</p></li>
<li><p><strong>LLM 自评：</strong> 提示模型在 1–10 的尺度上为重要性打分。</p></li>
</ul>

#### 冲突检测。 {#contradiction-detection.}

<p>在写入新事实 <span class="arithmatex">\(f_\text{new}\)</span> 之前，应检查它与已有记忆是否冲突：</p>



<div class="arithmatex">\[\text{Conflict}(f_\text{new}, \mathcal{M}) = \exists\, f \in \mathcal{M} : \text{Contradicts}(f_\text{new}, f).\]</div>

<p>冲突检测可以通过自然语言推理（NLI）模型实现，也可以通过提示 LLM 来完成。一旦发生冲突，Agent 必须决定：覆盖、带时间戳并保留两者，或者标记为需要人工审查。</p>

#### 记忆格式与粒度。 {#memory-format-and-granularity.}

<p>除了<em>存什么</em>，<em>怎么存</em>也至关重要。记忆条目从原子事实到冗长对话记录不等，各有不同的取舍：</p>

<p class="table-caption" id="table-17-1"><span class="table-number">表 17.1　</span>不同记忆粒度的取舍。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>格式</strong></th><th scope="col"><strong>优点</strong></th><th scope="col"><strong>缺点</strong></th></tr></thead>
<tbody><tr><td><strong>原子事实</strong></td><td></td><td></td></tr>
<tr><td>“用户偏好 Python。”</td><td>检索精确；可组合；去重与冲突检测容易</td><td>失去上下文；存在抽取错误；对细微信息脆弱</td></tr>
<tr><td><strong>结构化笔记</strong></td><td></td><td></td></tr>
<tr><td>(A-MEM <a class="book-citation" id="cite-xu2025amem-1" href="../../references.md#ref-xu2025amem" aria-label="参考文献 400">[400]</a>)</td><td>元数据丰富（标签、链接）；支持图遍历；在精确性与上下文之间取得平衡</td><td>写入成本更高；需要预先设计 schema</td></tr>
<tr><td><strong>摘要式 Episode</strong></td><td></td><td></td></tr>
<tr><td>(MemGPT <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a>)</td><td>保留叙事连贯性；紧凑；适合多轮（Multi-Turn）推理</td><td>摘要有损；难以局部更新</td></tr>
<tr><td><strong>原始对话记录</strong></td><td>无损；无抽取错误；支持精确引用</td><td>存储量大；检索嘈杂；扫描代价高</td></tr></tbody>
</table>
</div>

<p>实际生产系统往往会组合多种粒度 <a class="book-citation" id="cite-chhikara2025mem0-1" href="../../references.md#ref-chhikara2025mem0" aria-label="参考文献 53">[53]</a>：抽取原子事实以支持精确召回，维护摘要式 Episode 以保留叙事性上下文，并把原始对话记录存入冷存储以满足可审计性需求。Generative Agents 架构 <a class="book-citation" id="cite-park2023generative-1" href="../../references.md#ref-park2023generative" aria-label="参考文献 285">[285]</a> 将观察存储为原子的“记忆对象”，附带自然语言描述、重要性分数与时间戳——同时支持精确检索与时序推理。</p>

#### 设计准则。 {#design-guidelines.}

<ul>
<li><p><strong>粒度匹配查询类型。</strong> 如果用户问的是事实性问题（“我的 API key 是什么？”），原子事实更佳；如果问的是上下文性问题（“我们当时为什么决定用 Redis？”），就需要 Episode 摘要。</p></li>
<li><p><strong>在可承受的前提下用尽可能细的粒度存储</strong>，然后在其上构建粗粒度视图。把原子事实摘要化很容易；但从有损摘要中恢复原子是不可能的。</p></li>
<li><p><strong>记录来源。</strong> 每条记忆都应回链至来源（对话轮次、文档、工具输出），以便 Agent 可校验、用户可审计。</p></li>
</ul>

## 17.4.2　读取 / 检索 {#read-retrieve}

#### 查询的构造。 {#query-formulation.}

<p>检索查询 <span class="arithmatex">\(q\)</span> 未必要直接使用原始观察。更好的策略包括：</p>

<ul>
<li><p><strong>HyDE（假设性文档 Embedding，Hypothetical Document Embeddings） <a class="book-citation" id="cite-gao2022precise-1" href="../../references.md#ref-gao2022precise" aria-label="参考文献 106">[106]</a>：</strong> 先生成一个假设性的答案，对其做 Embedding，然后以该 Embedding 作为查询。</p></li>
<li><p><strong>查询扩展（Query expansion）：</strong> 生成查询的多个改写，并取检索结果的并集。</p></li>
<li><p><strong>回退式 Prompt（Step-back prompting）：</strong> 在检索之前先把具体查询抽象为一个更一般的问题。</p></li>
</ul>

#### 时间衰减与近期性偏置。 {#temporal-decay-and-recency-bias.}

<p>更久远的记忆可能相关性更低。一个带时间权重的评分是：</p>



<div class="arithmatex">\[\text{score}(d, q, t) = \lambda \cdot \text{sim}(\mathbf{q}, \mathbf{v}_d) + (1-\lambda) \cdot \exp\!\left(-\frac{t - t_d}{\tau_\text{decay}}\right),\]</div>

<p>其中 <span class="arithmatex">\(t_d\)</span> 是记忆的创建时间，<span class="arithmatex">\(\tau_\text{decay}\)</span> 控制衰减速率。Generative Agents 论文 <a class="book-citation" id="cite-park2023generative-2" href="../../references.md#ref-park2023generative" aria-label="参考文献 285">[285]</a> 采用了类似的近期性加权检索。</p>

## 17.4.3　更新：冲突解决与巩固 {#update-conflict-resolution-and-consolidation}

<p>记忆巩固（consolidation）会合并相关记忆，以减少冗余并浮现出更高层的模式：</p>



<div class="arithmatex">\[\mathcal{M}&#x27; = \text{Consolidate}(\mathcal{M}) = \text{Cluster}(\mathcal{M}) \cup \text{Summarize}(\text{Cluster}(\mathcal{M})).\]</div>

#### 遗忘机制。 {#forgetting-mechanisms.}

<p>生物记忆会遗忘；人工记忆也应如此。可用策略包括：</p>

<ul>
<li><p><strong>LRU 驱逐：</strong> 容量超限时移除最近最少使用的条目。</p></li>
<li><p><strong>重要性加权的遗忘：</strong> <span class="arithmatex">\(p(\text{forget}\,|\,d) \propto \exp(-\text{importance}(d))\)</span>。</p></li>
<li><p><strong>间隔重复（Spaced repetition）：</strong> 反复被访问的记忆保留得更久，遵循指数遗忘曲线 <a class="book-citation" id="cite-ebbinghaus1885memory-1" href="../../references.md#ref-ebbinghaus1885memory" aria-label="参考文献 81">[81]</a>。</p></li>
</ul>

## 17.4.4　反思：元认知操作 {#reflect-meta-cognitive-operations}

<p>反思（reflection） <a class="book-citation" id="cite-park2023generative-3" href="../../references.md#ref-park2023generative" aria-label="参考文献 285">[285]</a><a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 是一种高阶记忆操作：Agent 读取自己的记忆并生成<em>洞见</em>：</p>



<div class="arithmatex">\[\text{Reflect}(\mathcal{M}) \to \{i_1, i_2, \ldots\} \subset \mathcal{M}_\text{semantic},\]</div>

<p>其中每条洞见 <span class="arithmatex">\(i_j\)</span> 都是从多条情景记忆中归纳出的更高层抽象。</p>

<div class="admonition example"><p class="admonition-title">反思的实践（Reflexion）</p>
<p>在三次解决某个编程题失败之后，Agent 进行反思：</p>

<ol>
<li><p>从情景记忆中检索这三次失败的 Episode。</p></li>
<li><p>生成一条洞见：“我总是忘记处理输入列表为空的边界情况。”</p></li>
<li><p>把这条洞见存入语义记忆。</p></li>
<li><p>在下一次尝试中检索这条洞见，并显式地检查空输入。</p></li>
</ol>

<p>这就是 Reflexion <a class="book-citation" id="cite-shinn2023reflexion-2" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 的核心机制：通过自我反思的“言语化强化学习”。</p>
</div>

#### 反思结果存放在哪里？ {#where-do-reflections-live}

<p>反思从情景记忆<em>读取</em>，但向语义记忆<em>写入</em>。所产生的洞见是上下文无关的概括（“始终检查空输入”），并非具体 Episode 的记录——因此属于语义记忆 <span class="arithmatex">\(\mathcal{M}_\text{semantic}\)</span>。然而在反思过程本身中，中间推理（检索到的 Episode + 综合 Prompt + 生成的洞见）占用的是<em>工作记忆</em>（上下文窗口）。一言以蔽之：</p>

<ul>
<li><p><strong>输入</strong>：情景记忆（具体的过往事件）</p></li>
<li><p><strong>计算</strong>：工作记忆（上下文中的主动推理）</p></li>
<li><p><strong>输出</strong>：语义记忆（持久化的、归纳后的洞见）</p></li>
</ul>

<p>这与生物记忆巩固相对应——情景经验在睡眠与反思中逐渐被转化为语义知识。</p>
