# 第 1 章 · 文本生成与 Prompt 工程 {#reading-reading-04}

## 1.12　文本生成：解码方法 {#text-generation-decoding-methods}

<p>一个训练好的语言模型在每一步都会输出一个在词表上的概率分布：<span class="arithmatex">\(P(x_t | x_{&lt;t})\)</span>。<strong>解码策略</strong> 决定了我们如何从该分布中选择下一个 Token。这个选择深刻地影响输出质量、多样性和连贯性。</p>

### 1.12.1　贪心解码（Greedy Decoding） {#greedy-decoding}

<p>最简单的策略：总是选择概率最高的 Token。</p>



<div class="arithmatex">\[x_t = \arg\max_{v \in \mathcal{V}} P(v | x_{&lt;t})\]</div>

<p><strong>直觉：</strong> 就像在句子中总是选取最显而易见的下一个词。&quot;The capital of France is...&quot; <span class="arithmatex">\(\to\)</span> &quot;Paris&quot;（概率 0.92）。</p>

<p><strong>优点：</strong> 确定性、快速、无超参数。<br><strong>缺点：</strong> 产生重复、泛泛的文本。当一个早期的低概率 Token 会带来全局更优输出时会被错过。没有多样性。</p>

### 1.12.2　束搜索（Beam Search） {#beam-search}

<p>并行维护 <span class="arithmatex">\(B\)</span> 个（束宽）部分假设，每一步用 top-<span class="arithmatex">\(k\)</span> 个 Token 扩展每个假设，并保留得分最高的 <span class="arithmatex">\(B\)</span> 个完整序列：</p>



<div class="arithmatex">\[\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i | y_{&lt;i})\]</div>

<p>配合 <strong>长度归一化（length normalization）</strong> 避免偏好短序列：</p>



<div class="arithmatex">\[\text{score}_{\text{norm}}(y) = \frac{1}{|y|^\alpha} \sum_{i=1}^{|y|} \log P(y_i | y_{&lt;i}), \quad \alpha \in [0.6, 1.0]\]</div>

<p><strong>直觉：</strong> 就像同时在迷宫中探索多条路径，并在每个岔路口只保留 <span class="arithmatex">\(B\)</span> 条最有希望的路径。</p>

<p><strong>优点：</strong> 能找到比贪心解码似然更高的序列；适合翻译和摘要等存在单一“正确”输出的任务。<br><strong>缺点：</strong> 对开放式生成仍倾向于产生泛泛、重复的文本；计算量增加 <span class="arithmatex">\(B\)</span> 倍；所有束往往收敛到相似的输出。</p>

<figure id="figure-1-12"><a href="../../assets/figures/fig_013_fig13.png" target="_blank" rel="noopener" aria-label="查看图 1.12 原图" title="点击查看原图"><img src="../../assets/figures/fig_013_fig13.png" alt="束宽为 \(B=2\) 的束搜索。在每一步，只有得分最高的 2 个部分序列会被保留（蓝色）。得分较低的候选会被剪枝（灰色）。" loading="lazy"></a><figcaption><span class="figure-number">图 1.12　</span>束宽为 <span class="arithmatex">\(B=2\)</span> 的束搜索。在每一步，只有得分最高的 2 个部分序列会被保留（蓝色）。得分较低的候选会被剪枝（灰色）。</figcaption></figure>

### 1.12.3　多样化束搜索（Diverse Beam Search） {#diverse-beam-search}

<p>标准束搜索会产生近似重复的束。多样化束搜索（Diverse Beam Search） <a class="book-citation" id="cite-vijayakumar2018diverse-1" href="../../references.md#ref-vijayakumar2018diverse" aria-label="参考文献 360">[360]</a> 将束划分为 <span class="arithmatex">\(G\)</span> 个组，并在组之间添加一个 <strong>不相似性惩罚（dissimilarity penalty）</strong>：</p>



<div class="arithmatex">\[\text{score}_g(y_t) = \log P(y_t | y_{&lt;t}) - \lambda \sum_{g&#x27;&lt;g} \Delta(y_t, Y_{g&#x27;})\]</div>

<p>其中 <span class="arithmatex">\(\Delta\)</span> 衡量与较早组已选 Token 的重叠（例如 Hamming 多样性），<span class="arithmatex">\(\lambda\)</span> 控制多样性强度。</p>

<p><strong>直觉：</strong> 就像强迫一个头脑风暴小组生成不同的想法——每个子组若重复了较早子组所说的内容就会被惩罚。</p>

<p><strong>优点：</strong> 产生真正不同的候选序列；对重排序流水线很有用。<br><strong>缺点：</strong> 多样性惩罚可能降低单个束的质量；超参数更多（<span class="arithmatex">\(G\)</span>、<span class="arithmatex">\(\lambda\)</span>）。</p>

### 1.12.4　Top-<span class="arithmatex">\(k\)</span> 采样 {#top-k-sampling}

<p>仅从概率最高的 <span class="arithmatex">\(k\)</span> 个 Token 中采样，并重新分配概率质量：</p>



<div class="arithmatex">\[P&#x27;(v | x_{&lt;t}) = \begin{cases}
    \dfrac{P(v | x_{&lt;t})}{\sum_{v&#x27; \in \text{Top-}k} P(v&#x27; | x_{&lt;t})} &amp; \text{if } v \in \text{Top-}k \\[6pt]
    0 &amp; \text{otherwise}
  \end{cases}\]</div>

<p><strong>直觉：</strong> 在 &quot;The cat sat on the...&quot; 之后，只考虑前 <span class="arithmatex">\(k\)</span> 个合理的延续（&quot;mat&quot;、&quot;floor&quot;、&quot;couch&quot;……），忽略那些极不可能的（&quot;quantum&quot;、&quot;archipelago&quot;）。</p>

<p><strong>优点：</strong> 去除尾部噪声；实现简单。<br><strong>缺点：</strong> 固定的 <span class="arithmatex">\(k\)</span> 对尖峰分布而言过于严格（浪费概率质量），对平坦分布而言又过于宽松（让垃圾 Token 进入）。</p>

### 1.12.5　Top-<span class="arithmatex">\(p\)</span> 采样（核采样，Nucleus Sampling） {#top-p-nucleus-sampling}

<p>从累计概率超过 <span class="arithmatex">\(p\)</span> 的最小 Token 集合中采样：</p>



<div class="arithmatex">\[\text{Top-}p = \min \left\{ S \subseteq \mathcal{V} : \sum_{v \in S} P(v | x_{&lt;t}) \geq p \right\}\]</div>

<p>其中 Token 按概率降序排序，并逐个加入直到达到阈值 <span class="arithmatex">\(p\)</span>。</p>

<p><strong>直觉：</strong> 自适应地调整候选池大小。如果模型很自信（&quot;Paris&quot; 概率 95%），核就很小。如果不确定（&quot;The movie was...&quot;），核会扩展以包含许多合理的形容词。</p>

<p><strong>优点：</strong> 能适应分布形状；广泛使用的默认设置（<span class="arithmatex">\(p=0.9\)</span>–<span class="arithmatex">\(0.95\)</span>）。<br><strong>缺点：</strong> 在核的尾部仍会包含一些低质量 Token；阈值是一个单一的全局超参数。</p>

<figure id="figure-1-13"><a href="../../assets/figures/fig_014_fig14.png" target="_blank" rel="noopener" aria-label="查看图 1.13 原图" title="点击查看原图"><img src="../../assets/figures/fig_014_fig14.png" alt="Top-\(p\)（核）采样：Token 按概率排序并依次加入，直到累计质量达到 \(p=0.9\)。核（深蓝）会根据分布形状自适应其大小——这里 5 个 Token 就足够了。" loading="lazy"></a><figcaption><span class="figure-number">图 1.13　</span>Top-<span class="arithmatex">\(p\)</span>（核）采样：Token 按概率排序并依次加入，直到累计质量达到 <span class="arithmatex">\(p=0.9\)</span>。核（深蓝）会根据分布形状自适应其大小——这里 5 个 Token 就足够了。</figcaption></figure>

<div class="admonition tip"><p class="admonition-title">Top-<span class="arithmatex">\(k\)</span> 与 Top-<span class="arithmatex">\(p\)</span></p>
<p>考虑预测下一个词：</p>

<ul>
<li><p>在 &quot;2 + 2 =&quot; 之后：分布是尖峰的——top-1 Token（&quot;4&quot;）占据 99% 的质量。Top-<span class="arithmatex">\(k\)</span>=50 会浪费地考虑 49 个错误答案。Top-<span class="arithmatex">\(p\)</span>=0.9 则正确地只选取 &quot;4&quot;。</p></li>
<li><p>在 &quot;I enjoy eating&quot; 之后：分布是平坦的——许多食物都合理。Top-<span class="arithmatex">\(k\)</span>=5 太过严格。Top-<span class="arithmatex">\(p\)</span>=0.9 可能会包含 50 多个 Token，与实际的不确定性相匹配。</p></li>
</ul>

<p>Top-<span class="arithmatex">\(p\)</span> 自适应；top-<span class="arithmatex">\(k\)</span> 不会。实践中，两者常被结合使用：从 top-<span class="arithmatex">\(p\)</span> 与 top-<span class="arithmatex">\(k\)</span> 的交集中采样。</p>
</div>

### 1.12.6　Min-<span class="arithmatex">\(p\)</span> 采样 {#min-p-sampling}

<p>一种较新的替代方法，它设置一个 <strong>相对</strong> 概率下限 <a class="book-citation" id="cite-nguyen2024minp-1" href="../../references.md#ref-nguyen2024minp" aria-label="参考文献 266">[266]</a>：</p>



<div class="arithmatex">\[\text{Min-}p = \left\{ v \in \mathcal{V} : P(v | x_{&lt;t}) \geq p_{\min} \cdot \max_{v&#x27;} P(v&#x27; | x_{&lt;t}) \right\}\]</div>

<p>只有概率至少为最高 Token 概率 <span class="arithmatex">\(p_{\min}\)</span> 倍的 Token 才会被保留。</p>

<p><strong>直觉：</strong> “只考虑那些其可能性至少为最优 Token 10% 的 Token。” 如果最高 Token 的概率为 0.8，那么只有概率高于 0.08 的 Token 能留下。如果最高 Token 的概率为 0.05（非常不确定），概率高于 0.005 的 Token 都能留下——自然地扩大了候选池。</p>

<p><strong>优点：</strong> 会随模型置信度自然伸缩；在尖峰分布上比 top-<span class="arithmatex">\(p\)</span> 产生更少的退化样本；只有一个直观的参数。<br><strong>缺点：</strong> 较新、经受的实战检验较少；尚未在所有推理框架中成为标准。</p>

### 1.12.7　温度缩放 {#temperature-scaling}

<p>在应用任何采样策略之前，将 logits 除以温度 <span class="arithmatex">\(T\)</span>：</p>



<div class="arithmatex">\[P_T(v | x_{&lt;t}) = \frac{\exp(z_v / T)}{\sum_{v&#x27;} \exp(z_{v&#x27;} / T)}\]</div>

<ul>
<li><p><span class="arithmatex">\(T &lt; 1\)</span>：使分布更尖锐 <span class="arithmatex">\(\to\)</span> 更确定、更聚焦的输出。</p></li>
<li><p><span class="arithmatex">\(T = 1\)</span>：未修改的模型分布。</p></li>
<li><p><span class="arithmatex">\(T &gt; 1\)</span>：使分布更平坦 <span class="arithmatex">\(\to\)</span> 更随机、更有创造性的输出。</p></li>
<li><p><span class="arithmatex">\(T \to 0\)</span>：退化为贪心解码。<span class="arithmatex">\(T \to \infty\)</span>：退化为均匀采样。</p></li>
</ul>

<p><strong>常见设置：</strong> 事实性任务 <span class="arithmatex">\(T=0.7\)</span>，创意写作 <span class="arithmatex">\(T=1.0\)</span>–<span class="arithmatex">\(1.2\)</span>，代码/数学 <span class="arithmatex">\(T=0.0\)</span>（贪心）。</p>

### 1.12.8　对比解码（Contrastive Decoding） {#contrastive-decoding}

<p>对比解码（Contrastive Decoding） <a class="book-citation" id="cite-li2023contrastive-1" href="../../references.md#ref-li2023contrastive" aria-label="参考文献 210">[210]</a> 利用一个强模型（专家）与一个弱模型（业余者）之间的差异，来放大专家独有的知识：</p>



<div class="arithmatex">\[x_t = \arg\max_{v \in \mathcal{V}(x_{&lt;t})} \left[ \log P_{\text{expert}}(v | x_{&lt;t}) - \log P_{\text{amateur}}(v | x_{&lt;t}) \right]\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{V}(x_{&lt;t}) = \{v : P_{\text{expert}}(v | x_{&lt;t}) \geq \alpha \cdot \max_{v&#x27;} P_{\text{expert}}(v&#x27; | x_{&lt;t})\}\)</span> 是一个自适应的合理性约束。</p>

<p><strong>直觉：</strong> 业余模型捕捉到的是泛泛的、显而易见的模式（常用词、重复）。减去其对数概率就去除了这种“泛泛信号”，留下专家独有的知识和推理。就像从录音中去除背景噪声以听见信号一样。</p>

<p><strong>优点：</strong> 减少重复和泛泛的措辞；在不额外训练的情况下提升事实性和连贯性；可与任意一对模型搭配使用。<br><strong>缺点：</strong> 需要运行两个模型（计算量加倍）；对业余模型的选择敏感；合理性阈值 <span class="arithmatex">\(\alpha\)</span> 需要调优。</p>

### 1.12.9　重复惩罚 {#repetition-penalties}

<p>与采样策略正交，重复惩罚（repetition penalties）阻止模型重复 Token。给定 Token <span class="arithmatex">\(v\)</span> 的原始 logit <span class="arithmatex">\(z_v\)</span>（即 LM 头在 softmax <em>之前</em> 输出的未归一化得分），惩罚后的 logit 为：</p>



<div class="arithmatex">\[z_v&#x27; = \begin{cases}
    z_v / \theta &amp; \text{if } v \in \text{generated tokens and } z_v &gt; 0 \\
    z_v \cdot \theta &amp; \text{if } v \in \text{generated tokens and } z_v &lt; 0
  \end{cases}\]</div>

<p>其中 <span class="arithmatex">\(\theta &gt; 1\)</span> 是惩罚因子（通常为 1.1–1.3）。在两种情况下，其效果都是把 logit 推向零——从而降低先前生成 Token 的概率。频率惩罚和出现惩罚是 OpenAI API 使用的更简单的加性变体：</p>



<div class="arithmatex">\[z_v&#x27; = z_v - \alpha \cdot \text{count}(v) - \beta \cdot \mathbf{1}[v \in \text{generated}]\]</div>

<p>其中 <span class="arithmatex">\(\alpha\)</span> 是频率惩罚（与 <span class="arithmatex">\(v\)</span> 出现次数成正比），<span class="arithmatex">\(\beta\)</span> 是出现惩罚（对任何先前出现给予恒定惩罚）。</p>

### 1.12.10　实际对比 {#practical-comparison}

<p class="table-caption" id="table-1-15"><span class="table-number">表 1.15　</span>LLM 文本生成中各解码方法的对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>是否确定性</strong></th><th scope="col"><strong>多样性</strong></th><th scope="col"><strong>质量</strong></th><th scope="col"><strong>最佳适用</strong></th></tr></thead>
<tbody><tr><td>Greedy</td><td>是</td><td>无</td><td>中等</td><td>代码、事实问答</td></tr>
<tr><td>Beam Search（<span class="arithmatex">\(B\)</span>=4–8）</td><td>是</td><td>低</td><td>高（较窄）</td><td>翻译、摘要</td></tr>
<tr><td>Diverse Beam Search</td><td>是</td><td>中等</td><td>高</td><td>用于重排序的候选生成</td></tr>
<tr><td>Top-<span class="arithmatex">\(k\)</span>（<span class="arithmatex">\(k\)</span>=50）</td><td>否</td><td>中等</td><td>中等</td><td>通用生成</td></tr>
<tr><td>Top-<span class="arithmatex">\(p\)</span>（<span class="arithmatex">\(p\)</span>=0.9）</td><td>否</td><td>自适应</td><td>高</td><td>开放式任务的默认选择</td></tr>
<tr><td>Min-<span class="arithmatex">\(p\)</span>（<span class="arithmatex">\(p_{\min}\)</span>=0.1）</td><td>否</td><td>自适应</td><td>高</td><td>top-<span class="arithmatex">\(p\)</span> 的稳健替代</td></tr>
<tr><td>Contrastive</td><td>是</td><td>低</td><td>非常高</td><td>事实性、连贯的长文本</td></tr></tbody>
</table>
</div>

<div class="admonition example"><p class="admonition-title">实际中的解码：“Once upon a time”</p>
<p>给定提示 &quot;Once upon a time,&quot;：</p>

<ul>
<li><p><strong>Greedy</strong>: &quot;there was a young girl who lived in a small village...&quot;（泛泛的童话）</p></li>
<li><p><strong>Top-<span class="arithmatex">\(p\)</span>=0.9, <span class="arithmatex">\(T\)</span>=1.0</strong>: &quot;the rivers ran backwards and the fish learned to fly...&quot;（有创意、出人意料）</p></li>
<li><p><strong>Top-<span class="arithmatex">\(p\)</span>=0.9, <span class="arithmatex">\(T\)</span>=0.3</strong>: &quot;there was a kingdom ruled by a wise and just king...&quot;（连贯、常规）</p></li>
<li><p><strong>Contrastive</strong>: &quot;in the amber-lit corridors of a collapsing star, two minds argued about the nature of time...&quot;（独特，避免了陈词滥调）</p></li>
</ul>

<p>同一个模型、同一个提示——解码策略决定了输出的风格。</p>
</div>

### 1.12.11　受限解码（Constrained Decoding，结构化生成 Structured Generation） {#sec:constrained-decoding}

<p>上述所有方法都在每一步从 <em>完整的</em> 词表中采样。<strong>受限解码（Constrained Decoding）</strong> 限制允许的 Token 集合，从而 <em>保证</em> 输出符合某种形式语法——通常是 JSON 模式、正则表达式或上下文无关文法（Context-Free Grammar，CFG）。</p>

##### 核心机制。 {#core-mechanism.}

<p>在每个解码步 <span class="arithmatex">\(t\)</span>，会根据当前解析器状态计算一个 <strong>Token 掩码（token mask）</strong> <span class="arithmatex">\(M_t \subseteq \mathcal{V}\)</span>。只有 <span class="arithmatex">\(M_t\)</span> 中的 Token 保留其原始 logits；在 softmax 之前，所有其它 Token 都被设为 <span class="arithmatex">\(-\infty\)</span>：</p>



<div class="arithmatex">\[P&#x27;(v | x_{&lt;t}) = \begin{cases}
    P(v | x_{&lt;t}) / Z &amp; \text{if } v \in M_t \\
    0 &amp; \text{otherwise}
  \end{cases}\]</div>

<p>其中 <span class="arithmatex">\(Z = \sum_{v \in M_t} P(v | x_{&lt;t})\)</span> 用于重新归一化。由于掩码每一步都会变化（它取决于到目前为止已经生成的内容），约束是 <em>逐步</em> 强制实施的——模型在任何位置都不可能生成一个非法前缀。</p>

##### 从模式到掩码。 {#from-schema-to-mask.}

<p>编译流水线为：</p>



<div class="arithmatex">\[\text{JSON Schema} \;\xrightarrow{\text{compile}}\; \text{Regex}
  \;\xrightarrow{\text{compile}}\; \text{FSM (DFA)}
  \;\xrightarrow{\text{index}}\; \text{Token Mask per State}\]</div>

<p>有限状态机（FSM）的状态对应正则表达式中的位置。对每个状态，所有能使字符串保持在该语言内的词表 Token 都被预计算成索引（对每个模式只需一次性的开销）。在运行时，查询掩码只是一次 <span class="arithmatex">\(O(1)\)</span> 的表访问——对每个解码步带来的延迟可忽略不计。</p>

##### 关键库。 {#key-libraries.}

<ul>
<li><p><strong>Outlines</strong> <a class="book-citation" id="cite-willard2023outlines-1" href="../../references.md#ref-willard2023outlines" aria-label="参考文献 378">[378]</a>：将 JSON 模式和正则表达式编译为交织的、由 FSM 引导的生成。支持任何提供 logits 接口的模型。</p></li>
<li><p><strong>lm-format-enforcer</strong><sup id="note-ref-1"><a href="#note-1" aria-label="注 1">1</a></sup>：类似的 FSM 方法，重点放在与服务框架（vLLM、TGI）的集成上。</p></li>
<li><p><strong>Guidance</strong><sup id="note-ref-2"><a href="#note-2" aria-label="注 2">2</a></sup>（Microsoft）：将受限生成与控制流（循环、条件）交织在一起，可实现超越扁平模式的复杂结构化输出。</p></li>
<li><p><strong>XGrammar</strong> <a class="book-citation" id="cite-dong2024xgrammar-1" href="../../references.md#ref-dong2024xgrammar" aria-label="参考文献 77">[77]</a>：基于下推自动机的引擎，支持完整的上下文无关文法（不仅限于正则语言），被 MLC-LLM 和 vLLM 用于语法模式的解码。</p></li>
</ul>

##### 权衡。 {#trade-offs.}

<p>受限解码 <em>保证</em> 了语法有效性——不会出现事后解析失败，也无需重试。然而：</p>

<ul>
<li><p><strong>语义质量</strong>：如果模型对“正确”答案的概率质量位于语法之外，强制结构可能会降低内容质量。在实践中，对于设计良好的模式和训练良好的模型，这种情况很少出现。</p></li>
<li><p><strong>编译开销</strong>：必须为每个模式构建 FSM 索引。对于复杂模式可能需要 1–5 秒，但这一开销会在所有使用该模式的请求间摊销。</p></li>
<li><p><strong>语法覆盖</strong>：正则/FSM 可处理 JSON、YAML、SQL 片段以及大多数结构化格式。完整的 CFG（通过 XGrammar 或 LALR 解析器）可覆盖 Python 或 XML 等语言。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">何时使用受限解码</p>
<p>当模型输出的消费者是程序而不是人类时，就应使用受限解码。工具调用智能体、API 后端和数据抽取流水线都能从 <em>有保证的</em> 合法结构中受益。对于自由形式的散文或创意文本，无约束采样仍然是合适的选择。</p>
</div>

## 1.13　Prompt 工程 {#sec:prompt-engineering}

<p>Prompt 工程是一门设计 LLM 输入的学科，目标是在不修改模型权重的情况下可靠地激发预期行为。微调修改的是模型本身，而 Prompt 工程则通过精心的框架、示例与结构来挖掘模型<em>已有</em>的能力。它是改善 LLM 输出最快、最便宜、也最易获得的杠杆，即使在使用微调模型时也依然不可或缺。</p>

### 1.13.1　上下文学习（In-Context Learning，ICL） {#in-context-learning-icl}

<p>上下文学习 <a class="book-citation" id="cite-brown2020language-1" href="../../references.md#ref-brown2020language" aria-label="参考文献 32">[32]</a> 是大语言模型一项令人瞩目的能力：在推理时仅凭 Prompt 中提供的示例就能学习任务，完全无需梯度更新。模型从输入-输出对的模式中隐式推断任务，并泛化到新的输入上。</p>

<div class="admonition note"><p class="admonition-title">为什么上下文学习有效</p>
<ul>
<li><p><strong>隐式贝叶斯推断</strong>：模型在预训练时见过数百万种任务。Prompt 示例在模型学到的分布中<em>定位</em>到相关任务 <a class="book-citation" id="cite-xie2022explanation-1" href="../../references.md#ref-xie2022explanation" aria-label="参考文献 396">[396]</a>。</p></li>
<li><p><strong>归纳头（Induction heads）</strong>：特定的 attention 头学会复制模式（“A 之于 B，如 C 之于 ”），从而实现上下文泛化 <a class="book-citation" id="cite-olsson2022context-1" href="../../references.md#ref-olsson2022context" aria-label="参考文献 273">[273]</a>。</p></li>
<li><p><strong>任务向量（Task vectors）</strong>：ICL 在残差流中创建隐式的任务表示，引导生成朝向所演示的格式与内容 <a class="book-citation" id="cite-todd2024function-1" href="../../references.md#ref-todd2024function" aria-label="参考文献 354">[354]</a>。</p></li>
</ul>
</div>

##### 扩展行为。 {#scaling-behaviour.}

<p>ICL 主要在参数量约 1B 以上的模型中涌现，并随模型规模呈对数线性提升 <a class="book-citation" id="cite-brown2020language-2" href="../../references.md#ref-brown2020language" aria-label="参考文献 32">[32]</a>。较小的模型可以记住示例，但难以在同一上下文窗口内泛化到新的输入。</p>

### 1.13.2　零样本提示（Zero-Shot Prompting） {#zero-shot-prompting}

<p>零样本提示<em>不</em>提供任何示例，只有任务描述或指令。模型必须完全依靠预训练知识与指令微调来生成正确的格式与内容。</p>

<div class="admonition example"><p class="admonition-title">零样本分类</p>
<div class="highlight"><pre><span></span><code><span class="n">Classify</span> <span class="n">the</span> <span class="n">following</span> <span class="n">movie</span> <span class="n">review</span> <span class="k">as</span> <span class="n">POSITIVE</span> <span class="ow">or</span> <span class="n">NEGATIVE</span><span class="o">.</span>

<span class="n">Review</span><span class="p">:</span> <span class="s2">"The cinematography was breathtaking but the plot</span>
<span class="n">felt</span> <span class="n">rushed</span> <span class="ow">and</span> <span class="n">predictable</span><span class="o">.</span><span class="s2">"</span>

<span class="n">Sentiment</span><span class="p">:</span>
</code></pre></div>
</div>

##### 零样本何时奏效： {#when-zero-shot-works-well}

<ul>
<li><p>模型在预训练/SFT 阶段大量见过的任务（翻译、摘要、情感分析）</p></li>
<li><p>指令明确、输出格式无歧义</p></li>
<li><p>指令微调模型（如 ChatGPT、Claude、Llama-3-Instruct）在零样本任务上显著优于基座模型 <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a></p></li>
</ul>

##### 零样本何时失效： {#when-zero-shot-fails}

<p>全新格式、领域特定的标注方案，或模型无法仅凭指令推断出确切需求的模糊任务。</p>

### 1.13.3　少样本提示（Few-Shot Prompting） {#few-shot-prompting}

<p>少样本提示 <a class="book-citation" id="cite-brown2020language-3" href="../../references.md#ref-brown2020language" aria-label="参考文献 32">[32]</a> 在实际查询之前提供 <span class="arithmatex">\(k\)</span> 个输入-输出示例（即 “shots”）。它是上下文学习最常见的形式，也仍然是最有效的提示策略之一。</p>

<div class="admonition example"><p class="admonition-title">少样本命名实体识别</p>
<div class="highlight"><pre><span></span><code><span class="n">Extract</span> <span class="n">named</span> <span class="n">entities</span> <span class="kn">from</span><span class="w"> </span><span class="nn">the</span> <span class="n">text</span><span class="o">.</span> <span class="n">Format</span><span class="p">:</span> <span class="p">[</span><span class="n">ENTITY</span><span class="p">](</span><span class="n">TYPE</span><span class="p">)</span>

<span class="n">Text</span><span class="p">:</span> <span class="s2">"Apple released the iPhone 15 in Cupertino."</span>
<span class="n">Entities</span><span class="p">:</span> <span class="p">[</span><span class="n">Apple</span><span class="p">](</span><span class="n">ORG</span><span class="p">),</span> <span class="p">[</span><span class="n">iPhone</span> <span class="mi">15</span><span class="p">](</span><span class="n">PRODUCT</span><span class="p">),</span> <span class="p">[</span><span class="n">Cupertino</span><span class="p">](</span><span class="n">LOC</span><span class="p">)</span>

<span class="n">Text</span><span class="p">:</span> <span class="s2">"Elon Musk announced Tesla's new factory in Berlin."</span>
<span class="n">Entities</span><span class="p">:</span> <span class="p">[</span><span class="n">Elon</span> <span class="n">Musk</span><span class="p">](</span><span class="n">PER</span><span class="p">),</span> <span class="p">[</span><span class="n">Tesla</span><span class="p">](</span><span class="n">ORG</span><span class="p">),</span> <span class="p">[</span><span class="n">Berlin</span><span class="p">](</span><span class="n">LOC</span><span class="p">)</span>

<span class="n">Text</span><span class="p">:</span> <span class="s2">"OpenAI partnered with Microsoft to deploy GPT-4."</span>
<span class="n">Entities</span><span class="p">:</span>
</code></pre></div>
</div>

##### 少样本示例的关键设计原则： {#key-design-principles-for-few-shot-examples}

<ol>
<li><p><strong>多样性</strong>：覆盖预期输入的范围（不同长度、边界情况、类别）。</p></li>
<li><p><strong>顺序</strong>：将较难或更具代表性的示例放在最后（近因偏差，recency bias） <a class="book-citation" id="cite-lu2022fantastically-1" href="../../references.md#ref-lu2022fantastically" aria-label="参考文献 241">[241]</a>。</p></li>
<li><p><strong>标签平衡</strong>：分类任务中要包含所有类别的示例，以避免多数类偏差。</p></li>
<li><p><strong>格式一致性</strong>：每个示例都必须遵循<em>完全</em>相同的结构。模型会模仿这种模式。</p></li>
<li><p><strong>相关性</strong>：选用与目标查询语义相近的示例可获得最佳效果 <a class="book-citation" id="cite-liu2022makes-1" href="../../references.md#ref-liu2022makes" aria-label="参考文献 223">[223]</a>。</p></li>
</ol>

##### 需要多少个示例？ {#how-many-shots}

<p>性能通常会从 0 个示例提升到 4–8 个示例，然后趋于平稳。超过约 20 个示例后收益微乎其微，反而有占满上下文窗口的风险。Min 等人 <a class="book-citation" id="cite-min2022rethinking-1" href="../../references.md#ref-min2022rethinking" aria-label="参考文献 257">[257]</a> 表明，示例的<em>格式</em>与<em>标签空间</em>比标签正确性更重要——即使随机标签也有帮助（不过正确标签帮助更大）。</p>

### 1.13.4　指令跟随类 Prompt {#instruction-following-prompts}

<p>指令微调模型对清晰、结构化的指令响应最好。关键洞察是：把 Prompt 当作<em>规范说明</em>，而不是建议。</p>

<div class="admonition note"><p class="admonition-title">高效指令型 Prompt 的解剖</p>
<ol>
<li><p><strong>角色/人格</strong>：定义模型是谁（“你是一名资深数据科学家……”）</p></li>
<li><p><strong>任务</strong>：要做什么，表述清晰无歧义</p></li>
<li><p><strong>上下文</strong>：模型需要的背景信息</p></li>
<li><p><strong>约束</strong>：长度限制、语气、需要避免的内容、输出格式</p></li>
<li><p><strong>示例</strong>（可选）：展示期望的输出格式</p></li>
<li><p><strong>输入</strong>：实际待处理的数据</p></li>
</ol>
</div>

<div class="admonition example"><p class="admonition-title">带约束的指令型 Prompt</p>
<div class="highlight"><pre><span></span><code><span class="n">Role</span><span class="p">:</span> <span class="n">You</span> <span class="n">are</span> <span class="n">a</span> <span class="n">medical</span> <span class="n">literature</span> <span class="n">reviewer</span><span class="o">.</span>

<span class="n">Task</span><span class="p">:</span> <span class="n">Summarize</span> <span class="n">the</span> <span class="n">following</span> <span class="n">research</span> <span class="n">abstract</span> <span class="k">for</span> <span class="n">a</span>
<span class="n">general</span> <span class="n">audience</span><span class="o">.</span>

<span class="n">Constraints</span><span class="p">:</span>
<span class="o">-</span> <span class="n">Maximum</span> <span class="mi">3</span> <span class="n">sentences</span>
<span class="o">-</span> <span class="n">No</span> <span class="n">jargon</span> <span class="p">(</span><span class="n">explain</span> <span class="nb">any</span> <span class="n">technical</span> <span class="n">terms</span><span class="p">)</span>
<span class="o">-</span> <span class="n">Include</span> <span class="n">the</span> <span class="n">key</span> <span class="n">finding</span> <span class="ow">and</span> <span class="n">its</span> <span class="n">clinical</span> <span class="n">implication</span>
<span class="o">-</span> <span class="n">Do</span> <span class="n">NOT</span> <span class="n">speculate</span> <span class="n">beyond</span> <span class="n">what</span> <span class="n">the</span> <span class="n">abstract</span> <span class="n">states</span>

<span class="n">Abstract</span><span class="p">:</span> <span class="p">[</span><span class="o">...</span><span class="p">]</span>
</code></pre></div>
</div>

##### System Prompt 与 User Prompt 的对比。 {#system-prompts-vs.-user-prompts.}

<p>现代聊天 API 将<em>系统</em> Prompt（持久指令、角色定义）与<em>用户</em>消息（每轮输入）分离。多数模型对 system prompt 赋予更高的 attention 优先级，是放置角色定义、约束和输出格式说明的天然位置 <a class="book-citation" id="cite-openai2023gpt4-1" href="../../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a>。</p>

### 1.13.5　结构化输出 Prompt（JSON/XML） {#structured-output-prompts-jsonxml}

<p>对于程序化用途，最关键的提示技术是强制结构化输出——尤其是 JSON。</p>

<div class="admonition example"><p class="admonition-title">JSON 输出 Prompt</p>
<div class="highlight"><pre><span></span><code><span class="n">Extract</span> <span class="n">the</span> <span class="n">following</span> <span class="n">information</span> <span class="kn">from</span><span class="w"> </span><span class="nn">the</span> <span class="n">customer</span> <span class="n">email</span><span class="o">.</span>
<span class="n">Respond</span> <span class="n">ONLY</span> <span class="k">with</span> <span class="n">valid</span> <span class="n">JSON</span><span class="p">,</span> <span class="n">no</span> <span class="n">other</span> <span class="n">text</span><span class="o">.</span>

<span class="n">Schema</span><span class="p">:</span>
<span class="p">{</span>
  <span class="s2">"intent"</span><span class="p">:</span> <span class="s2">"refund|complaint|question|praise"</span><span class="p">,</span>
  <span class="s2">"urgency"</span><span class="p">:</span> <span class="s2">"low|medium|high"</span><span class="p">,</span>
  <span class="s2">"product_mentioned"</span><span class="p">:</span> <span class="s2">"string or null"</span><span class="p">,</span>
  <span class="s2">"summary"</span><span class="p">:</span> <span class="s2">"one sentence summary"</span>
<span class="p">}</span>

<span class="n">Email</span><span class="p">:</span> <span class="p">[</span><span class="o">...</span><span class="p">]</span>
</code></pre></div>
</div>

##### 获得可靠结构化输出的技巧： {#techniques-for-reliable-structured-output}

<ul>
<li><p><strong>Schema 优先</strong>：在输入<em>之前</em>展示精确的 JSON schema，模型会将其当作模板。</p></li>
<li><p><strong>约束解码（Constrained decoding）</strong>：使用基于语法的采样（如 Outlines <a class="book-citation" id="cite-willard2023outlines-1--section-13" href="../../references.md#ref-willard2023outlines" aria-label="参考文献 378">[378]</a>、Guidance）在 token 级别保证 JSON 语法合法。</p></li>
<li><p><strong>XML 标签</strong>：对于嵌套或多段输出，XML 标签（如 <code>&lt;thinking&gt;...&lt;/thinking&gt;</code>）提供无歧义的分隔符，模型能可靠地遵循。</p></li>
<li><p><strong>Pydantic/TypeScript 类型</strong>：提供类型定义有助于模型理解字段约束（OpenAI 的 function calling 内部就使用 JSON Schema）。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">Prompt 中的 JSON —— 常见陷阱</p>
<ul>
<li><p>模型可能加上 markdown 代码围栏（<code>‘‘‘json ... ‘‘‘</code>）——需明确要求输出原始 JSON。</p></li>
<li><p>嵌套对象和数组会增加幻觉风险——尽量将 schema 扁平化。</p></li>
<li><p>枚举字段（固定选项）比自由文本字段可靠得多。</p></li>
<li><p>始终要在程序中校验输出；不借助约束解码，任何 Prompt 都无法 100% 保证合规。</p></li>
</ul>
</div>

##### JSON Prompting：结构化输入本身。 {#json-prompting-structuring-the-input.}

<p>一种独立但互补的技术是 <em>JSON prompting</em>——将 Prompt <em>本身</em>格式化为 JSON 而非自然语言。这利用了模型在结构化数据（API、配置、代码）上的大量预训练，从而提升指令遵循度、降低歧义，并使多字段请求能被确定性地解析。</p>

<div class="admonition example"><p class="admonition-title">结合 System Prompt 的 JSON Prompting</p>
<div class="highlight"><pre><span></span><code><span class="o">===</span> <span class="n">SYSTEM</span> <span class="o">===</span>
<span class="n">You</span> <span class="n">are</span> <span class="n">a</span> <span class="n">senior</span> <span class="n">code</span> <span class="n">reviewer</span><span class="o">.</span> <span class="n">Analyze</span> <span class="n">code</span> <span class="k">for</span> <span class="n">bugs</span><span class="p">,</span>
<span class="n">security</span> <span class="n">issues</span><span class="p">,</span> <span class="ow">and</span> <span class="n">style</span> <span class="n">violations</span><span class="o">.</span> <span class="n">Always</span> <span class="n">respond</span>
<span class="ow">in</span> <span class="n">the</span> <span class="n">JSON</span> <span class="n">schema</span> <span class="n">provided</span><span class="o">.</span>

<span class="o">===</span> <span class="n">USER</span> <span class="p">(</span><span class="n">JSON</span> <span class="n">prompt</span><span class="p">)</span> <span class="o">===</span>
<span class="p">{</span>
  <span class="s2">"task"</span><span class="p">:</span> <span class="s2">"code_review"</span><span class="p">,</span>
  <span class="s2">"language"</span><span class="p">:</span> <span class="s2">"python"</span><span class="p">,</span>
  <span class="s2">"severity_filter"</span><span class="p">:</span> <span class="s2">"high"</span><span class="p">,</span>
  <span class="s2">"code"</span><span class="p">:</span> <span class="s2">"def login(user, pw):</span><span class="se">\n</span><span class="s2">    query = ..."</span><span class="p">,</span>
  <span class="s2">"output_schema"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"issues"</span><span class="p">:</span> <span class="p">[{</span>
      <span class="s2">"line"</span><span class="p">:</span> <span class="s2">"int"</span><span class="p">,</span>
      <span class="s2">"severity"</span><span class="p">:</span> <span class="s2">"critical|high|medium|low"</span><span class="p">,</span>
      <span class="s2">"category"</span><span class="p">:</span> <span class="s2">"security|bug|style|performance"</span><span class="p">,</span>
      <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">,</span>
      <span class="s2">"fix"</span><span class="p">:</span> <span class="s2">"string"</span>
    <span class="p">}],</span>
    <span class="s2">"overall_risk"</span><span class="p">:</span> <span class="s2">"critical|high|medium|low"</span>
  <span class="p">}</span>
<span class="p">}</span>
</code></pre></div>

<p><strong>为何 JSON prompting 有效：</strong></p>

<ul>
<li><p><strong>无歧义的字段边界</strong>：不会混淆一条指令何处结束、下一条何处开始。</p></li>
<li><p><strong>带类型的约束</strong>：像 <code>&quot;severity_filter&quot;: &quot;high&quot;</code> 这样的字段比 “只显示高严重度问题” 更清晰。</p></li>
<li><p><strong>Schema 即契约</strong>：在输入中加入 <code>output_schema</code> 复刻了模型在预训练中大量见过的 API 设计模式。</p></li>
<li><p><strong>System prompt 依然不可或缺</strong>：System 消息提供<em>角色</em>、<em>语气</em>与<em>行为约束</em>，这些内容并不适合放进 JSON 负载里。</p></li>
</ul>
</div>

### 1.13.6　思维链（Chain-of-Thought，CoT）Prompting {#chain-of-thought-cot-prompting}

<p>思维链 Prompting <a class="book-citation" id="cite-wei2022chain-1" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a> 要求模型在给出最终答案之前先生成中间推理步骤。这一简单技术在需要多步推理的任务上（算术、逻辑、常识推断与代码生成）能带来显著的性能提升。</p>

##### CoT 为何奏效： {#why-cot-works}

<ul>
<li><p><strong>将计算序列化</strong>：Transformer 深度固定，但生成长度可变。CoT 将并行（困难）问题转化为串行（容易）的步骤，相当于增加了模型的计算预算。</p></li>
<li><p><strong>降低累积误差</strong>：每一步都是更简单的子问题，单步错误率更低。</p></li>
<li><p><strong>暴露中间状态</strong>：让推理过程可审计、可调试。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">思维链的变体</p>
<span id="table-1-inline-23"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>零样本 CoT（Zero-Shot CoT） <a class="book-citation" id="cite-kojima2022large-1" href="../../references.md#ref-kojima2022large" aria-label="参考文献 189">[189]</a></td><td>在任意 Prompt 末尾追加 “Let’s think step by step”</td></tr>
<tr><td>少样本 CoT（Few-shot CoT） <a class="book-citation" id="cite-wei2022chain-2" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a></td><td>提供带显式推理链的示例</td></tr>
<tr><td>自一致性（Self-Consistency） <a class="book-citation" id="cite-wang2023selfconsistency-1" href="../../references.md#ref-wang2023selfconsistency" aria-label="参考文献 367">[367]</a></td><td>采样 <span class="arithmatex">\(N\)</span> 条 CoT 路径；对最终答案做多数投票</td></tr>
<tr><td>思维树（Tree of Thoughts） <a class="book-citation" id="cite-yao2023tree-1" href="../../references.md#ref-yao2023tree" aria-label="参考文献 407">[407]</a></td><td>探索多条推理分支并支持回溯</td></tr>
<tr><td>Plan-and-Solve <a class="book-citation" id="cite-wang2023planandsolve-1" href="../../references.md#ref-wang2023planandsolve" aria-label="参考文献 364">[364]</a></td><td>先规划步骤，再逐步执行</td></tr>
<tr><td>ReAct <a class="book-citation" id="cite-yao2023react-1" href="../../references.md#ref-yao2023react" aria-label="参考文献 408">[408]</a></td><td>交替进行 Reasoning 与 Acting（工具调用）</td></tr></tbody>
</table>
</div>
</div>

<div class="admonition example"><p class="admonition-title">零样本思维链</p>
<div class="highlight"><pre><span></span><code><span class="n">Q</span><span class="p">:</span> <span class="n">A</span> <span class="n">store</span> <span class="n">has</span> <span class="mi">45</span> <span class="n">apples</span><span class="o">.</span> <span class="n">They</span> <span class="n">sell</span> <span class="mi">3</span><span class="o">/</span><span class="mi">5</span> <span class="n">of</span> <span class="n">them</span> <span class="ow">in</span> <span class="n">the</span>
<span class="n">morning</span> <span class="ow">and</span> <span class="mi">1</span><span class="o">/</span><span class="mi">3</span> <span class="n">of</span> <span class="n">the</span> <span class="n">remaining</span> <span class="ow">in</span> <span class="n">the</span> <span class="n">afternoon</span><span class="o">.</span>
<span class="n">How</span> <span class="n">many</span> <span class="n">apples</span> <span class="n">are</span> <span class="n">left</span><span class="err">?</span>

<span class="n">Let</span><span class="s1">'s think step by step.</span>

<span class="n">A</span><span class="p">:</span> <span class="n">Morning</span> <span class="n">sales</span><span class="p">:</span> <span class="mi">45</span> <span class="o">*</span> <span class="mi">3</span><span class="o">/</span><span class="mi">5</span> <span class="o">=</span> <span class="mi">27</span> <span class="n">apples</span> <span class="n">sold</span><span class="o">.</span>
<span class="n">Remaining</span> <span class="n">after</span> <span class="n">morning</span><span class="p">:</span> <span class="mi">45</span> <span class="o">-</span> <span class="mi">27</span> <span class="o">=</span> <span class="mf">18.</span>
<span class="n">Afternoon</span> <span class="n">sales</span><span class="p">:</span> <span class="mi">18</span> <span class="o">*</span> <span class="mi">1</span><span class="o">/</span><span class="mi">3</span> <span class="o">=</span> <span class="mi">6</span> <span class="n">apples</span> <span class="n">sold</span><span class="o">.</span>
<span class="n">Remaining</span><span class="p">:</span> <span class="mi">18</span> <span class="o">-</span> <span class="mi">6</span> <span class="o">=</span> <span class="mi">12</span> <span class="n">apples</span><span class="o">.</span>
</code></pre></div>
</div>

##### 自一致性（Self-Consistency）。 {#self-consistency.}

<p>Wang 等人 <a class="book-citation" id="cite-wang2023selfconsistency-2" href="../../references.md#ref-wang2023selfconsistency" aria-label="参考文献 367">[367]</a> 表明，采样多条思维链推理路径并对最终答案做多数投票，能显著优于单路径 CoT。直觉是：正确的推理路径往往收敛于同一答案，而错误通常各有不同。这是用算力（生成 <span class="arithmatex">\(N\)</span> 个样本）换取准确率——在延迟不如正确性重要时非常实用。</p>

##### CoT 何时反而有害。 {#when-cot-hurts.}

<p>CoT 并非普遍有益。对于简单任务（单步分类、检索、格式化），CoT 会增加不必要的 token、提高延迟，甚至会因过度思考引入错误。应有选择地在确实需要多步推理的任务中使用 CoT。</p>

### 1.13.7　进阶 Prompting 技术 {#advanced-prompting-techniques}

##### 检索增强生成（Retrieval-Augmented Generation，RAG）。 {#retrieval-augmented-generation-rag.}

<p>RAG <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a> 不再只依赖模型的参数化记忆，而是检索相关文档并将其纳入 Prompt：</p>

<div class="highlight"><pre><span></span><code><span class="n">Context</span> <span class="p">(</span><span class="n">retrieved</span><span class="p">):</span> <span class="p">[</span><span class="n">document</span> <span class="n">chunks</span><span class="p">]</span>
<span class="n">Question</span><span class="p">:</span> <span class="p">[</span><span class="n">user</span> <span class="n">query</span><span class="p">]</span>
<span class="n">Answer</span> <span class="n">based</span> <span class="n">ONLY</span> <span class="n">on</span> <span class="n">the</span> <span class="n">provided</span> <span class="n">context</span><span class="o">.</span>
</code></pre></div>

<p>这将模型的回答锚定在可验证的来源上，在知识密集型任务中能显著减少幻觉。</p>

##### Prompt 链式调用与分解。 {#prompt-chaining-and-decomposition.}

<p>复杂任务受益于被拆解成由多个简单 Prompt 组成的流水线，每一步的输出作为下一步的输入：</p>

<ol>
<li><p>从文档中<em>提取</em>关键事实</p></li>
<li><p>在提取出的事实上进行<em>推理</em></p></li>
<li><p><em>格式化</em>最终答案</p></li>
</ol>

<p>每一步都可以使用不同的 Prompt 模板、模型或温度设置。相较单一的大 Prompt，这种方式更可控，也便于有针对性地调试。</p>

##### 宪法 AI（Constitutional AI）/ 自我批评（Self-Critique）。 {#constitutional-ai-self-critique.}

<p>Bai 等人 <a class="book-citation" id="cite-bai2022constitutional-1" href="../../references.md#ref-bai2022constitutional" aria-label="参考文献 16">[16]</a> 引入了一类 Prompt：让模型根据一组原则批评并修订自己的输出：</p>

<div class="highlight"><pre><span></span><code><span class="p">[</span><span class="n">Generate</span> <span class="n">initial</span> <span class="n">response</span><span class="p">]</span>
<span class="n">Critique</span><span class="p">:</span> <span class="n">Does</span> <span class="n">this</span> <span class="n">response</span> <span class="n">violate</span> <span class="nb">any</span> <span class="n">of</span> <span class="n">the</span> <span class="n">following</span>
<span class="n">principles</span><span class="err">?</span> <span class="p">[</span><span class="nb">list</span> <span class="n">principles</span><span class="p">]</span>
<span class="n">Revision</span><span class="p">:</span> <span class="n">Rewrite</span> <span class="n">the</span> <span class="n">response</span> <span class="n">addressing</span> <span class="n">the</span> <span class="n">critique</span><span class="o">.</span>
</code></pre></div>

##### 元提示（Meta-Prompting）与 Prompt 优化。 {#meta-prompting-and-prompt-optimization.}

<p>与手工编写 Prompt 不同，近期工作开始自动化 Prompt 设计：</p>

<ul>
<li><p><strong>APE</strong> <a class="book-citation" id="cite-zhou2023large-1" href="../../references.md#ref-zhou2023large" aria-label="参考文献 441">[441]</a>：使用 LLM 自动生成并打分候选 Prompt。</p></li>
<li><p><strong>DSPy</strong> <a class="book-citation" id="cite-khattab2023dspy-1" href="../../references.md#ref-khattab2023dspy" aria-label="参考文献 183">[183]</a>：将声明式的任务描述编译为优化过的 Prompt 流水线，并带有学习到的少样本示例。</p></li>
<li><p><strong>OPRO</strong> <a class="book-citation" id="cite-yang2024large-1" href="../../references.md#ref-yang2024large" aria-label="参考文献 402">[402]</a>：将 Prompt 优化视为优化问题，使用 LLM 作为优化器。</p></li>
</ul>

##### 专注推理查询（Attentive Reasoning Queries，ARQ）。 {#attentive-reasoning-queries-arq.}

<p>ARQ <a class="book-citation" id="cite-yang2025arq-1" href="../../references.md#ref-yang2025arq" aria-label="参考文献 403">[403]</a> 解决了标准 Prompting 的一个根本弱点：随着上下文变长，模型越来越容易 “丢失” Prompt 中段的关键信息（即 <em>lost-in-the-middle</em> 效应）。ARQ 通过将复杂查询分解为若干聚焦的子查询来缓解这一问题，每个子查询都旨在将模型的 attention 引向上下文的特定部分：</p>

<ol>
<li><p><strong>查询分解</strong>：将用户问题拆解为原子级的子问题，每个聚焦一个狭窄方面。</p></li>
<li><p><strong>专注检索</strong>：对每个子查询，仅检索或高亮与之相关的上下文片段——迫使模型聚焦于它。</p></li>
<li><p><strong>聚合</strong>：将子答案合并为连贯的最终回答。</p></li>
</ol>

<p>这在长文档问答、对大规模检索集合的多跳推理以及上下文窗口中包含大量工具输出的智能体任务中尤为有效。ARQ 可被视为一种结构化的思维链，它显式地管理模型<em>看哪里</em>，而不仅仅是<em>怎么推理</em>。</p>

### 1.13.8　最佳实践：打造高效 Prompt {#best-practices-crafting-effective-prompts}

<p>基于文献中的实证发现与从业者经验，以下原则能够稳定地提升 Prompt 质量：</p>

<div class="admonition note"><p class="admonition-title">Prompt 工程检查清单</p>
<ol>
<li><p><strong>具体且无歧义</strong>：把 “总结一下” 替换为 “用 2–3 个要点总结，每条不超过 20 字，聚焦可操作的发现”。</p></li>
<li><p><strong>展示而非陈述</strong>：一个好示例胜过 100 字的说明。拿不准时就加一个少样本示例。</p></li>
<li><p><strong>显式定义输出格式</strong>：指定 JSON schema、要点列表、表格格式或精确分隔符。绝不把格式留给模型自行解读。</p></li>
<li><p><strong>为输入数据使用分隔符</strong>：用明确的分隔符（<code>&quot;&quot;&quot;</code>、<code>&lt;input&gt;...&lt;/input&gt;</code>、<code>—</code>）包裹用户输入，将指令与数据分开。</p></li>
<li><p><strong>指定角色</strong>：“你是一名 [领域专家]，[具体行为]” 可以激发相关知识与语气。</p></li>
<li><p><strong>指明不要做什么</strong>：负向约束（“不要解释你的推理”、“绝不输出超过 5 项”）通常比正向约束更有效。</p></li>
<li><p><strong>为推理任务加入思维链</strong>：对数学、逻辑或多跳问题，追加 “Think step by step” 或提供解题示例。</p></li>
<li><p><strong>合理控制温度</strong>：事实/确定性任务用 <span class="arithmatex">\(T \approx 0\)</span>；创造性/多样化输出用 <span class="arithmatex">\(T \approx 0.7\)</span>–<span class="arithmatex">\(1.0\)</span>。</p></li>
<li><p><strong>基于实证迭代</strong>：把 Prompt 当作代码——做版本管理、A/B 测试，并在代表性评测集上度量性能。</p></li>
<li><p><strong>利用近因偏差</strong>：将最关键的指令和示例放在 Prompt 的<em>末尾</em>（最接近生成位置）。</p></li>
</ol>
</div>

<p class="table-caption" id="table-1-16"><span class="table-number">表 1.16　</span>常见的 Prompting 失效模式与解决方案。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>失效模式</strong></th><th scope="col"><strong>症状</strong></th><th scope="col"><strong>解决方案</strong></th></tr></thead>
<tbody><tr><td>指令遗忘</td><td>模型在长 Prompt 中忽略约束</td><td>把约束放到末尾；重复关键规则；使用 system prompt</td></tr>
<tr><td>格式漂移</td><td>输出开头正确，但在长生成中逐渐崩坏</td><td>使用约束解码；拆分为多段较短的链式 Prompt</td></tr>
<tr><td>谄媚（Sycophancy）</td><td>模型附和 Prompt 中错误的前提</td><td>加上 “若前提不正确请质疑”；使用系统级指令</td></tr>
<tr><td>细节幻觉</td><td>模型编造上下文中不存在的事实</td><td>加上 “若未知请说不知道”；使用带溯源的 RAG</td></tr>
<tr><td>过度拒答</td><td>模型因安全训练而拒绝无害请求</td><td>改写以澄清正当意图；显式说明请求合理性的上下文</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">Prompt 工程的心智模型</p>
<p>把 Prompt 工程想象成<em>用自然语言编程</em>。模型是一个强大但字面化的解释器——它会严格按照你的要求执行，并按其训练分布中最可能的方式去理解。软件工程的常见原则同样适用：</p>

<ul>
<li><p><strong>DRY（Don’t Repeat Yourself）</strong>：除非是为对抗长上下文中的 attention 衰减</p></li>
<li><p><strong>关注点分离</strong>：角色、约束、示例与输入分置于不同的 Prompt 段落</p></li>
<li><p><strong>测试驱动开发</strong>：先定义期望输出，再编写 Prompt</p></li>
<li><p><strong>版本控制</strong>：跟踪 Prompt 的迭代及其评测分数</p></li>
<li><p><strong>模块化</strong>：构建可复用的 Prompt 模板；将可变部分参数化</p></li>
</ul>

<p>当经过系统迭代后 Prompt 仍无法达到所需质量，这就是切换到微调（SFT）或强化学习（RLHF/DPO）的信号。</p>
</div>

<div class="footnotes"><hr><ol start="1">
<li id="note-1"><p><a href="https://github.com/noamgat/lm-format-enforcer">https://github.com/noamgat/lm-format-enforcer</a></p><a href="#note-ref-1" aria-label="返回注 1">↩</a></li>
<li id="note-2"><p><a href="https://github.com/guidance-ai/guidance">https://github.com/guidance-ai/guidance</a></p><a href="#note-ref-2" aria-label="返回注 2">↩</a></li>
</ol></div>
