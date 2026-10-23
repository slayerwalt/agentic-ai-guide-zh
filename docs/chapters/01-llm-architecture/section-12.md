# 1.12　文本生成：解码方法 {#text-generation-decoding-methods}

<p>一个训练好的语言模型在每一步都会输出一个在词表上的概率分布：<span class="arithmatex">\(P(x_t | x_{&lt;t})\)</span>。<strong>解码策略</strong> 决定了我们如何从该分布中选择下一个 Token。这个选择深刻地影响输出质量、多样性和连贯性。</p>

## 1.12.1　贪心解码（Greedy Decoding） {#greedy-decoding}

<p>最简单的策略：总是选择概率最高的 Token。</p>



<div class="arithmatex">\[x_t = \arg\max_{v \in \mathcal{V}} P(v | x_{&lt;t})\]</div>

<p><strong>直觉：</strong> 就像在句子中总是选取最显而易见的下一个词。&quot;The capital of France is...&quot; <span class="arithmatex">\(\to\)</span> &quot;Paris&quot;（概率 0.92）。</p>

<p><strong>优点：</strong> 确定性、快速、无超参数。<br><strong>缺点：</strong> 产生重复、泛泛的文本。当一个早期的低概率 Token 会带来全局更优输出时会被错过。没有多样性。</p>

## 1.12.2　束搜索（Beam Search） {#beam-search}

<p>并行维护 <span class="arithmatex">\(B\)</span> 个（束宽）部分假设，每一步用 top-<span class="arithmatex">\(k\)</span> 个 Token 扩展每个假设，并保留得分最高的 <span class="arithmatex">\(B\)</span> 个完整序列：</p>



<div class="arithmatex">\[\text{score}(y_{1:t}) = \sum_{i=1}^{t} \log P(y_i | y_{&lt;i})\]</div>

<p>配合 <strong>长度归一化（length normalization）</strong> 避免偏好短序列：</p>



<div class="arithmatex">\[\text{score}_{\text{norm}}(y) = \frac{1}{|y|^\alpha} \sum_{i=1}^{|y|} \log P(y_i | y_{&lt;i}), \quad \alpha \in [0.6, 1.0]\]</div>

<p><strong>直觉：</strong> 就像同时在迷宫中探索多条路径，并在每个岔路口只保留 <span class="arithmatex">\(B\)</span> 条最有希望的路径。</p>

<p><strong>优点：</strong> 能找到比贪心解码似然更高的序列；适合翻译和摘要等存在单一“正确”输出的任务。<br><strong>缺点：</strong> 对开放式生成仍倾向于产生泛泛、重复的文本；计算量增加 <span class="arithmatex">\(B\)</span> 倍；所有束往往收敛到相似的输出。</p>

<figure id="figure-1-12"><a href="../../assets/figures/fig_013_fig13.png" target="_blank" rel="noopener" aria-label="查看图 1.12 原图" title="点击查看原图"><img src="../../assets/figures/fig_013_fig13.png" alt="束宽为 \(B=2\) 的束搜索。在每一步，只有得分最高的 2 个部分序列会被保留（蓝色）。得分较低的候选会被剪枝（灰色）。" loading="lazy"></a><figcaption><span class="figure-number">图 1.12　</span>束宽为 <span class="arithmatex">\(B=2\)</span> 的束搜索。在每一步，只有得分最高的 2 个部分序列会被保留（蓝色）。得分较低的候选会被剪枝（灰色）。</figcaption></figure>

## 1.12.3　多样化束搜索（Diverse Beam Search） {#diverse-beam-search}

<p>标准束搜索会产生近似重复的束。多样化束搜索（Diverse Beam Search） <a class="book-citation" id="cite-vijayakumar2018diverse-1" href="../../references.md#ref-vijayakumar2018diverse" aria-label="参考文献 360">[360]</a> 将束划分为 <span class="arithmatex">\(G\)</span> 个组，并在组之间添加一个 <strong>不相似性惩罚（dissimilarity penalty）</strong>：</p>



<div class="arithmatex">\[\text{score}_g(y_t) = \log P(y_t | y_{&lt;t}) - \lambda \sum_{g&#x27;&lt;g} \Delta(y_t, Y_{g&#x27;})\]</div>

<p>其中 <span class="arithmatex">\(\Delta\)</span> 衡量与较早组已选 Token 的重叠（例如 Hamming 多样性），<span class="arithmatex">\(\lambda\)</span> 控制多样性强度。</p>

<p><strong>直觉：</strong> 就像强迫一个头脑风暴小组生成不同的想法——每个子组若重复了较早子组所说的内容就会被惩罚。</p>

<p><strong>优点：</strong> 产生真正不同的候选序列；对重排序流水线很有用。<br><strong>缺点：</strong> 多样性惩罚可能降低单个束的质量；超参数更多（<span class="arithmatex">\(G\)</span>、<span class="arithmatex">\(\lambda\)</span>）。</p>

## 1.12.4　Top-<span class="arithmatex">\(k\)</span> 采样 {#top-k-sampling}

<p>仅从概率最高的 <span class="arithmatex">\(k\)</span> 个 Token 中采样，并重新分配概率质量：</p>



<div class="arithmatex">\[P&#x27;(v | x_{&lt;t}) = \begin{cases}
    \dfrac{P(v | x_{&lt;t})}{\sum_{v&#x27; \in \text{Top-}k} P(v&#x27; | x_{&lt;t})} &amp; \text{if } v \in \text{Top-}k \\[6pt]
    0 &amp; \text{otherwise}
  \end{cases}\]</div>

<p><strong>直觉：</strong> 在 &quot;The cat sat on the...&quot; 之后，只考虑前 <span class="arithmatex">\(k\)</span> 个合理的延续（&quot;mat&quot;、&quot;floor&quot;、&quot;couch&quot;……），忽略那些极不可能的（&quot;quantum&quot;、&quot;archipelago&quot;）。</p>

<p><strong>优点：</strong> 去除尾部噪声；实现简单。<br><strong>缺点：</strong> 固定的 <span class="arithmatex">\(k\)</span> 对尖峰分布而言过于严格（浪费概率质量），对平坦分布而言又过于宽松（让垃圾 Token 进入）。</p>

## 1.12.5　Top-<span class="arithmatex">\(p\)</span> 采样（核采样，Nucleus Sampling） {#top-p-nucleus-sampling}

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

## 1.12.6　Min-<span class="arithmatex">\(p\)</span> 采样 {#min-p-sampling}

<p>一种较新的替代方法，它设置一个 <strong>相对</strong> 概率下限 <a class="book-citation" id="cite-nguyen2024minp-1" href="../../references.md#ref-nguyen2024minp" aria-label="参考文献 266">[266]</a>：</p>



<div class="arithmatex">\[\text{Min-}p = \left\{ v \in \mathcal{V} : P(v | x_{&lt;t}) \geq p_{\min} \cdot \max_{v&#x27;} P(v&#x27; | x_{&lt;t}) \right\}\]</div>

<p>只有概率至少为最高 Token 概率 <span class="arithmatex">\(p_{\min}\)</span> 倍的 Token 才会被保留。</p>

<p><strong>直觉：</strong> “只考虑那些其可能性至少为最优 Token 10% 的 Token。” 如果最高 Token 的概率为 0.8，那么只有概率高于 0.08 的 Token 能留下。如果最高 Token 的概率为 0.05（非常不确定），概率高于 0.005 的 Token 都能留下——自然地扩大了候选池。</p>

<p><strong>优点：</strong> 会随模型置信度自然伸缩；在尖峰分布上比 top-<span class="arithmatex">\(p\)</span> 产生更少的退化样本；只有一个直观的参数。<br><strong>缺点：</strong> 较新、经受的实战检验较少；尚未在所有推理框架中成为标准。</p>

## 1.12.7　温度缩放 {#temperature-scaling}

<p>在应用任何采样策略之前，将 logits 除以温度 <span class="arithmatex">\(T\)</span>：</p>



<div class="arithmatex">\[P_T(v | x_{&lt;t}) = \frac{\exp(z_v / T)}{\sum_{v&#x27;} \exp(z_{v&#x27;} / T)}\]</div>

<ul>
<li><p><span class="arithmatex">\(T &lt; 1\)</span>：使分布更尖锐 <span class="arithmatex">\(\to\)</span> 更确定、更聚焦的输出。</p></li>
<li><p><span class="arithmatex">\(T = 1\)</span>：未修改的模型分布。</p></li>
<li><p><span class="arithmatex">\(T &gt; 1\)</span>：使分布更平坦 <span class="arithmatex">\(\to\)</span> 更随机、更有创造性的输出。</p></li>
<li><p><span class="arithmatex">\(T \to 0\)</span>：退化为贪心解码。<span class="arithmatex">\(T \to \infty\)</span>：退化为均匀采样。</p></li>
</ul>

<p><strong>常见设置：</strong> 事实性任务 <span class="arithmatex">\(T=0.7\)</span>，创意写作 <span class="arithmatex">\(T=1.0\)</span>–<span class="arithmatex">\(1.2\)</span>，代码/数学 <span class="arithmatex">\(T=0.0\)</span>（贪心）。</p>

## 1.12.8　对比解码（Contrastive Decoding） {#contrastive-decoding}

<p>对比解码（Contrastive Decoding） <a class="book-citation" id="cite-li2023contrastive-1" href="../../references.md#ref-li2023contrastive" aria-label="参考文献 210">[210]</a> 利用一个强模型（专家）与一个弱模型（业余者）之间的差异，来放大专家独有的知识：</p>



<div class="arithmatex">\[x_t = \arg\max_{v \in \mathcal{V}(x_{&lt;t})} \left[ \log P_{\text{expert}}(v | x_{&lt;t}) - \log P_{\text{amateur}}(v | x_{&lt;t}) \right]\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{V}(x_{&lt;t}) = \{v : P_{\text{expert}}(v | x_{&lt;t}) \geq \alpha \cdot \max_{v&#x27;} P_{\text{expert}}(v&#x27; | x_{&lt;t})\}\)</span> 是一个自适应的合理性约束。</p>

<p><strong>直觉：</strong> 业余模型捕捉到的是泛泛的、显而易见的模式（常用词、重复）。减去其对数概率就去除了这种“泛泛信号”，留下专家独有的知识和推理。就像从录音中去除背景噪声以听见信号一样。</p>

<p><strong>优点：</strong> 减少重复和泛泛的措辞；在不额外训练的情况下提升事实性和连贯性；可与任意一对模型搭配使用。<br><strong>缺点：</strong> 需要运行两个模型（计算量加倍）；对业余模型的选择敏感；合理性阈值 <span class="arithmatex">\(\alpha\)</span> 需要调优。</p>

## 1.12.9　重复惩罚 {#repetition-penalties}

<p>与采样策略正交，重复惩罚（repetition penalties）阻止模型重复 Token。给定 Token <span class="arithmatex">\(v\)</span> 的原始 logit <span class="arithmatex">\(z_v\)</span>（即 LM 头在 softmax <em>之前</em> 输出的未归一化得分），惩罚后的 logit 为：</p>



<div class="arithmatex">\[z_v&#x27; = \begin{cases}
    z_v / \theta &amp; \text{if } v \in \text{generated tokens and } z_v &gt; 0 \\
    z_v \cdot \theta &amp; \text{if } v \in \text{generated tokens and } z_v &lt; 0
  \end{cases}\]</div>

<p>其中 <span class="arithmatex">\(\theta &gt; 1\)</span> 是惩罚因子（通常为 1.1–1.3）。在两种情况下，其效果都是把 logit 推向零——从而降低先前生成 Token 的概率。频率惩罚和出现惩罚是 OpenAI API 使用的更简单的加性变体：</p>



<div class="arithmatex">\[z_v&#x27; = z_v - \alpha \cdot \text{count}(v) - \beta \cdot \mathbf{1}[v \in \text{generated}]\]</div>

<p>其中 <span class="arithmatex">\(\alpha\)</span> 是频率惩罚（与 <span class="arithmatex">\(v\)</span> 出现次数成正比），<span class="arithmatex">\(\beta\)</span> 是出现惩罚（对任何先前出现给予恒定惩罚）。</p>

## 1.12.10　实际对比 {#practical-comparison}

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

## 1.12.11　受限解码（Constrained Decoding，结构化生成 Structured Generation） {#sec:constrained-decoding}

<p>上述所有方法都在每一步从 <em>完整的</em> 词表中采样。<strong>受限解码（Constrained Decoding）</strong> 限制允许的 Token 集合，从而 <em>保证</em> 输出符合某种形式语法——通常是 JSON 模式、正则表达式或上下文无关文法（Context-Free Grammar，CFG）。</p>

#### 核心机制。 {#core-mechanism.}

<p>在每个解码步 <span class="arithmatex">\(t\)</span>，会根据当前解析器状态计算一个 <strong>Token 掩码（token mask）</strong> <span class="arithmatex">\(M_t \subseteq \mathcal{V}\)</span>。只有 <span class="arithmatex">\(M_t\)</span> 中的 Token 保留其原始 logits；在 softmax 之前，所有其它 Token 都被设为 <span class="arithmatex">\(-\infty\)</span>：</p>



<div class="arithmatex">\[P&#x27;(v | x_{&lt;t}) = \begin{cases}
    P(v | x_{&lt;t}) / Z &amp; \text{if } v \in M_t \\
    0 &amp; \text{otherwise}
  \end{cases}\]</div>

<p>其中 <span class="arithmatex">\(Z = \sum_{v \in M_t} P(v | x_{&lt;t})\)</span> 用于重新归一化。由于掩码每一步都会变化（它取决于到目前为止已经生成的内容），约束是 <em>逐步</em> 强制实施的——模型在任何位置都不可能生成一个非法前缀。</p>

#### 从模式到掩码。 {#from-schema-to-mask.}

<p>编译流水线为：</p>



<div class="arithmatex">\[\text{JSON Schema} \;\xrightarrow{\text{compile}}\; \text{Regex}
  \;\xrightarrow{\text{compile}}\; \text{FSM (DFA)}
  \;\xrightarrow{\text{index}}\; \text{Token Mask per State}\]</div>

<p>有限状态机（FSM）的状态对应正则表达式中的位置。对每个状态，所有能使字符串保持在该语言内的词表 Token 都被预计算成索引（对每个模式只需一次性的开销）。在运行时，查询掩码只是一次 <span class="arithmatex">\(O(1)\)</span> 的表访问——对每个解码步带来的延迟可忽略不计。</p>

#### 关键库。 {#key-libraries.}

<ul>
<li><p><strong>Outlines</strong> <a class="book-citation" id="cite-willard2023outlines-1" href="../../references.md#ref-willard2023outlines" aria-label="参考文献 378">[378]</a>：将 JSON 模式和正则表达式编译为交织的、由 FSM 引导的生成。支持任何提供 logits 接口的模型。</p></li>
<li><p><strong>lm-format-enforcer</strong><sup id="note-ref-1"><a href="#note-1" aria-label="注 1">1</a></sup>：类似的 FSM 方法，重点放在与服务框架（vLLM、TGI）的集成上。</p></li>
<li><p><strong>Guidance</strong><sup id="note-ref-2"><a href="#note-2" aria-label="注 2">2</a></sup>（Microsoft）：将受限生成与控制流（循环、条件）交织在一起，可实现超越扁平模式的复杂结构化输出。</p></li>
<li><p><strong>XGrammar</strong> <a class="book-citation" id="cite-dong2024xgrammar-1" href="../../references.md#ref-dong2024xgrammar" aria-label="参考文献 77">[77]</a>：基于下推自动机的引擎，支持完整的上下文无关文法（不仅限于正则语言），被 MLC-LLM 和 vLLM 用于语法模式的解码。</p></li>
</ul>

#### 权衡。 {#trade-offs.}

<p>受限解码 <em>保证</em> 了语法有效性——不会出现事后解析失败，也无需重试。然而：</p>

<ul>
<li><p><strong>语义质量</strong>：如果模型对“正确”答案的概率质量位于语法之外，强制结构可能会降低内容质量。在实践中，对于设计良好的模式和训练良好的模型，这种情况很少出现。</p></li>
<li><p><strong>编译开销</strong>：必须为每个模式构建 FSM 索引。对于复杂模式可能需要 1–5 秒，但这一开销会在所有使用该模式的请求间摊销。</p></li>
<li><p><strong>语法覆盖</strong>：正则/FSM 可处理 JSON、YAML、SQL 片段以及大多数结构化格式。完整的 CFG（通过 XGrammar 或 LALR 解析器）可覆盖 Python 或 XML 等语言。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">何时使用受限解码</p>
<p>当模型输出的消费者是程序而不是人类时，就应使用受限解码。工具调用智能体、API 后端和数据抽取流水线都能从 <em>有保证的</em> 合法结构中受益。对于自由形式的散文或创意文本，无约束采样仍然是合适的选择。</p>
</div>

<div class="footnotes"><hr><ol start="1">
<li id="note-1"><p><a href="https://github.com/noamgat/lm-format-enforcer">https://github.com/noamgat/lm-format-enforcer</a></p><a href="#note-ref-1" aria-label="返回注 1">↩</a></li>
<li id="note-2"><p><a href="https://github.com/guidance-ai/guidance">https://github.com/guidance-ai/guidance</a></p><a href="#note-ref-2" aria-label="返回注 2">↩</a></li>
</ol></div>
