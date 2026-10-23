# 28.17　推测解码（Speculative Decoding）问题 {#speculative-decoding-questions}

<section class="quiz-question" id="q-source-69"><h2 class="question-title">Q：推测解码声称“无质量损失”。以不同方式生成 token 怎么能产生完全相同的输出分布？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：接受/拒绝方案保证了分布等价性：</p>

<p>对每个 draft token <span class="arithmatex">\(\hat{x}\)</span>，给定 draft 概率 <span class="arithmatex">\(q(\hat{x})\)</span> 与目标概率 <span class="arithmatex">\(p(\hat{x})\)</span>：</p>

<ul>
<li><p>以概率 <span class="arithmatex">\(\min(1, p(\hat{x})/q(\hat{x}))\)</span> 接受</p></li>
<li><p>拒绝时：从<em>残差分布</em> <span class="arithmatex">\(\propto \max(0, p(x) - q(x))\)</span> 中采样</p></li>
</ul>

<p>这在数学上等价于直接从 <span class="arithmatex">\(p\)</span>（目标）采样。证明草图：输出 token <span class="arithmatex">\(x\)</span> 的概率为 <span class="arithmatex">\(q(x) \cdot \min(1, p(x)/q(x)) + P(\text{reject}) \cdot \frac{\max(0, p(x)-q(x))}{\sum_y \max(0, p(y)-q(y))} = p(x)\)</span>。</p>

<p>加速来自摊销：当 draft 质量好（接受率高）时，一次目标模型前向就能确认多个 token。无论 draft 质量如何，这一保证始终成立——糟糕的 draft 只是带来更低的加速比（更多拒绝），而不是更差的质量。</p>

<p><em><strong>复习：</strong>第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-70"><h2 class="question-title">Q：比较 Medusa 和 Eagle 在推测解码上的差异。何时选择哪一个？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>Medusa</strong>：在目标模型上增加 <span class="arithmatex">\(k\)</span> 个并行预测头。每个头独立预测位置 <span class="arithmatex">\(t+i\)</span> 的 token。<em>优点</em>：无需独立模型，<span class="arithmatex">\(&lt;\)</span>1% 显存开销。<em>缺点</em>：各头独立预测——无法让位置 <span class="arithmatex">\(t+2\)</span> 以 <span class="arithmatex">\(t+1\)</span> 的预测为条件。接受率：60%–80%。</p>

<p><strong>Eagle</strong>：在目标模型隐状态上的轻量自回归解码器。Draft token 自回归生成（每个以前一个为条件）。<em>优点</em>：捕捉 token 间依赖 <span class="arithmatex">\(\to\)</span> 85%–95% 接受率。<em>缺点</em>：稍多显存（小型解码器）以及顺序的 draft 生成。</p>

<p><strong>选 Medusa 当</strong>：显存极度紧张；集成简单；中等加速即可（2–2.5<span class="arithmatex">\(\times\)</span>）。</p>

<p><strong>选 Eagle 当</strong>：需要最大加速（3–4<span class="arithmatex">\(\times\)</span>）；可承受额外的小模型；延迟敏感的单流生成。</p>

<p><strong>选 N-gram 当</strong>：输出重复性强（代码、结构化数据）；零成本；无需训练。</p>

<p><em><strong>复习：</strong>第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-71"><h2 class="question-title">Q：为什么推测解码在大 Batch 下<em>帮不上忙</em>？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：在大 Batch（<span class="arithmatex">\(\geq\)</span>64）下，自回归生成已经<em>计算高效</em>：权重读取成本在多个序列间摊销。算术强度接近 roofline 拐点。</p>

<p>推测解码会带来开销：</p>

<ol>
<li><p>Draft 生成成本（即便是小模型，大 Batch 下也不是免费的）</p></li>
<li><p>验证前向每个序列处理 <span class="arithmatex">\(k\)</span> 个额外 token（Batch <span class="arithmatex">\(\times\)</span> <span class="arithmatex">\(k\)</span> 个 token）</p></li>
<li><p>Draft 模型或 Medusa 头的显存</p></li>
<li><p>被拒绝的 token 浪费算力</p></li>
</ol>

<p>在 batch=1（延迟瓶颈、显存瓶颈）：推测把 1 token/步变成 3–4 token/步——巨大胜利。</p>

<p>在 batch=128（已经计算高效）：推测带来的额外 token 几乎不能提升吞吐，因为 GPU 已接近饱和。开销甚至可能<em>降低</em>吞吐。</p>

<p><strong>规则</strong>：推测解码用于延迟（小 Batch）；批处理用于吞吐（大 Batch）。别把它们组合在一起。</p>

<p><em><strong>复习：</strong>第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>
