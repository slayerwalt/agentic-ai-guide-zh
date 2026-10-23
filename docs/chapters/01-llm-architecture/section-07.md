# 1.7　预训练：最佳实践 {#sec:pretraining}

<p>预训练是 LLM 开发中代价最高的阶段——消耗数百万 GPU 小时，需要对数据、算力和超参数进行精心编排。本节提炼 Llama-3 <a class="book-citation" id="cite-grattafiori2024llama3-1" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>、Chinchilla <a class="book-citation" id="cite-hoffmann2022chinchilla-1" href="../../references.md#ref-hoffmann2022chinchilla" aria-label="参考文献 140">[140]</a> 和 GPT-4 <a class="book-citation" id="cite-openai2023gpt4-1" href="../../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a> 的关键经验。</p>

## 1.7.1　训练目标 {#training-objective}

<p>所有现代仅解码器 LLM 都使用<strong>因果语言建模（Causal Language Modeling，CLM）</strong>：</p>



<div class="arithmatex">\[\mathcal{L}_\text{CLM} = -\frac{1}{T}\sum_{t=1}^T \log P_\theta(x_t \mid x_{&lt;t})\]</div>

<p>这个简单的目标——在足够的数据和规模下——无需显式监督就能产生涌现能力（上下文学习、推理、指令遵循） <a class="book-citation" id="cite-brown2020language-1" href="../../references.md#ref-brown2020language" aria-label="参考文献 32">[32]</a>。</p>

## 1.7.2　数据流水线 {#data-pipeline}

<div class="admonition note"><p class="admonition-title">预训练数据配方</p>
<ul>
<li><p><strong>规模</strong>：前沿模型为 1–15 万亿 Token（Llama-3：15T Token）</p></li>
<li><p><strong>来源</strong>：网页抓取（80%）、代码（10%）、书籍/论文（5%）、精选数据（5%）</p></li>
<li><p><strong>去重</strong>：MinHash + 精确子串去重可降低记忆化 <a class="book-citation" id="cite-lee2022deduplicating-1" href="../../references.md#ref-lee2022deduplicating" aria-label="参考文献 203">[203]</a></p></li>
<li><p><strong>质量过滤</strong>：基于困惑度的分类器、启发式过滤器（长度、语种识别、毒性）</p></li>
<li><p><strong>数据混合</strong>：跨域温度加权采样；为推理能力提升代码与数学的权重</p></li>
</ul>
</div>

## 1.7.3　扩展律 {#scaling-laws}

<p>Hoffmann 等 <a class="book-citation" id="cite-hoffmann2022chinchilla-2" href="../../references.md#ref-hoffmann2022chinchilla" aria-label="参考文献 140">[140]</a>表明，算力最优的训练需要平衡模型大小 <span class="arithmatex">\(N\)</span> 和数据大小 <span class="arithmatex">\(D\)</span>：<span class="arithmatex">\(N_\text{opt} \propto C^{0.50}\)</span>，<span class="arithmatex">\(D_\text{opt} \propto C^{0.50}\)</span>。70B 模型的算力最优点约在 1.4T Token。实际中，模型通常被<em>过训练</em>（Token 数超过 Chinchilla 最优值），因为推理成本随模型大小而非训练 Token 数扩展——较小的过训练模型部署成本更低。</p>

## 1.7.4　关键超参数 {#key-hyperparameters}

<p class="table-caption" id="table-1-10"><span class="table-number">表 1.10　</span>已发表模型的预训练超参数。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>设置</strong></th><th scope="col"><strong>Llama-3 405B</strong></th><th scope="col"><strong>Llama-3 8B</strong></th><th scope="col"><strong>Qwen-2.5 72B</strong></th><th scope="col"><strong>Mistral 7B</strong></th></tr></thead>
<tbody><tr><td>Token 数</td><td>15T</td><td>15T</td><td>18T</td><td>8T</td></tr>
<tr><td>批大小（Token）</td><td>16M</td><td>4M</td><td>4M</td><td>4M</td></tr>
<tr><td>峰值学习率</td><td><span class="arithmatex">\(8\text{e-}5\)</span></td><td><span class="arithmatex">\(3\text{e-}4\)</span></td><td><span class="arithmatex">\(3\text{e-}4\)</span></td><td><span class="arithmatex">\(3\text{e-}4\)</span></td></tr>
<tr><td>调度策略</td><td>WSD</td><td>WSD</td><td>Cosine</td><td>Cosine</td></tr>
<tr><td>权重衰减</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr>
<tr><td>上下文长度</td><td>8192</td><td>8192</td><td>4096<span class="arithmatex">\(\to\)</span>32K</td><td>8192</td></tr></tbody>
</table>
</div>

## 1.7.5　常见失败模式 {#common-failure-modes}

<div class="admonition warning"><p class="admonition-title">预训练陷阱</p>
<ul>
<li><p><strong>损失尖峰</strong>：由坏数据批次或数值不稳定引发的损失突增。Llama-3 报告通过回滚检查点并跳过问题批次来处理。</p></li>
<li><p><strong>记忆化</strong>：模型逐字复述训练数据。修复：激进去重；监控抽取攻击。</p></li>
<li><p><strong>上下文长度</strong>：在短序列上训练却部署到长上下文会失败。应在长文档上做继续预训练 + RoPE 缩放。</p></li>
</ul>
</div>

## 1.7.6　中期训练：为强化学习做准备 {#sec:mid-training}

<p>在原始预训练与后训练之间，一个新的流水线阶段已经浮现。<em>中期训练（Mid-training）</em>（有时也叫“继续预训练（continued pretraining）”或“退火（annealing）”）在高质量精选数据上提高 STEM、数学与代码的权重，同时把上下文长度扩展到 128K–256K token。它的目的不是传授新知识，而是<em>让模型为 RL 做好准备</em>：植入验证、回溯、子目标分解这些认知行为，供之后的强化学习放大。</p>

<div class="admonition warning"><p class="admonition-title">中期训练决定 RL 就绪度</p>
<p>OctoThinker <a class="book-citation" id="cite-octothinker2025-1" href="../../references.md#ref-octothinker2025" aria-label="参考文献 236">[236]</a> 证明：中期训练的数据构成决定了 base 模型是否会对 RL 产生响应。同一套 GRPO 配方在 Qwen（其中期训练富含推理内容）上带来了显著的推理提升，在 Llama（缺少此类内容）上却完全失败。差异不在模型大小或架构——而在于 base 模型是否已经展现出 RL 可以强化的自我纠正行为。</p>
</div>

<p>典型的中期训练使用 100B–500B token 的精选高质量数据，搭配余弦衰减的学习率。Llama 3 的“退火尾巴（annealing tail）”、MiMo 的三阶段混合（把数学+代码提高到 70%），以及 MAI-Thinking-1 对 STEM 的显式加权，都是这一模式的实例。</p>
