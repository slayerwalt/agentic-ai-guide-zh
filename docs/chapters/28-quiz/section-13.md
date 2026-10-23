# 28.13　LoRA 与 PEFT 问题 {#lora-and-peft-questions}

<section class="quiz-question" id="q-source-59"><h2 class="question-title">Q：LoRA 为什么有效？什么理论洞见支持低秩更新？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：Aghajanyan 等人 <a class="book-citation" id="cite-aghajanyan2020intrinsic-1" href="../../references.md#ref-aghajanyan2020intrinsic" aria-label="参考文献 3">[3]</a> 表明，微调是在一个非常低的<em>内在维度（intrinsic dimensionality）</em>上进行的——某个微调任务的有效参数空间远远小于模型的总参数量。175B 模型在给定任务上的内在维度可能 <span class="arithmatex">\(&lt;\)</span>10,000。</p>

<p>LoRA 直接利用了这一点：通过将更新约束到秩 <span class="arithmatex">\(r\)</span>（<span class="arithmatex">\(W&#x27; = W + BA\)</span>，<span class="arithmatex">\(B \in \mathbb{R}^{d \times r}\)</span>），它将每个权重矩阵的学习限制在一个 <span class="arithmatex">\(r\)</span> 维子空间内。由于真实的任务子空间是低维的，这几乎没有任何损失，同时将可训练参数减少 100–1000<span class="arithmatex">\(\times\)</span>。</p>

<p><strong>直觉</strong>：微调并不改变模型“知道什么”（满秩 <span class="arithmatex">\(W\)</span> 保持冻结）；它只调整<em>如何</em>为新任务组合现有知识——一个低秩扰动。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-60"><h2 class="question-title">Q：对 70B 模型，比较 QLoRA、全 LoRA、全量微调。何时选择每种？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<span id="table-28-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>显存</strong></th><th scope="col"><strong>GPU 数</strong></th><th scope="col"><strong>质量</strong></th><th scope="col"><strong>适用场景</strong></th></tr></thead>
<tbody><tr><td>全量微调</td><td>560+ GB</td><td>8+ A100</td><td>最佳</td><td>预算无上限；持续预训练</td></tr>
<tr><td>LoRA (<span class="arithmatex">\(r=16\)</span>)</td><td>145 GB</td><td>2 A100</td><td>95–98%</td><td>预算充足；一般微调</td></tr>
<tr><td>QLoRA (<span class="arithmatex">\(r=16\)</span>)</td><td>44 GB</td><td>1<span class="arithmatex">\(\times\)</span>48GB</td><td>93–96%</td><td>单 GPU；原型；资源受限</td></tr></tbody>
</table>
</div>

<p><strong>决策树</strong>：(1) 若任务需要深层知识变化 <span class="arithmatex">\(\to\)</span> 全量微调。(2) 若适配新风格/格式 <span class="arithmatex">\(\to\)</span> LoRA。(3) 若显存受限或快速迭代 <span class="arithmatex">\(\to\)</span> QLoRA。(4) 若秩很关键：从 <span class="arithmatex">\(r=16\)</span> 开始；如果训练 Loss 在全量微调水平之上停滞则增大。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../10-sft/index.md">10</a> 章（LLM 架构；SFT 最佳实践）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-61"><h2 class="question-title">Q：什么是 DoRA，为何它优于标准 LoRA？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：DoRA（权重分解低秩自适应，Weight-Decomposed Low-Rank Adaptation）将 <span class="arithmatex">\(W\)</span> 分解为幅度 <span class="arithmatex">\(\|W\|\)</span> 和方向 <span class="arithmatex">\(W/\|W\|\)</span>，然后仅对方向分量应用 LoRA：</p>



<div class="arithmatex">\[W&#x27; = m \odot \frac{W + BA}{\|W + BA\|}\]</div>

<p>其中 <span class="arithmatex">\(m\)</span>（幅度）也是可训练的，但仅作为每个输出神经元上的简单标量。</p>

<p><strong>为何有效</strong>：全量微调自然地独立更新幅度和方向。标准 LoRA 将二者耦合（低秩更新以受约束的方式同时改变两者）。DoRA 将它们解耦，让 LoRA 拥有与全量微调相同的“自由度”结构。结果：推理任务上提升 1%–3%，推理阶段无额外计算开销（合并适配器即可）。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>
