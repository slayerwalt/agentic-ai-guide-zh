# 28.11　Transformer 架构问题 {#transformer-architecture-questions}

<section class="quiz-question" id="q-source-52"><h2 class="question-title">Q：为什么在现代 LLM 中 RoPE 主导地位胜过可学习的绝对位置嵌入？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：RoPE 通过旋转矩阵将<em>相对</em>位置直接编码到 Q/K 点积中。关键优势：</p>

<ol>
<li><p>Attention 分数只依赖于相对距离 <span class="arithmatex">\(i-j\)</span>，而不依赖绝对位置——这能更好地泛化到未见过的序列长度。</p></li>
<li><p>可通过频率缩放（NTK-aware、YaRN）扩展到超出训练长度，无需重新训练。</p></li>
<li><p>无需额外参数（旋转由位置索引确定性地给出）。</p></li>
<li><p>可学习的绝对嵌入固定到训练长度，无法外推——在 4K 上下文训练的模型在 8K 时会失败。</p></li>
</ol>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-53"><h2 class="question-title">Q：解释 SwiGLU，以及它为何在现代 Transformer 中取代了 ReLU。</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：SwiGLU：<span class="arithmatex">\(\text{FFN}(x) = W_2 (\text{Swish}(W_1 x) \odot W_3 x)\)</span>，其中 <span class="arithmatex">\(\text{Swish}(x) = x \cdot \sigma(x)\)</span>。</p>

<p><strong>为什么它更好</strong>：</p>

<ul>
<li><p><em>门控</em>机制（<span class="arithmatex">\(\odot W_3 x\)</span>）让网络可以有选择地抑制或放大维度——比逐点 ReLU 更具表达力。</p></li>
<li><p>Swish 是平滑的（不存在 ReLU 零梯度区那样的死神经元）。</p></li>
<li><p>经验上：在相同 FLOP 预算下，语言建模基准上提升 1%–2%。</p></li>
<li><p>权衡：需要 3 个权重矩阵而非 2 个（通过将隐藏维度从 <span class="arithmatex">\(4d\)</span> 降到 <span class="arithmatex">\(8d/3\)</span> 来解决）。</p></li>
</ul>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-54"><h2 class="question-title">Q：什么是分组查询注意力（Grouped Query Attention, GQA），Llama-3 为何采用它？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 MHA：<span class="arithmatex">\(H\)</span> 个 query 头、<span class="arithmatex">\(H\)</span> 个 key 头、<span class="arithmatex">\(H\)</span> 个 value 头。GQA：<span class="arithmatex">\(H\)</span> 个 query 头但只有 <span class="arithmatex">\(G &lt; H\)</span> 个 key/value 头（在 query 组间共享）。</p>

<p>Llama-3 70B：64 个 query 头、8 个 KV 头（每个 KV 头被 8 个 query 头共享）。</p>

<p><strong>优势</strong>：</p>

<ul>
<li><p>KV 缓存大小缩小 <span class="arithmatex">\(H/G = 8\times\)</span>——对推理至关重要（长序列下 KV 缓存是主导的显存开销）。</p></li>
<li><p>质量损失极小（基准上 <span class="arithmatex">\(&lt;\)</span>0.5%），因为 KV 模式在不同头之间高度相关。</p></li>
<li><p>推理吞吐与 KV 缓存的减少成比例增长（更多序列能装进显存 = 更大的 Batch）。</p></li>
</ul>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-55"><h2 class="question-title">Q：为何 decoder-only 架构在 LLM 中胜过了 encoder-decoder？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<ol>
<li><p><strong>统一目标</strong>：预训练 = 微调 = 推理，都使用下一个 token 预测。不存在架构不匹配。</p></li>
<li><p><strong>参数效率</strong>：所有参数都对生成有贡献。在 encoder-decoder 中，纯生成任务下 encoder 参数被“浪费”。</p></li>
<li><p><strong>扩展更简单</strong>：一个模型、一个损失函数、一套超参数需要调。</p></li>
<li><p><strong>KV 缓存效率</strong>：decoder-only 只有一份 KV 缓存；encoder-decoder 有两份（encoder + decoder 的交叉注意力）。</p></li>
<li><p><strong>涌现的少样本能力</strong>：decoder-only 天然支持上下文学习（把示例前置到 Prompt 中）。</p></li>
</ol>

<p>对于固定输入长度的 seq2seq 任务（如翻译），encoder-decoder 仍然占优，但这部分在 LLM 用例中占比正在缩小。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>
