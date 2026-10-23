# 第 28 章 · 模型架构与效率 {#reading-reading-03}

## 28.11　Transformer 架构问题 {#transformer-architecture-questions}

<section class="quiz-question" id="q-source-52"><h3 class="question-title">Q：为什么在现代 LLM 中 RoPE 主导地位胜过可学习的绝对位置嵌入？</h3>
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

<section class="quiz-question" id="q-source-53"><h3 class="question-title">Q：解释 SwiGLU，以及它为何在现代 Transformer 中取代了 ReLU。</h3>
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

<section class="quiz-question" id="q-source-54"><h3 class="question-title">Q：什么是分组查询注意力（Grouped Query Attention, GQA），Llama-3 为何采用它？</h3>
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

<section class="quiz-question" id="q-source-55"><h3 class="question-title">Q：为何 decoder-only 架构在 LLM 中胜过了 encoder-decoder？</h3>
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

## 28.12　FlashAttention 问题 {#flash-attention-questions}

<section class="quiz-question" id="q-source-56"><h3 class="question-title">Q：FlashAttention 计算的结果与标准 attention 相同，但快 2–4 倍。如果 FLOPs 数相同，这怎么可能？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：FlashAttention 之所以更快，是因为它减少了<em>HBM 显存流量</em>，而非 FLOPs。标准 attention 在 HBM（慢）中物化 <span class="arithmatex">\(n \times n\)</span> 的 attention 矩阵，再读回来做 Softmax，再读一次做 <span class="arithmatex">\(PV\)</span> 乘法——总共在 <span class="arithmatex">\(O(n^2)\)</span> 数据上有 4 次 HBM 往返。</p>

<p>FlashAttention 对计算进行分块，使 <span class="arithmatex">\(n \times n\)</span> 矩阵完全在 SRAM（快，19 TB/s）中计算并消费，从不写入 HBM（2 TB/s）。“在线 Softmax”技巧通过维护滚动统计量来实现这一点。</p>

<p>结果：HBM 流量从 <span class="arithmatex">\(O(n^2 d)\)</span> 降到 <span class="arithmatex">\(O(n^2 d / M)\)</span>，其中 <span class="arithmatex">\(M\)</span> 为 SRAM 大小。相同 FLOPs，显存流量减少 10–50<span class="arithmatex">\(\times\)</span> <span class="arithmatex">\(\to\)</span> 实际墙钟加速 2–4<span class="arithmatex">\(\times\)</span>。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-57"><h3 class="question-title">Q：为什么 FlashAttention 对 FFN 层没帮助？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：FFN 层是<em>计算瓶颈</em>，而非显存瓶颈。其算术强度（大 Batch GEMM 下 <span class="arithmatex">\(I \approx 300\)</span> FLOP/byte）已经高于 roofline 拐点（A100 上为 156 FLOP/byte）。</p>

<p>FlashAttention 对 attention 有帮助是因为 attention 严重受限于显存（<span class="arithmatex">\(I \approx 1\)</span>–<span class="arithmatex">\(60\)</span> FLOP/byte）。通过将数据保留在 SRAM 中，它消除了显存瓶颈。</p>

<p>对 FFN 而言：瓶颈已经是 Tensor Core（而非显存带宽），所以减少显存流量没用。FFN 的收益来自量化（减小权重大小 <span class="arithmatex">\(\to\)</span> 提高算术强度）和更大的 Batch。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-58"><h3 class="question-title">Q：解释在线 Softmax 技巧，以及它对 FlashAttention 为何不可或缺。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 Softmax 在计算任何输出之前都需要全局最大值 <span class="arithmatex">\(m = \max_j x_j\)</span>——这要求先看到所有 <span class="arithmatex">\(n\)</span> 个 attention 分数，迫使物化完整的 <span class="arithmatex">\(n \times n\)</span> 矩阵。</p>

<p>在线 Softmax 技巧按顺序处理块，维护一个滚动的 <span class="arithmatex">\((m, \ell, O)\)</span> 状态：</p>

<ol>
<li><p>处理新块 <span class="arithmatex">\(\to\)</span> 更新滚动最大值：<span class="arithmatex">\(m_{\text{new}} = \max(m_{\text{old}}, \max(s_{\text{new}}))\)</span></p></li>
<li><p>重新缩放旧的求和：<span class="arithmatex">\(\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \text{new terms}\)</span></p></li>
<li><p>重新缩放输出：<span class="arithmatex">\(O_{\text{new}} = \text{rescaled}(O_{\text{old}}) + \text{new contribution}\)</span></p></li>
</ol>

<p>这在数学上是完全精确的——没有任何近似。它使得逐块处理成为可能，每块都能放进 SRAM，永远不需要在显存中保留完整的 <span class="arithmatex">\(n \times n\)</span> 矩阵。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

## 28.13　LoRA 与 PEFT 问题 {#lora-and-peft-questions}

<section class="quiz-question" id="q-source-59"><h3 class="question-title">Q：LoRA 为什么有效？什么理论洞见支持低秩更新？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：Aghajanyan 等人 <a class="book-citation" id="cite-aghajanyan2020intrinsic-1" href="../../references.md#ref-aghajanyan2020intrinsic" aria-label="参考文献 3">[3]</a> 表明，微调是在一个非常低的<em>内在维度（intrinsic dimensionality）</em>上进行的——某个微调任务的有效参数空间远远小于模型的总参数量。175B 模型在给定任务上的内在维度可能 <span class="arithmatex">\(&lt;\)</span>10,000。</p>

<p>LoRA 直接利用了这一点：通过将更新约束到秩 <span class="arithmatex">\(r\)</span>（<span class="arithmatex">\(W&#x27; = W + BA\)</span>，<span class="arithmatex">\(B \in \mathbb{R}^{d \times r}\)</span>），它将每个权重矩阵的学习限制在一个 <span class="arithmatex">\(r\)</span> 维子空间内。由于真实的任务子空间是低维的，这几乎没有任何损失，同时将可训练参数减少 100–1000<span class="arithmatex">\(\times\)</span>。</p>

<p><strong>直觉</strong>：微调并不改变模型“知道什么”（满秩 <span class="arithmatex">\(W\)</span> 保持冻结）；它只调整<em>如何</em>为新任务组合现有知识——一个低秩扰动。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-60"><h3 class="question-title">Q：对 70B 模型，比较 QLoRA、全 LoRA、全量微调。何时选择每种？</h3>
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

<section class="quiz-question" id="q-source-61"><h3 class="question-title">Q：什么是 DoRA，为何它优于标准 LoRA？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：DoRA（权重分解低秩自适应，Weight-Decomposed Low-Rank Adaptation）将 <span class="arithmatex">\(W\)</span> 分解为幅度 <span class="arithmatex">\(\|W\|\)</span> 和方向 <span class="arithmatex">\(W/\|W\|\)</span>，然后仅对方向分量应用 LoRA：</p>



<div class="arithmatex">\[W&#x27; = m \odot \frac{W + BA}{\|W + BA\|}\]</div>

<p>其中 <span class="arithmatex">\(m\)</span>（幅度）也是可训练的，但仅作为每个输出神经元上的简单标量。</p>

<p><strong>为何有效</strong>：全量微调自然地独立更新幅度和方向。标准 LoRA 将二者耦合（低秩更新以受约束的方式同时改变两者）。DoRA 将它们解耦，让 LoRA 拥有与全量微调相同的“自由度”结构。结果：推理任务上提升 1%–3%，推理阶段无额外计算开销（合并适配器即可）。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

## 28.14　模型压缩问题 {#model-compression-questions}

<section class="quiz-question" id="q-source-62"><h3 class="question-title">Q：解释 AWQ。为什么保护 1% 的权重就能保留 99% 的质量？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：AWQ（激活感知权重量化，Activation-Aware Weight Quantization）观察到权重的重要性高度不均匀：乘以大激活值的权重对输出的贡献不成比例地大。</p>

<p>关键洞见：<span class="arithmatex">\(\|W \cdot X\|\)</span> 同时取决于 <span class="arithmatex">\(W\)</span> 和 <span class="arithmatex">\(X\)</span>。一个小权重乘以一个大激活，比一个大权重乘以一个接近零的激活更重要。</p>

<p>AWQ 识别出 top 1% 的“显著”通道（在校准数据上始终具有较大激活幅度的通道），并通过缩放对其加以保护：量化前将显著通道乘以因子 <span class="arithmatex">\(s &gt; 1\)</span>（然后在激活中除以 <span class="arithmatex">\(s\)</span>）。这降低了重要通道的相对量化误差。</p>

<p>结果：在 70B 模型上 4-bit 量化的质量损失 <span class="arithmatex">\(&lt;\)</span>1%，因为 99% 的非显著权重能容忍激进的量化。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-63"><h3 class="question-title">Q：什么时候用 FP8、4-bit 量化、BF16？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<ul>
<li><p><strong>BF16</strong>：训练（RLHF 中的 Policy 模型），当精度重要时。任何被 Gradient 更新的模型的默认选择。</p></li>
<li><p><strong>FP8 (E4M3)</strong>：H100 上使用 Transformer Engine 训练（2<span class="arithmatex">\(\times\)</span> 吞吐，<span class="arithmatex">\(&lt;\)</span>0.5% 质量损失）。也用于 H100 上需要最大吞吐的推理。</p></li>
<li><p><strong>INT8/FP8 推理</strong>：RLHF 中的冻结模型（reference 模型、Reward 模型）——不被训练，所以降低精度是安全的。</p></li>
<li><p><strong>4-bit (AWQ/GPTQ)</strong>：大规模推理服务。部署中显存/质量权衡最佳。也用于 QLoRA 的基模。</p></li>
<li><p><strong>2-bit</strong>：实验性；显存极度受限的边缘部署。质量损失 5%–10%。</p></li>
</ul>

<p><strong>规则</strong>：推理时尽可能激进地量化，训练时保持 BF16（H100 上可用 FP8）。</p>

<p><em><strong>复习：</strong>第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-64"><h3 class="question-title">Q：解释 NVIDIA 2:4 结构化稀疏。加速比和约束是什么？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：2:4 稀疏意味着：每 4 个连续元素一组中，恰好有 2 个必须为零。这在权重层面强制执行。</p>

<p><strong>硬件支持</strong>：A100/H100 的 Tensor Core 有专门的 2:4 稀疏 GEMM 指令，可以跳过零元素，无软件开销地实现恰好 <strong>2<span class="arithmatex">\(\times\)</span> 吞吐</strong>。</p>

<p><strong>约束</strong>：你必须在这一特定模式下实现<em>恰好</em>50% 的稀疏度。不能是 30% 或 70% 的稀疏度；不能是任意稀疏模式。剪枝必须遵守 4 元素组结构。</p>

<p><strong>如何实现</strong>：训练之后（或微调期间），对每组 4 个权重，将幅度最小的 2 个置零。然后微调数百步以恢复质量。质量损失：大模型（70B+）通常 <span class="arithmatex">\(&lt;\)</span>1%。</p>

<p><em><strong>复习：</strong>第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

## 28.15　专家混合（Mixture of Experts）问题 {#mixture-of-experts-questions}

<section class="quiz-question" id="q-source-65"><h3 class="question-title">Q：Mixtral 8x7B 总共有 47B 参数但每个 token 只激活 13B。请解释它如何工作，以及为何高效。</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：Mixtral 将每个 FFN 层替换为 8 个并行的专家 FFN（每个专家的 FFN 部分约 <span class="arithmatex">\(\sim\)</span>7B 参数）。一个 router 网络为每个 token 选择 Top-2 专家。</p>

<p><strong>为什么总共 47B</strong>：Attention 层是共享的（不复制）= <span class="arithmatex">\(\sim\)</span>5B。FFN 专家：8 <span class="arithmatex">\(\times\)</span> <span class="arithmatex">\(\sim\)</span>5.25B = 42B。总计：<span class="arithmatex">\(\sim\)</span>47B。</p>

<p><strong>为什么 13B 激活</strong>：每个 token 只有 2 个专家被激活。激活参数 = attention（<span class="arithmatex">\(\sim\)</span>5B）+ 2 个 FFN 专家（<span class="arithmatex">\(\sim\)</span>2 <span class="arithmatex">\(\times\)</span> 5.25B）<span class="arithmatex">\(\approx\)</span> 13B。</p>

<p><strong>为什么高效</strong>：计算成本随<em>激活</em>参数（13B）扩展，与 13B 稠密模型相当。但容量（存储的知识）随<em>总</em>参数（47B）扩展，与更大的模型相当。结果：Mixtral 以 13B 的计算成本匹配 Llama-2 70B 的质量。</p>

<p><strong>显存成本</strong>：仍需把全部 47B 参数装进显存（所有专家都加载），所以显存 = 47B 模型，但计算 = 13B 模型。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-66"><h3 class="question-title">Q：MoE 中的负载均衡问题是什么，如何解决？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：不加约束时，router 倾向于将大部分 token 路由到 1–2 个“热门”专家（强者愈强的正反馈）。这导致：</p>

<ul>
<li><p>容量浪费：8 个专家中有 6 个未被使用，模型实际上缩小到 2 个专家的规模。</p></li>
<li><p>计算不均衡：如果每个专家在不同 GPU 上，热门专家会成为瓶颈，其他专家闲置。</p></li>
</ul>

<p><strong>解决方案</strong>：辅助负载均衡 Loss：<span class="arithmatex">\(\mathcal{L}_{\text{bal}} = \alpha \cdot N \sum_{i=1}^N f_i \cdot p_i\)</span>，其中 <span class="arithmatex">\(f_i\)</span> = 路由到专家 <span class="arithmatex">\(i\)</span> 的 token 比例，<span class="arithmatex">\(p_i\)</span> = 专家 <span class="arithmatex">\(i\)</span> 的平均 router 概率。这会惩罚不均匀的分布。</p>

<p><strong>替代方案</strong>：专家容量因子——对每个 Batch 中每个专家的最大 token 数做硬上限。溢出的 token 被丢弃或重新路由。</p>

<p>典型 <span class="arithmatex">\(\alpha\)</span>：0.01–0.1（足够小以不损害主 Loss，足够大以防止坍缩）。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

## 28.16　训练中的多样性问题 {#diversity-in-training-questions}

<section class="quiz-question" id="q-source-67"><h3 class="question-title">Q：如果一个 GRPO 组中 N 个响应全部相同会怎样？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：如果全部 <span class="arithmatex">\(N\)</span> 个响应相同：所有 Reward <span class="arithmatex">\(r_i\)</span> 相等，因此 <span class="arithmatex">\(\sigma_G = 0\)</span>，优势 <span class="arithmatex">\(\hat{A}_i = (r_i - \mu_G)/\sigma_G\)</span> 未定义（除以零）。实践中，实现把所有 <span class="arithmatex">\(\hat{A}_i = 0\)</span>，意味着<strong>零学习信号</strong>——这一步被浪费。</p>

<p><strong>预防</strong>：</p>

<ol>
<li><p><strong>温度</strong>：生成时使用 <span class="arithmatex">\(\tau = 0.7\)</span>–<span class="arithmatex">\(1.0\)</span>（不要贪心解码）。</p></li>
<li><p><strong>大 <span class="arithmatex">\(N\)</span></strong>：<span class="arithmatex">\(N=8\)</span>–<span class="arithmatex">\(16\)</span> 增加多样化响应的概率。</p></li>
<li><p><strong>重复拒绝</strong>：DAPO 的做法——拒绝重复响应并重采样。</p></li>
<li><p><strong>频率惩罚</strong>：生成时对重复的 n-gram 施加惩罚。</p></li>
<li><p><strong>监控</strong>：跟踪每组唯一响应比例。若 <span class="arithmatex">\(&lt;\)</span>50%，调高温度。</p></li>
</ol>

<p><em><strong>复习：</strong>第 <a href="../07-grpo/index.md">7</a> 章（GRPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-68"><h3 class="question-title">Q：解释 RLHF 中的多样性—质量权衡。如何检测模式坍缩？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>权衡</strong>：高多样性（高熵/高温度）= 多样但可能随机/低质量的响应。低多样性 = 一致但重复、被 Reward hack 的响应。</p>

<p><strong>检测模式坍缩</strong>（训练期间均应监控）：</p>

<ol>
<li><p><strong>响应熵</strong>：计算每个 token 的熵 <span class="arithmatex">\(H = -\sum p_i \log p_i\)</span>。若快速下降 <span class="arithmatex">\(\to\)</span> 坍缩。</p></li>
<li><p><strong>唯一 n-gram 比例</strong>：同一 Prompt 不同响应间唯一 4-gram 的比例。健康：<span class="arithmatex">\(&gt;\)</span>0.6。</p></li>
<li><p><strong>Reward 分布宽度</strong>：若 <span class="arithmatex">\(\sigma(\text{rewards})\)</span> 收缩到接近零 <span class="arithmatex">\(\to\)</span> 所有响应质量相同 <span class="arithmatex">\(\to\)</span> 可能完全一致。</p></li>
<li><p><strong>KL 散度</strong>：若 <span class="arithmatex">\(D_\text{KL}[\pi_\theta \| \pi_\text{ref}]\)</span> 快速增长，Policy 正远离参考 <span class="arithmatex">\(\to\)</span> 通常朝向某个狭窄模式。</p></li>
<li><p><strong>长度直方图</strong>：若所有响应收敛到相同长度 <span class="arithmatex">\(\to\)</span> 模板化行为。</p></li>
</ol>

<p><strong>修复</strong>：提高 KL 系数 <span class="arithmatex">\(\beta\)</span>、增大熵奖励、提高采样温度，或回滚到更早的 checkpoint。</p>

<p><em><strong>复习：</strong>第 <a href="../07-grpo/index.md">7</a> 章与第 <a href="../09-reward-models/index.md">9</a> 章（GRPO；Reward 模型训练）。</em></p>
</details></section>

## 28.17　推测解码（Speculative Decoding）问题 {#speculative-decoding-questions}

<section class="quiz-question" id="q-source-69"><h3 class="question-title">Q：推测解码声称“无质量损失”。以不同方式生成 token 怎么能产生完全相同的输出分布？</h3>
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

<section class="quiz-question" id="q-source-70"><h3 class="question-title">Q：比较 Medusa 和 Eagle 在推测解码上的差异。何时选择哪一个？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>Medusa</strong>：在目标模型上增加 <span class="arithmatex">\(k\)</span> 个并行预测头。每个头独立预测位置 <span class="arithmatex">\(t+i\)</span> 的 token。<em>优点</em>：无需独立模型，<span class="arithmatex">\(&lt;\)</span>1% 显存开销。<em>缺点</em>：各头独立预测——无法让位置 <span class="arithmatex">\(t+2\)</span> 以 <span class="arithmatex">\(t+1\)</span> 的预测为条件。接受率：60%–80%。</p>

<p><strong>Eagle</strong>：在目标模型隐状态上的轻量自回归解码器。Draft token 自回归生成（每个以前一个为条件）。<em>优点</em>：捕捉 token 间依赖 <span class="arithmatex">\(\to\)</span> 85%–95% 接受率。<em>缺点</em>：稍多显存（小型解码器）以及顺序的 draft 生成。</p>

<p><strong>选 Medusa 当</strong>：显存极度紧张；集成简单；中等加速即可（2–2.5<span class="arithmatex">\(\times\)</span>）。</p>

<p><strong>选 Eagle 当</strong>：需要最大加速（3–4<span class="arithmatex">\(\times\)</span>）；可承受额外的小模型；延迟敏感的单流生成。</p>

<p><strong>选 N-gram 当</strong>：输出重复性强（代码、结构化数据）；零成本；无需训练。</p>

<p><em><strong>复习：</strong>第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-71"><h3 class="question-title">Q：为什么推测解码在大 Batch 下<em>帮不上忙</em>？</h3>
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
