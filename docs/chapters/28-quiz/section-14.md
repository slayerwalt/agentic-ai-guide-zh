# 28.14　模型压缩问题 {#model-compression-questions}

<section class="quiz-question" id="q-source-62"><h2 class="question-title">Q：解释 AWQ。为什么保护 1% 的权重就能保留 99% 的质量？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：AWQ（激活感知权重量化，Activation-Aware Weight Quantization）观察到权重的重要性高度不均匀：乘以大激活值的权重对输出的贡献不成比例地大。</p>

<p>关键洞见：<span class="arithmatex">\(\|W \cdot X\|\)</span> 同时取决于 <span class="arithmatex">\(W\)</span> 和 <span class="arithmatex">\(X\)</span>。一个小权重乘以一个大激活，比一个大权重乘以一个接近零的激活更重要。</p>

<p>AWQ 识别出 top 1% 的“显著”通道（在校准数据上始终具有较大激活幅度的通道），并通过缩放对其加以保护：量化前将显著通道乘以因子 <span class="arithmatex">\(s &gt; 1\)</span>（然后在激活中除以 <span class="arithmatex">\(s\)</span>）。这降低了重要通道的相对量化误差。</p>

<p>结果：在 70B 模型上 4-bit 量化的质量损失 <span class="arithmatex">\(&lt;\)</span>1%，因为 99% 的非显著权重能容忍激进的量化。</p>

<p><em><strong>复习：</strong>第 <a href="../01-llm-architecture/index.md">1</a> 章与第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-63"><h2 class="question-title">Q：什么时候用 FP8、4-bit 量化、BF16？</h2>
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

<section class="quiz-question" id="q-source-64"><h2 class="question-title">Q：解释 NVIDIA 2:4 结构化稀疏。加速比和约束是什么？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：2:4 稀疏意味着：每 4 个连续元素一组中，恰好有 2 个必须为零。这在权重层面强制执行。</p>

<p><strong>硬件支持</strong>：A100/H100 的 Tensor Core 有专门的 2:4 稀疏 GEMM 指令，可以跳过零元素，无软件开销地实现恰好 <strong>2<span class="arithmatex">\(\times\)</span> 吞吐</strong>。</p>

<p><strong>约束</strong>：你必须在这一特定模式下实现<em>恰好</em>50% 的稀疏度。不能是 30% 或 70% 的稀疏度；不能是任意稀疏模式。剪枝必须遵守 4 元素组结构。</p>

<p><strong>如何实现</strong>：训练之后（或微调期间），对每组 4 个权重，将幅度最小的 2 个置零。然后微调数百步以恢复质量。质量损失：大模型（70B+）通常 <span class="arithmatex">\(&lt;\)</span>1%。</p>

<p><em><strong>复习：</strong>第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>
