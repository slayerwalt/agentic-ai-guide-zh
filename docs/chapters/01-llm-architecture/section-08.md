# 1.8　监督微调（Supervised Fine-Tuning，SFT） {#sec:sft}

<p>SFT 通过在精选的「提示——回复」对上训练，将预训练语言模型转化为指令遵循助手。它是从原始语言建模到 RLHF 的桥梁。</p>

## 1.8.1　SFT 目标 {#sft-objective}

<p>损失与 CLM 相同，但只在<strong>回复 Token</strong> 上计算：</p>



<div class="arithmatex">\[\mathcal{L}_\text{SFT} = -\frac{1}{|y|}\sum_{t=1}^{|y|} \log P_\theta(y_t \mid x_\text{prompt}, y_{&lt;t})\]</div>

<p>提示 Token 提供上下文但不接收梯度（标签设为 <span class="arithmatex">\(-100\)</span>）。</p>

## 1.8.2　数据质量：LIMA 原则 {#data-quality-the-lima-principle}

<p>Zhou 等 <a class="book-citation" id="cite-zhou2023lima-1" href="../../references.md#ref-zhou2023lima" aria-label="参考文献 439">[439]</a>证明，1,000 个精心策划的样例可以匹敌在 50K+ 嘈杂样例上训练的模型。关键要求：</p>

<ul>
<li><p><strong>多样性</strong>：涵盖问答、摘要、代码、数学、创意写作、多轮对话</p></li>
<li><p><strong>正确性</strong>：每条回复必须事实准确且格式良好</p></li>
<li><p><strong>长度平衡</strong>：混合短（单句）和长（多段落）回复</p></li>
<li><p><strong>去污染</strong>：移除与评估基准的重叠</p></li>
</ul>

## 1.8.3　训练配置 {#training-configuration}

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">SFTTrainer</span><span class="p">,</span> <span class="n">SFTConfig</span>

<span class="n">sft_config</span> <span class="o">=</span> <span class="n">SFTConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./sft_output"</span><span class="p">,</span>
    <span class="n">max_seq_length</span><span class="o">=</span><span class="mi">4096</span><span class="p">,</span>
    <span class="n">packing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>              <span class="c1"># 将多个短样例打包进完整序列</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">2e-5</span><span class="p">,</span>
    <span class="n">lr_scheduler_type</span><span class="o">=</span><span class="s2">"cosine"</span><span class="p">,</span>
    <span class="n">warmup_ratio</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
    <span class="n">weight_decay</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span>
    <span class="n">max_grad_norm</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">gradient_checkpointing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span> <span class="o">=</span> <span class="n">SFTTrainer</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span> <span class="n">args</span><span class="o">=</span><span class="n">sft_config</span><span class="p">,</span>
                     <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span> <span class="n">processing_class</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>

## 1.8.4　高效训练方案 {#efficient-training-solutions}

<p>标准的 HuggingFace 训练存在显著的性能浪费。多个库为 SFT 工作负载提供了即插即用的效率提升：</p>

#### Liger Kernel <a class="book-citation" id="cite-hsu2024liger-1" href="../../references.md#ref-hsu2024liger" aria-label="参考文献 145">[145]</a>。 {#liger-kernel-.}

<p>LinkedIn 开源的一组 <strong>Triton 融合内核</strong>，在训练时替换标准的 PyTorch 算子。关键融合包括：</p>

<ul>
<li><p><strong>融合交叉熵</strong>：将最后的线性投影、Softmax 和损失计算合并为单个内核——避免物化完整的 <span class="arithmatex">\((\text{batch} \times \text{seq} \times \text{vocab})\)</span> logit 张量。</p></li>
<li><p><strong>融合 RMSNorm / SwiGLU / RoPE</strong>：消除常见 LLM 构建块的中间内存分配。</p></li>
<li><p><strong>分块操作</strong>：将大张量按块处理，从而限制峰值内存。</p></li>
</ul>

<p><strong>效果</strong>：仅需一行集成代码（<code>apply_liger_kernel_to_llama()</code>）即可获得 20% 的吞吐量提升和最高 60% 的内存降低。兼容 FSDP、DeepSpeed 和 LoRA。</p>

#### Unsloth <a class="book-citation" id="cite-unsloth2024-1" href="../../references.md#ref-unsloth2024" aria-label="参考文献 132">[132]</a>。 {#unsloth-.}

<p>一个专注于微调的库，将<strong>自定义 CUDA/Triton 内核</strong>与激进的内存优化相结合：</p>

<ul>
<li><p>对 LoRA 层进行手工反向传播（避免 autograd 开销）。</p></li>
<li><p>带融合反量化的 4-bit QLoRA——在单张 48 GB GPU 上训练 70B 模型。</p></li>
<li><p>针对每种架构（Llama、Mistral、Qwen、Gemma）的智能 RoPE 与 Attention 内核融合。</p></li>
</ul>

<p><strong>效果</strong>：比原版 HuggingFace + PEFT 快 2–5<span class="arithmatex">\(\times\)</span>，显存占用降低 60–70%。对单 GPU 与消费级硬件工作流尤为重要。</p>

#### torchtune <a class="book-citation" id="cite-torchtune2024-1" href="../../references.md#ref-torchtune2024" aria-label="参考文献 349">[349]</a>。 {#torchtune-.}

<p>Meta 的原生 PyTorch 微调库（2025 年开发逐步收尾），其设计围绕<strong>可组合性</strong>而非单体抽象：</p>

<ul>
<li><p>纯 PyTorch——没有 trainer 类；配方（recipe）是可读的单文件脚本。</p></li>
<li><p>与 <code>torch.compile</code>、FSDP2 以及激活检查点的原生集成。</p></li>
<li><p>对 QLoRA、全量微调与知识蒸馏的一等支持。</p></li>
<li><p>内置量化感知训练（QAT），用于训练后压缩。</p></li>
</ul>

<p><strong>效果</strong>：速度与自定义方案相当，但具备完整的可调试性，无框架锁定。</p>

<div class="admonition note"><p class="admonition-title">选择效率栈</p>
<ul>
<li><p><strong>单 GPU（<span class="arithmatex">\(\leq\)</span>1 GPU）上的快速 LoRA/QLoRA</strong>：Unsloth（训练上手最快，配置最少）</p></li>
<li><p><strong>多 GPU 全量微调</strong>：TRL/DeepSpeed + Liger Kernel（大规模下的最佳吞吐）</p></li>
<li><p><strong>研究 / 自定义训练循环</strong>：torchtune（透明、可改造、原生 PyTorch）</p></li>
</ul>

<p>它们是<em>互补的</em>：Liger 内核可同时在 TRL 和 torchtune 工作流中使用。</p>
</div>

## 1.8.5　最佳实践 {#best-practices}

<p class="table-caption" id="table-1-11"><span class="table-number">表 1.11　</span>SFT 训练指南。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>实践</strong></th><th scope="col"><strong>细节</strong></th></tr></thead>
<tbody><tr><td>打包（Packing）</td><td>将多个短样例拼接为一个序列（用 EOS 分隔）。避免填充浪费。</td></tr>
<tr><td>NEFTune <a class="book-citation" id="cite-jain2024neftune-1" href="../../references.md#ref-jain2024neftune" aria-label="参考文献 159">[159]</a></td><td>在 Embedding 上添加均匀噪声（<span class="arithmatex">\(\alpha=5\)</span>）。零代价提升 MT-Bench 5–15%。</td></tr>
<tr><td>对话模板</td><td>始终使用模型原生模板。模板不匹配会降低质量。</td></tr>
<tr><td>Epoch 数</td><td>大数据集 2–3 个；小型（<span class="arithmatex">\(&lt;\)</span>10K）精选集最多 5 个。过训练会导致格式记忆化。</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">SFT 还不够</p>
<p>SFT 能教会格式和基础的指令遵循，但难以可靠地教会：<em>偏好</em>（哪一个回复更好——需要 RLHF/DPO）、<em>拒绝</em>（何时不该回答——需要安全训练）、<em>校准</em>（说「我不知道」——需要带真实性奖励的 RL）、或<em>复杂推理</em>（多步推理链——需要带可验证奖励的 RL）。完整流水线是：Pretrain <span class="arithmatex">\(\to\)</span> SFT <span class="arithmatex">\(\to\)</span> RLHF/DPO。</p>
</div>
