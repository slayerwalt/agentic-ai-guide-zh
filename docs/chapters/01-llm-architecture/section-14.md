# 1.14　模型压缩方法 {#model-compression-methods}

<p>模型压缩在保持质量的同时降低模型规模和推理成本。三种主要方法：量化（Quantization，降低数值精度）、剪枝（Pruning，移除参数）和蒸馏（Distillation，训练较小模型模仿更大模型）。</p>

## 1.14.1　量化（Quantization） {#quantization}

<p>量化通过将权重（以及可选的激活值）表示为更低精度的格式来缩小模型规模、降低推理成本。其核心权衡在于压缩率与质量下降之间。</p>

<div class="admonition note"><p class="admonition-title">量化概览</p>
<p>量化将模型权重（以及可选的激活值）的数值精度从 FP32/BF16 降至更低位宽的格式：</p>



<div class="arithmatex">\[x_q = \text{round}\!\left(\frac{x - z}{s}\right), \quad x_{\text{dequant}} = s \cdot x_q + z\]</div>

<p>其中 <span class="arithmatex">\(s\)</span> 是缩放因子，<span class="arithmatex">\(z\)</span> 是零点。</p>
</div>

<p class="table-caption" id="table-1-17"><span class="table-number">表 1.17　</span>面向 LLM 的量化方法。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>位宽</strong></th><th scope="col"><strong>类型</strong></th><th scope="col"><strong>核心思想</strong></th></tr></thead>
<tbody><tr><td><strong>GPTQ</strong> <a class="book-citation" id="cite-frantar2023gptq-1" href="../../references.md#ref-frantar2023gptq" aria-label="参考文献 99">[99]</a></td><td>4-bit</td><td>PTQ，仅权重</td><td>基于 optimal brain surgeon 进行逐层量化，最小化 <span class="arithmatex">\(\|WX - \hat{W}X\|^2\)</span>。</td></tr>
<tr><td><strong>AWQ</strong> <a class="book-citation" id="cite-lin2024awq-1" href="../../references.md#ref-lin2024awq" aria-label="参考文献 217">[217]</a></td><td>4-bit</td><td>PTQ，仅权重</td><td>保护显著权重（与大激活相对应）。1% 的权重承担 99% 的重要性。</td></tr>
<tr><td><strong>GGUF</strong> <a class="book-citation" id="cite-gerganov2023gguf-1" href="../../references.md#ref-gerganov2023gguf" aria-label="参考文献 109">[109]</a></td><td>2–8 bit</td><td>PTQ，仅权重</td><td>面向 CPU 优化的格式（llama.cpp）。按 block 量化，支持多种类型。</td></tr>
<tr><td><strong>FP8</strong> (E4M3)</td><td>8-bit</td><td>训练 + 推理</td><td>H100 原生支持。相比 BF16 提供 2<span class="arithmatex">\(\times\)</span> 吞吐。</td></tr>
<tr><td><strong>SmoothQuant</strong> <a class="book-citation" id="cite-xiao2023smoothquant-1" href="../../references.md#ref-xiao2023smoothquant" aria-label="参考文献 392">[392]</a></td><td>W8A8</td><td>PTQ，权重+激活</td><td>在量化前将激活离群值平滑迁移到权重。使 INT8 GEMM 可行。</td></tr>
<tr><td><strong>QAT</strong> <a class="book-citation" id="cite-liu2023llmqat-1" href="../../references.md#ref-liu2023llmqat" aria-label="参考文献 234">[234]</a></td><td>4-bit</td><td>QAT</td><td>使用模拟量化进行训练。质量最高但成本昂贵。</td></tr>
<tr><td><strong>AQLM</strong> <a class="book-citation" id="cite-egiazarian2024aqlm-1" href="../../references.md#ref-egiazarian2024aqlm" aria-label="参考文献 83">[83]</a></td><td>2-bit</td><td>PTQ，加性码字</td><td>通过学习到的加性量化码本实现极端压缩。</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">何时进行量化</p>
<ul>
<li><p><strong>推理服务</strong>：始终量化。W4A16（4-bit 权重，BF16 激活）是甜点——节省 2<span class="arithmatex">\(\times\)</span> 内存，对 70B 以上模型质量损失 <span class="arithmatex">\(&lt;\)</span>1%。</p></li>
<li><p><strong>训练</strong>：H100 上的 FP8 在质量损失极小的情况下提供 2<span class="arithmatex">\(\times\)</span> 吞吐。BF16 仍是较小模型的默认选项。</p></li>
<li><p><strong>边缘部署</strong>：在消费级硬件上做本地推理时使用 GGUF Q4_K_M。</p></li>
<li><p><strong>RLHF</strong>：将冻结模型（参考模型、奖励模型）量化为 INT8/FP8。策略模型保持 BF16 以确保训练精度。</p></li>
</ul>
</div>

## 1.14.2　剪枝（Pruning） {#pruning}

#### 为何要剪枝？ {#why-prune}

<p>现代 LLM 包含数十亿参数，而实证研究一致表明其中相当大一部分权重对模型输出贡献甚微。剪枝正是利用这种过参数化：通过移除冗余权重，可以降低<strong>内存占用</strong>（使其能部署于较小的 GPU 或边缘设备）、<strong>推理延迟</strong>（每次前向传递的乘加操作更少），以及<strong>服务成本</strong>（每美元的吞吐更高）。与对所有权重统一降低精度的量化不同，剪枝有选择地消除权重——与量化结合时能带来乘性的节省（例如 50% 稀疏、4-bit 的模型相比密集 BF16 基线节省 <span class="arithmatex">\(4\times\)</span> 内存）。挑战在于既要实现高稀疏度又不损害生成质量，这推动了无需重新训练的一次性（one-shot）原则性方法的发展。</p>

<div class="admonition note"><p class="admonition-title">剪枝方法</p>
<ul>
<li><p><strong>非结构化剪枝</strong>：将低于阈值的单个权重置零。可实现高稀疏度（50–90%）。需要稀疏 GEMM 内核（A100/H100 上的 2:4）。</p></li>
<li><p><strong>结构化剪枝</strong>：移除整个 attention 头、层或 FFN 神经元。无需专门内核即可直接降低 FLOPS。</p></li>
<li><p><strong>SparseGPT</strong> <a class="book-citation" id="cite-frantar2023sparsegpt-1" href="../../references.md#ref-frantar2023sparsegpt" aria-label="参考文献 98">[98]</a>：使用近似逆 Hessian 进行一次性剪枝。在 175B 模型上以极小的质量损失实现 50% 非结构化稀疏。</p></li>
<li><p><strong>Wanda</strong> <a class="book-citation" id="cite-sun2024wanda-1" href="../../references.md#ref-sun2024wanda" aria-label="参考文献 345">[345]</a>：按 <span class="arithmatex">\(|w| \times \|x\|\)</span>（权重幅值乘以输入激活范数）剪枝。无需校准数据，效果可与 SparseGPT 竞争。</p></li>
</ul>
</div>

<div class="admonition warning"><p class="admonition-title">NVIDIA 2:4 结构化稀疏</p>
<p>A100/H100 Tensor Core 原生支持 2:4 稀疏：每 4 个元素中至多 2 个非零。在硬件加速支持的操作上恰好带来 2<span class="arithmatex">\(\times\)</span> 加速。其约束是：必须<em>恰好</em>在这一特定模式下达到 50% 稀疏，相比任意稀疏度灵活性较低。</p>
</div>

## 1.14.3　知识蒸馏（Knowledge Distillation） {#knowledge-distillation}

<p>知识蒸馏 <a class="book-citation" id="cite-hinton2015distilling-1" href="../../references.md#ref-hinton2015distilling" aria-label="参考文献 139">[139]</a> 将一个大型<em>教师</em>模型已学到的行为迁移到一个更小、更便宜的<em>学生</em>模型中。核心思想是：教师在 token 上的输出分布所携带的信号远比单纯的硬标签丰富——揭示了类间相似性、置信度校准与不确定性，学生可以加以利用。</p>

#### 温度缩放（Temperature Scaling）的 Softmax。 {#temperature-scaled-softmax.}

<p>为了揭示教师 logits 中的 “暗知识”（dark knowledge），我们用温度 <span class="arithmatex">\(T &gt; 1\)</span> 对分布进行软化：</p>



<div class="arithmatex">\[p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}\]</div>

<p>温度较高时概率质量在更多 token 上分布开来，使接近正确的备选项变得可见。训练时学生使用相同温度；推理时学生使用 <span class="arithmatex">\(T=1\)</span>。</p>

#### 通用蒸馏损失。 {#general-distillation-loss.}



<div class="arithmatex">\[\mathcal{L}_{\text{distill}} = \alpha \, T^2 \cdot \text{KL}\!\bigl(P_{\text{teacher}}^{(T)} \;\|\; P_{\text{student}}^{(T)}\bigr) \;+\; (1-\alpha) \cdot \mathcal{L}_{\text{CE}}(y, P_{\text{student}}^{(1)})\]</div>

<p><span class="arithmatex">\(T^2\)</span> 因子补偿了软化分布造成的梯度幅度下降。典型取值：<span class="arithmatex">\(T \in [2, 20]\)</span>，<span class="arithmatex">\(\alpha \in [0.5, 0.9]\)</span>（教师质量越高，KL 的权重越大）。</p>

<p class="table-caption" id="table-1-18"><span class="table-number">表 1.18　</span>面向 LLM 的知识蒸馏范式。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>范式</strong></th><th scope="col"><strong>机制</strong></th><th scope="col"><strong>优点</strong></th><th scope="col"><strong>缺点</strong></th></tr></thead>
<tbody><tr><td><strong>离线 / 白盒（Offline / White-box）</strong></td><td>预先计算教师 logits；学生在完整分布上训练</td><td>完整分布信号；教师代价一次性付出</td><td>数据陈旧；存储开销大</td></tr>
<tr><td><strong>在线 / 协同训练（Online / Co-training）</strong></td><td>教师在线生成；学生看到新鲜 logits</td><td>适应学生弱点</td><td>算力 <span class="arithmatex">\(2\times\)</span>；基础设施复杂</td></tr>
<tr><td><strong>黑盒（API）</strong></td><td>仅有教师的<em>文本</em>输出（无 logits）</td><td>适用于专有模型</td><td>失去暗知识；近似于 SFT</td></tr>
<tr><td><strong>自蒸馏（Self-distillation）</strong></td><td>模型蒸馏到自身的较小版本</td><td>不需要单独的教师</td><td>教师=学生家族；存在上限</td></tr></tbody>
</table>
</div>

#### 离线（白盒）蒸馏。 {#offline-white-box-distillation.}

<p>为每个训练 token 记录教师完整的 logit 向量（或为节省存储而记录 top-<span class="arithmatex">\(k\)</span> logits）。学生最小化与这些存储分布的 KL 散度。在教师访问不受限时，这是最数据高效的范式。</p>

<p><strong>动机：</strong>将教师推理与学生训练解耦——在高端硬件上跑一次教师，之后廉价地训练多个学生。</p>

<p><strong>优点：</strong>确定性、可复现；教师代价被摊销；完整分布信号。<br><strong>缺点：</strong>需要为每个 token 存储 <span class="arithmatex">\(|V|\)</span> 维向量（可通过 top-<span class="arithmatex">\(k\)</span> 剪枝缓解）；教师无法针对学生的失误进行调整。</p>

#### 在线（协同训练）蒸馏。 {#online-co-training-distillation.}

<p>教师与学生联合运行：教师为学生当前的训练批次生成 logits。</p>

<p><strong>动机：</strong>让教师聚焦于学生当前难以处理的输入（类似课程学习）。</p>

<p><strong>优点：</strong>数据新鲜；可使用学生生成的输入进行 on-policy 蒸馏。<br><strong>缺点：</strong>GPU 代价翻倍；同步复杂度高；更难扩展。</p>

#### 黑盒（API）蒸馏。 {#black-box-api-distillation.}

<p>当仅有文本输出可用时（例如从专有 API 蒸馏），学生通过对教师生成结果做 SFT 进行训练，并可选地辅以思维链轨迹。</p>

<p><strong>动机：</strong>现实约束——大多数前沿模型不暴露 logits。</p>

<p><strong>优点：</strong>流程简单；适用于任何带 API 的模型。<br><strong>缺点：</strong>没有软标签信号；容易放大幻觉；本质上等同于监督微调。</p>

#### 自蒸馏。 {#self-distillation.}

<p>模型从同一架构家族中的较大版本（例如 Llama-3 70B <span class="arithmatex">\(\to\)</span> 8B）或训练过程中的自身 checkpoint 蒸馏。</p>

<p><strong>动机：</strong>避免训练单独的教师；在不同规模上利用模型自身的能力。</p>

<p><strong>优点：</strong>架构兼容；无外部依赖。<br><strong>缺点：</strong>教师天花板等于模型天花板；无法引入真正新的知识。</p>

<div class="admonition tip"><p class="admonition-title">暗知识（Dark Knowledge）</p>
<p>设想一个语言模型在预测 “The capital of France is” 之后的词。硬标签只告诉它 “Paris” 是正确答案。但教师的软分布可能把 5% 分给 “Lyon”、2% 分给 “Marseille”、对 “banana” 接近零——这告诉学生<em>哪些错误是合理的</em>，从而显著提升校准度与泛化能力。</p>
</div>

#### LLM 蒸馏的实践考量。 {#practical-considerations-for-llm-distillation.}

<ul>
<li><p><strong>序列级 vs. token 级：</strong>token 级 KL 是标准做法；序列级蒸馏（在完整序列上最小化 KL）能更好地捕捉长程一致性，但更难优化。</p></li>
<li><p><strong>逐层提示：</strong>匹配中间表示（attention 图、隐藏状态）能提供额外的学习信号——尤其在学生架构不同的情况下有用。</p></li>
<li><p><strong>数据选择：</strong>蒸馏数据质量很重要；精挑多样化的困难样本比随机采样能产出更好的学生。</p></li>
<li><p><strong>学生容量：</strong>学生参数低于教师约 10% 时收益递减；在极端压缩下可能需要架构变更（例如 MoE <span class="arithmatex">\(\to\)</span> dense）。</p></li>
<li><p><strong>与量化结合：</strong>蒸馏 + 4-bit 量化（例如 QLoRA 蒸馏模型）能在 <span class="arithmatex">\(20\times\)</span> 压缩下达到接近教师的质量。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">压缩方法对比 —— 70B 模型</p>
<span id="table-1-inline-27"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>大小</strong></th><th scope="col"><strong>速度</strong></th><th scope="col"><strong>质量</strong></th><th scope="col"><strong>使用场景</strong></th></tr></thead>
<tbody><tr><td>BF16（基线）</td><td>140 GB</td><td>1<span class="arithmatex">\(\times\)</span></td><td>100%</td><td>训练、参考</td></tr>
<tr><td>FP8 (E4M3)</td><td>70 GB</td><td>2<span class="arithmatex">\(\times\)</span></td><td>99.5%</td><td>H100 推理</td></tr>
<tr><td>INT8 (SmoothQuant)</td><td>70 GB</td><td>1.8<span class="arithmatex">\(\times\)</span></td><td>99%</td><td>A100 推理</td></tr>
<tr><td>4-bit (AWQ)</td><td>35 GB</td><td>2.5<span class="arithmatex">\(\times\)</span></td><td>97–98%</td><td>大规模服务</td></tr>
<tr><td>2-bit (AQLM)</td><td>17.5 GB</td><td>3<span class="arithmatex">\(\times\)</span></td><td>90–93%</td><td>边缘、实验</td></tr>
<tr><td>Pruned 50% (2:4)</td><td>70 GB</td><td>1.8<span class="arithmatex">\(\times\)</span></td><td>97%</td><td>结构化加速</td></tr>
<tr><td>Distilled 8B</td><td>16 GB</td><td>10<span class="arithmatex">\(\times\)</span></td><td>80–85%</td><td>移动、边缘</td></tr></tbody>
</table>
</div>
</div>
