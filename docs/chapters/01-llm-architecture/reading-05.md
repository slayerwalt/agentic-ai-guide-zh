# 第 1 章 · 压缩、加速与安全 {#reading-reading-05}

## 1.14　模型压缩方法 {#model-compression-methods}

<p>模型压缩在保持质量的同时降低模型规模和推理成本。三种主要方法：量化（Quantization，降低数值精度）、剪枝（Pruning，移除参数）和蒸馏（Distillation，训练较小模型模仿更大模型）。</p>

### 1.14.1　量化（Quantization） {#quantization}

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

### 1.14.2　剪枝（Pruning） {#pruning}

##### 为何要剪枝？ {#why-prune}

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

### 1.14.3　知识蒸馏（Knowledge Distillation） {#knowledge-distillation}

<p>知识蒸馏 <a class="book-citation" id="cite-hinton2015distilling-1" href="../../references.md#ref-hinton2015distilling" aria-label="参考文献 139">[139]</a> 将一个大型<em>教师</em>模型已学到的行为迁移到一个更小、更便宜的<em>学生</em>模型中。核心思想是：教师在 token 上的输出分布所携带的信号远比单纯的硬标签丰富——揭示了类间相似性、置信度校准与不确定性，学生可以加以利用。</p>

##### 温度缩放（Temperature Scaling）的 Softmax。 {#temperature-scaled-softmax.}

<p>为了揭示教师 logits 中的 “暗知识”（dark knowledge），我们用温度 <span class="arithmatex">\(T &gt; 1\)</span> 对分布进行软化：</p>



<div class="arithmatex">\[p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}\]</div>

<p>温度较高时概率质量在更多 token 上分布开来，使接近正确的备选项变得可见。训练时学生使用相同温度；推理时学生使用 <span class="arithmatex">\(T=1\)</span>。</p>

##### 通用蒸馏损失。 {#general-distillation-loss.}



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

##### 离线（白盒）蒸馏。 {#offline-white-box-distillation.}

<p>为每个训练 token 记录教师完整的 logit 向量（或为节省存储而记录 top-<span class="arithmatex">\(k\)</span> logits）。学生最小化与这些存储分布的 KL 散度。在教师访问不受限时，这是最数据高效的范式。</p>

<p><strong>动机：</strong>将教师推理与学生训练解耦——在高端硬件上跑一次教师，之后廉价地训练多个学生。</p>

<p><strong>优点：</strong>确定性、可复现；教师代价被摊销；完整分布信号。<br><strong>缺点：</strong>需要为每个 token 存储 <span class="arithmatex">\(|V|\)</span> 维向量（可通过 top-<span class="arithmatex">\(k\)</span> 剪枝缓解）；教师无法针对学生的失误进行调整。</p>

##### 在线（协同训练）蒸馏。 {#online-co-training-distillation.}

<p>教师与学生联合运行：教师为学生当前的训练批次生成 logits。</p>

<p><strong>动机：</strong>让教师聚焦于学生当前难以处理的输入（类似课程学习）。</p>

<p><strong>优点：</strong>数据新鲜；可使用学生生成的输入进行 on-policy 蒸馏。<br><strong>缺点：</strong>GPU 代价翻倍；同步复杂度高；更难扩展。</p>

##### 黑盒（API）蒸馏。 {#black-box-api-distillation.}

<p>当仅有文本输出可用时（例如从专有 API 蒸馏），学生通过对教师生成结果做 SFT 进行训练，并可选地辅以思维链轨迹。</p>

<p><strong>动机：</strong>现实约束——大多数前沿模型不暴露 logits。</p>

<p><strong>优点：</strong>流程简单；适用于任何带 API 的模型。<br><strong>缺点：</strong>没有软标签信号；容易放大幻觉；本质上等同于监督微调。</p>

##### 自蒸馏。 {#self-distillation.}

<p>模型从同一架构家族中的较大版本（例如 Llama-3 70B <span class="arithmatex">\(\to\)</span> 8B）或训练过程中的自身 checkpoint 蒸馏。</p>

<p><strong>动机：</strong>避免训练单独的教师；在不同规模上利用模型自身的能力。</p>

<p><strong>优点：</strong>架构兼容；无外部依赖。<br><strong>缺点：</strong>教师天花板等于模型天花板；无法引入真正新的知识。</p>

<div class="admonition tip"><p class="admonition-title">暗知识（Dark Knowledge）</p>
<p>设想一个语言模型在预测 “The capital of France is” 之后的词。硬标签只告诉它 “Paris” 是正确答案。但教师的软分布可能把 5% 分给 “Lyon”、2% 分给 “Marseille”、对 “banana” 接近零——这告诉学生<em>哪些错误是合理的</em>，从而显著提升校准度与泛化能力。</p>
</div>

##### LLM 蒸馏的实践考量。 {#practical-considerations-for-llm-distillation.}

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

## 1.15　投机解码（Speculative Decoding）方法 {#speculative-decoding-methods}

<p>投机解码 <a class="book-citation" id="cite-leviathan2023fast-1" href="../../references.md#ref-leviathan2023fast" aria-label="参考文献 205">[205]</a> 通过同时预测多个 token，然后在目标模型的一次前向传递中对其进行验证，来加速自回归生成。它产生与标准解码<strong>完全相同的输出分布</strong>（无质量损失），同时实现 2–3<span class="arithmatex">\(\times\)</span> 加速。</p>

### 1.15.1　核心原理 {#core-principle}

<div class="admonition note"><p class="admonition-title">投机解码框架</p>
<ol>
<li><p>一个快速的<strong>草稿（draft）机制</strong>提议 <span class="arithmatex">\(k\)</span> 个候选 token：<span class="arithmatex">\(\hat{x}_1, \ldots, \hat{x}_k\)</span></p></li>
<li><p>大的<strong>目标模型</strong>对所有 <span class="arithmatex">\(k\)</span> 个 token 进行一次（批量）前向传递</p></li>
<li><p><strong>验证</strong>：从左到右依次接受 token，只要 <span class="arithmatex">\(P_{\text{target}}(\hat{x}_i) \geq r_i \cdot P_{\text{draft}}(\hat{x}_i)\)</span>（其中 <span class="arithmatex">\(r_i \sim U[0,1]\)</span>）</p></li>
<li><p>在位置 <span class="arithmatex">\(j\)</span> 首次被拒绝时：从调整后的分布重新采样 <span class="arithmatex">\(x_j\)</span>，并丢弃 <span class="arithmatex">\(\hat{x}_{j+1}, \ldots, \hat{x}_k\)</span></p></li>
</ol>

<p><strong>关键性质</strong>：这一接受/拒绝方案保证最终分布与 <span class="arithmatex">\(P_{\text{target}}\)</span> 完全一致。</p>

<p><strong>加速比</strong>：若接受率为 <span class="arithmatex">\(\alpha\)</span>，每步期望 token 数 <span class="arithmatex">\(= \frac{1 - \alpha^{k+1}}{1 - \alpha}\)</span>。在 <span class="arithmatex">\(\alpha=0.8\)</span>、<span class="arithmatex">\(k=5\)</span> 时：每步期望 3.4 个 token，相较标准解码的 1 个。</p>
</div>

### 1.15.2　方法对比 {#methods-comparison}

<p class="table-caption" id="table-1-19"><span class="table-number">表 1.19　</span>现代推理引擎支持的投机解码方法。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>草稿来源</strong></th><th scope="col"><strong>加速比</strong></th><th scope="col"><strong>核心思想</strong></th></tr></thead>
<tbody><tr><td><strong>标准方案</strong> <a class="book-citation" id="cite-leviathan2023fast-2" href="../../references.md#ref-leviathan2023fast" aria-label="参考文献 205">[205]</a></td><td>小模型（1–7B）</td><td>2–3<span class="arithmatex">\(\times\)</span></td><td>独立的草稿模型生成候选。简单但需要同时加载 2 个模型。</td></tr>
<tr><td><strong>Medusa</strong> <a class="book-citation" id="cite-cai2024medusa-1" href="../../references.md#ref-cai2024medusa" aria-label="参考文献 35">[35]</a></td><td>并行 LM 头</td><td>2–3<span class="arithmatex">\(\times\)</span></td><td>在目标模型上增加 <span class="arithmatex">\(k\)</span> 个额外的预测头。每个分别预测 <span class="arithmatex">\(+1, +2, \ldots, +k\)</span> 位置的 token。</td></tr>
<tr><td><strong>Eagle</strong> <a class="book-citation" id="cite-li2024eagle-1" href="../../references.md#ref-li2024eagle" aria-label="参考文献 212">[212]</a></td><td>特征层级</td><td>2.5–3.5<span class="arithmatex">\(\times\)</span></td><td>轻量解码器根据目标模型的隐藏状态生成草稿 token。接受率高于 Medusa。</td></tr>
<tr><td><strong>Eagle-2</strong> <a class="book-citation" id="cite-li2024eagle-2" href="../../references.md#ref-li2024eagle" aria-label="参考文献 212">[212]</a></td><td>上下文感知</td><td>3–4<span class="arithmatex">\(\times\)</span></td><td>基于置信度扩展的动态草稿树。当前最先进的接受率。</td></tr>
<tr><td><strong>N-gram Lookup</strong></td><td>N-gram 缓存</td><td>1.5–2<span class="arithmatex">\(\times\)</span></td><td>将 Prompt 的 n-gram 与已生成文本进行匹配。零成本；对重复输出极佳。</td></tr>
<tr><td><strong>Lookahead</strong> <a class="book-citation" id="cite-fu2024lookahead-1" href="../../references.md#ref-fu2024lookahead" aria-label="参考文献 101">[101]</a></td><td>Jacobi 迭代</td><td>2–2.5<span class="arithmatex">\(\times\)</span></td><td>并行 Jacobi 解码配合 n-gram 验证。无需草稿模型；使用目标模型自身。</td></tr>
<tr><td><strong>多 token 预测</strong> <a class="book-citation" id="cite-gloeckle2024multi-1" href="../../references.md#ref-gloeckle2024multi" aria-label="参考文献 111">[111]</a></td><td>修改架构</td><td>2–3<span class="arithmatex">\(\times\)</span></td><td>训练模型原生地在每步预测多个 token（Meta 在 Llama 中的方案）。</td></tr></tbody>
</table>
</div>

### 1.15.3　Medusa 多头推测解码 {#medusa-multi-head-speculative-decoding}

<div class="admonition note"><p class="admonition-title">Medusa 工作原理</p>
<p>Medusa 在 LLM 上额外增加 <span class="arithmatex">\(k\)</span> 个 “预测头”（共享同一主干）：</p>

<ul>
<li><p>Head 0（原始）：预测位置 <span class="arithmatex">\(t+1\)</span> 的 token（标准的下一个 token）</p></li>
<li><p>Head 1：预测位置 <span class="arithmatex">\(t+2\)</span> 的 token（跳过一个）</p></li>
<li><p>Head <span class="arithmatex">\(i\)</span>：预测位置 <span class="arithmatex">\(t+i+1\)</span> 的 token</p></li>
<li><p>所有头在一次前向传递中并行运行</p></li>
<li><p>一种<strong>树状结构</strong>的验证同时校验多条候选序列</p></li>
</ul>

<p><strong>训练</strong>：只微调 Medusa 头（主干冻结）。成本：在代表性数据上约 1 个 epoch。</p>

<p><strong>优势</strong>：无需独立的草稿模型；每个头都很小（一层线性层）。内存开销：<span class="arithmatex">\(&lt;\)</span>1%。</p>
</div>

### 1.15.4　Eagle：特征级草稿 {#eagle-feature-level-drafting}

<div class="admonition tip"><p class="admonition-title">为何 Eagle 优于 Medusa</p>
<p>Medusa 的各个头在不同位置独立预测——它们无法以自己先前的预测为条件（<span class="arithmatex">\(t+2\)</span> 位置的 token 不知道 <span class="arithmatex">\(t+1\)</span> 预测了什么）。Eagle 通过一个作用在目标模型隐藏状态上的轻量自回归解码器修正了这一点：</p>

<ol>
<li><p>从目标模型最后一层提取隐藏状态</p></li>
<li><p>输入一个小型（1 层）解码器，以先前隐藏状态为条件自回归地生成草稿 token</p></li>
<li><p>这捕捉到了 Medusa 所缺失的 token 间依赖</p></li>
</ol>

<p>结果：Eagle 实现 85–95% 的接受率，而 Medusa 为 60–80%。</p>
</div>

### 1.15.5　N-gram 投机解码 {#n-gram-speculative-decoding}

<div class="admonition note"><p class="admonition-title">N-gram 查找方法</p>
<p>最简单的投机解码——无需额外模型或训练：</p>

<ol>
<li><p>维护一份来自 Prompt 与已生成文本的 n-gram 缓存</p></li>
<li><p>每一步检查当前上下文的最后 <span class="arithmatex">\(n-1\)</span> 个 token 是否匹配缓存中的任一 n-gram</p></li>
<li><p>若匹配：将其后继作为草稿 token 提议</p></li>
<li><p>像往常一样在目标模型上进行验证</p></li>
</ol>

<p><strong>最佳场景</strong>：代码生成（重复模式）、结构化输出（JSON/XML），以及带有重复元素的 Prompt。<strong>成本</strong>：基本为零。</p>
</div>

### 1.15.6　与 vLLM 的集成 {#integration-with-vllm}

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">vllm</span><span class="w"> </span><span class="kn">import</span> <span class="n">LLM</span><span class="p">,</span> <span class="n">SamplingParams</span>

<span class="c1"># 标准投机解码（独立的草稿模型）</span>
<span class="n">llm</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B"</span><span class="p">,</span>
    <span class="n">tensor_parallel_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">speculative_config</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"model"</span><span class="p">:</span> <span class="s2">"meta-llama/Llama-3-8B"</span><span class="p">,</span>
        <span class="s2">"num_speculative_tokens"</span><span class="p">:</span> <span class="mi">5</span><span class="p">,</span>
    <span class="p">},</span>
<span class="p">)</span>

<span class="c1"># N-gram 投机（零成本，无需草稿模型）</span>
<span class="n">llm</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B"</span><span class="p">,</span>
    <span class="n">speculative_config</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"ngram"</span><span class="p">,</span>
        <span class="s2">"num_speculative_tokens"</span><span class="p">:</span> <span class="mi">5</span><span class="p">,</span>
        <span class="s2">"prompt_lookup_max"</span><span class="p">:</span> <span class="mi">4</span><span class="p">,</span>  <span class="c1"># 从 Prompt 中匹配最长 4-gram</span>
    <span class="p">},</span>
<span class="p">)</span>

<span class="c1"># EAGLE 风格（特征级草稿，高接受率）</span>
<span class="n">llm</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Meta-Llama-3-8B-Instruct"</span><span class="p">,</span>
    <span class="n">tensor_parallel_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">speculative_config</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"model"</span><span class="p">:</span> <span class="s2">"yuhuili/EAGLE-LLaMA3-Instruct-8B"</span><span class="p">,</span>
        <span class="s2">"num_speculative_tokens"</span><span class="p">:</span> <span class="mi">2</span><span class="p">,</span>
        <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"eagle"</span><span class="p">,</span>
        <span class="s2">"draft_tensor_parallel_size"</span><span class="p">:</span> <span class="mi">1</span><span class="p">,</span>
    <span class="p">},</span>
<span class="p">)</span>

<span class="c1"># MLP 推测器（IBM 风格，轻量预测头）</span>
<span class="n">llm</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Meta-Llama-3.1-70B-Instruct"</span><span class="p">,</span>
    <span class="n">tensor_parallel_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">speculative_config</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"model"</span><span class="p">:</span> <span class="s2">"ibm-ai-platform/llama3-70b-accelerator"</span><span class="p">,</span>
        <span class="s2">"draft_tensor_parallel_size"</span><span class="p">:</span> <span class="mi">1</span><span class="p">,</span>
    <span class="p">},</span>
<span class="p">)</span>
</code></pre></div>

<div class="admonition warning"><p class="admonition-title">何时<em>不</em>该使用投机解码</p>
<ul>
<li><p><strong>高批量</strong>：当 batch <span class="arithmatex">\(\geq 64\)</span> 时，生成本身已经计算密集。投机解码引入的开销（草稿生成 + 验证）得不偿失。</p></li>
<li><p><strong>分布差异过大</strong>：若草稿模型与目标模型差异过大，接受率会跌至 50% 以下，反而比标准解码更慢。</p></li>
<li><p><strong>短输出</strong>：对少于 20 个 token 的输出，投机解码的启动开销超过收益。</p></li>
<li><p><strong>经验法则</strong>：投机解码对延迟敏感的单流生成（聊天机器人、交互式代码补全）帮助最大。</p></li>
</ul>
</div>

## 1.16　幻觉检测 {#sec:hallucination}

<p>LLM 会生成流畅但可能事实错误的文本——这种现象称为<strong>幻觉</strong>（hallucination） <a class="book-citation" id="cite-ji2023hallucination-1" href="../../references.md#ref-ji2023hallucination" aria-label="参考文献 164">[164]</a>。本节介绍在模型层面（不依赖外部检索或多智能体校验）的基本检测方法。</p>

### 1.16.1　幻觉的类型 {#types-of-hallucination}

<div class="admonition note"><p class="admonition-title">幻觉分类法</p>
<ul>
<li><p><strong>内在型（Intrinsic）</strong>：与所提供的输入/上下文矛盾（例如摘要与原文相反）</p></li>
<li><p><strong>外在型（Extrinsic）</strong>：生成无法从输入验证的陈述，且事实上错误</p></li>
<li><p><strong>忠实性（Faithfulness）</strong>：输出偏离指令或所指定的约束</p></li>
</ul>
</div>

### 1.16.2　检测方法（模型层级） {#detection-methods-model-level}

<p class="table-caption" id="table-1-20"><span class="table-number">表 1.20　</span>在模型层面运作的基础幻觉检测方法。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>机制</strong></th><th scope="col"><strong>信号</strong></th></tr></thead>
<tbody><tr><td>Token 级熵</td><td>生成时的高熵表示不确定 <a class="book-citation" id="cite-kadavath2022language-1" href="../../references.md#ref-kadavath2022language" aria-label="参考文献 175">[175]</a></td><td><span class="arithmatex">\(H(P(x_t)) &gt; \tau\)</span></td></tr>
<tr><td>序列对数概率</td><td>输出的平均对数概率较低提示存在虚构</td><td><span class="arithmatex">\(\frac{1}{T}\sum_t \log P(x_t)\)</span></td></tr>
<tr><td>一致性采样</td><td>生成 <span class="arithmatex">\(N\)</span> 个回复；一致性低 <span class="arithmatex">\(=\)</span> 可能幻觉 <a class="book-citation" id="cite-manakul2023selfcheckgpt-1" href="../../references.md#ref-manakul2023selfcheckgpt" aria-label="参考文献 249">[249]</a></td><td>矛盾率</td></tr>
<tr><td>语义熵（Semantic Entropy）</td><td>对语义（而非字符串）聚类；语义熵高 <span class="arithmatex">\(=\)</span> 不确定 <a class="book-citation" id="cite-kuhn2023semantic-1" href="../../references.md#ref-kuhn2023semantic" aria-label="参考文献 193">[193]</a></td><td>聚类多样性</td></tr>
<tr><td>DoLA</td><td>对比后层与前层的 logits；放大事实知识 <a class="book-citation" id="cite-chuang2024dola-1" href="../../references.md#ref-chuang2024dola" aria-label="参考文献 58">[58]</a></td><td>层间差异</td></tr></tbody>
</table>
</div>

##### 语义熵（Semantic Entropy）。 {#semantic-entropy.}

<p>Kuhn 等人 <a class="book-citation" id="cite-kuhn2023semantic-2" href="../../references.md#ref-kuhn2023semantic" aria-label="参考文献 193">[193]</a> 观察到 token 级熵并不可靠（同义改写包含不同 token 但意义相同）。他们改为生成多个回复，按语义等价（通过 NLI）聚类，并在语义聚类上计算熵：</p>



<div class="arithmatex">\[SE = -\sum_{c \in \text{clusters}} P(c) \log P(c)\]</div>

<p>高 SE 意味着模型产生<em>语义不同</em>的答案——这是强烈的幻觉信号。</p>

##### SelfCheckGPT。 {#selfcheckgpt.}

<p>Manakul 等人 <a class="book-citation" id="cite-manakul2023selfcheckgpt-2" href="../../references.md#ref-manakul2023selfcheckgpt" aria-label="参考文献 249">[249]</a> 通过检查自一致性来检测幻觉：生成多个回复并验证主回复中的陈述是否被其他回复支持。若模型 “自相矛盾”，则该陈述很可能是幻觉。无需外部知识。</p>

##### 按层对比解码（Decoding by Contrasting Layers，DoLA）。 {#dola-decoding-by-contrasting-layers.}

<p>Chuang 等人 <a class="book-citation" id="cite-chuang2024dola-2" href="../../references.md#ref-chuang2024dola" aria-label="参考文献 58">[58]</a> 观察到事实知识在 Transformer 较深的层中浮现，而较早的层保留更多通用/不确定的表示。DoLA 在每个解码步上对比较深（“成熟”）层与较早（“未成熟”）层的 logits 分布：</p>



<div class="arithmatex">\[\text{DoLA}(x_t) = \text{softmax}\!\bigl(\log P_{\text{late}}(x_t) - \log P_{\text{early}}(x_t)\bigr)\]</div>

<p>通过放大深层中事实知识所编码的信号，DoLA 在推理时降低幻觉，<em>无需重新训练</em>——只需对对比层进行一次额外的前向传递。它与基于采样的方法互补，可以与之结合使用。</p>

<div class="admonition warning"><p class="admonition-title">模型层级检测的局限</p>
<p>这些方法检测的是<em>不确定性</em>，而非<em>错误性</em>。模型可能自信地犯错（低熵、一致的回复——但事实上错误）。要实现可靠检测，应结合基于检索的验证（RAG）或外部事实核查工具。</p>
</div>

## 1.17　LLM 安全与负责任 AI {#sec:safety}

<p>安全并非事后补救——它是 LLM 训练流水线不可或缺的一部分。本节涵盖 LLM 安全的关键维度以及用于强制负责任行为的机制。</p>

### 1.17.1　威胁分类 {#threat-taxonomy}

<p class="table-caption" id="table-1-21"><span class="table-number">表 1.21　</span>LLM 安全威胁类别。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>类别</strong></th><th scope="col"><strong>描述与示例</strong></th></tr></thead>
<tbody><tr><td><strong>有害内容</strong></td><td>生成有毒、暴力或非法的指令（生物武器、CSAM）</td></tr>
<tr><td><strong>偏见与歧视</strong></td><td>延续刻板印象；在不同人口群体间的不公正对待 <a class="book-citation" id="cite-gallegos2024bias-1" href="../../references.md#ref-gallegos2024bias" aria-label="参考文献 104">[104]</a></td></tr>
<tr><td><strong>隐私侵犯</strong></td><td>泄露训练数据中的 PII；记忆攻击 <a class="book-citation" id="cite-carlini2021extracting-1" href="../../references.md#ref-carlini2021extracting" aria-label="参考文献 39">[39]</a></td></tr>
<tr><td><strong>越狱（Jailbreaking）</strong></td><td>绕过安全护栏的对抗性 Prompt <a class="book-citation" id="cite-zou2023universal-1" href="../../references.md#ref-zou2023universal" aria-label="参考文献 443">[443]</a></td></tr>
<tr><td><strong>虚假信息</strong></td><td>生成令人信服但虚假的陈述（规模化的幻觉）</td></tr>
<tr><td><strong>双重用途</strong></td><td>合法能力（编程、化学）被武器化用于伤害</td></tr></tbody>
</table>
</div>

### 1.17.2　安全训练流水线 {#safety-training-pipeline}

<figure id="figure-1-14"><a href="../../assets/figures/fig_015_fig15.png" target="_blank" rel="noopener" aria-label="查看图 1.14 原图" title="点击查看原图"><img src="../../assets/figures/fig_015_fig15.png" alt="安全贯穿每个阶段：预训练中的数据过滤、SFT 中的拒答示例、RLHF 中专门的安全奖励模型，以及迭代式红队测试。" loading="lazy"></a><figcaption><span class="figure-number">图 1.14　</span>安全贯穿每个阶段：预训练中的数据过滤、SFT 中的拒答示例、RLHF 中专门的安全奖励模型，以及迭代式红队测试。</figcaption></figure>

### 1.17.3　关键安全机制 {#key-safety-mechanisms}

<div class="admonition note"><p class="admonition-title">安全技术</p>
<ul>
<li><p><strong>数据过滤</strong>：从预训练语料中剔除有毒、有偏见以及含 PII 的文本</p></li>
<li><p><strong>安全 SFT</strong>：在恰当拒答的示例上训练（“我无法帮你做这件事，因为……”）</p></li>
<li><p><strong>宪法 AI（Constitutional AI）</strong> <a class="book-citation" id="cite-bai2022constitutional-1" href="../../references.md#ref-bai2022constitutional" aria-label="参考文献 16">[16]</a>：基于原则进行自我批评（Self-Critique）；模型依照一部规则宪法修订自身输出</p></li>
<li><p><strong>安全奖励模型</strong>：基于安全标注配对训练独立的 RM；在 RLHF 中通过加权和与 helpfulness RM 结合</p></li>
<li><p><strong>护栏（Guardrails）</strong>：在服务时拦截有害请求/回复的输入/输出分类器</p></li>
<li><p><strong>红队测试（Red teaming）</strong> <a class="book-citation" id="cite-perez2022red-1" href="../../references.md#ref-perez2022red" aria-label="参考文献 290">[290]</a>：系统性的对抗评估，在部署前发现失效模式</p></li>
</ul>
</div>

### 1.17.4　有用性与安全性的权衡 {#the-helpfulnesssafety-tradeoff}

<div class="admonition tip"><p class="admonition-title">在有用性与安全性之间平衡</p>
<p>对安全的过度优化会带来<em>过度拒答</em>问题：模型拒绝无害的请求（例如在教育语境中拒绝讨论历史暴力）。目标是一个帕累托最优策略，在安全约束<em>之内</em>最大化有用性：</p>



<div class="arithmatex">\[\max_\theta \; \mathbb{E}[R_\text{helpful}] \quad \text{subject to} \quad \mathbb{E}[R_\text{safety}] \geq \tau\]</div>

<p>在实践中，这通过加权奖励实现：<span class="arithmatex">\(R = \alpha R_\text{helpful} + (1-\alpha) R_\text{safety}\)</span>，并仔细调整 <span class="arithmatex">\(\alpha\)</span>（通常 0.6–0.8）。Meta 的 Llama-3 报告称使用独立的安全与有用性奖励模型，并采用基于 margin 的加权 <a class="book-citation" id="cite-grattafiori2024llama3-1" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>。</p>
</div>

### 1.17.5　评测 {#evaluation}

<ul>
<li><p><strong>安全基准</strong>：ToxiGen、RealToxicityPrompts、BBQ（偏见）、CrowS-Pairs</p></li>
<li><p><strong>越狱鲁棒性</strong>：GCG 攻击 <a class="book-citation" id="cite-zou2023universal-2" href="../../references.md#ref-zou2023universal" aria-label="参考文献 443">[443]</a>、多轮越狱、编码型 Prompt</p></li>
<li><p><strong>过度拒答率</strong>：度量无害 Prompt 上的假阳性拒答（目标 <span class="arithmatex">\(&lt;\)</span>5%）</p></li>
<li><p><strong>红队评估</strong>：由领域专家（生物安全、网络安全）执行的人类对抗性测试</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">安全永无完结</p>
<p>任何技术组合都无法提供绝对的安全。新的攻击向量不断被发现（多模态越狱、剥除安全训练的微调攻击、many-shot prompting）。安全需要持续监控、对新威胁的快速响应，以及纵深防御（多层独立防线）。</p>
</div>
