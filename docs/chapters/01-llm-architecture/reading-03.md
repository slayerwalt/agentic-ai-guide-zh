# 第 1 章 · 高效模型与训练多样性 {#reading-reading-03}

## 1.9　LoRA 与参数高效微调 {#lora-and-parameter-efficient-fine-tuning}

<p>对 70B 模型进行全量微调需要存储 70B 可训练参数及其优化器状态（560+ GB 内存）。低秩适配（Low-Rank Adaptation，LoRA） <a class="book-citation" id="cite-hu2021lora-1" href="../../references.md#ref-hu2021lora" aria-label="参考文献 146">[146]</a>提供了一种仅用 <span class="arithmatex">\(&lt;\)</span>1% 参数即可微调、并达到相当质量的方法。</p>

### 1.9.1　LoRA 的核心洞察 {#the-lora-insight}

<div class="admonition note"><p class="admonition-title">LoRA 核心思想</p>
<p>不更新完整的权重矩阵 <span class="arithmatex">\(W \in \mathbb{R}^{d \times d}\)</span>，而是学习一个低秩扰动：</p>



<div class="arithmatex">\[W&#x27; = W + \frac{\alpha}{r} \cdot BA, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times d}\]</div>

<ul>
<li><p><span class="arithmatex">\(W\)</span> 被<strong>冻结</strong>（无梯度，无优化器状态）</p></li>
<li><p>只训练 <span class="arithmatex">\(B\)</span> 和 <span class="arithmatex">\(A\)</span>：<span class="arithmatex">\(2 \times d \times r\)</span> 参数而非 <span class="arithmatex">\(d^2\)</span></p></li>
<li><p>在秩 <span class="arithmatex">\(r=16\)</span>、<span class="arithmatex">\(d=4096\)</span> 时：LoRA 每层新增 <span class="arithmatex">\(2 \times 4096 \times 16 = 131K\)</span> 参数，而完整矩阵为 <span class="arithmatex">\(16.8M\)</span></p></li>
<li><p><span class="arithmatex">\(\alpha/r\)</span> 缩放因子控制更新的幅度</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">低秩为何有效</p>
<p>Aghajanyan 等 <a class="book-citation" id="cite-aghajanyan2020intrinsic-1" href="../../references.md#ref-aghajanyan2020intrinsic" aria-label="参考文献 3">[3]</a>表明，微调发生在一个非常低维的子空间——微调任务的「内在维度（intrinsic dimensionality）」远小于模型的参数量。一个 175B 模型的微调任务内在维度可能 <span class="arithmatex">\(&lt;\)</span>10,000。LoRA 直接利用了这一点：秩 <span class="arithmatex">\(r\)</span> 把每个权重矩阵的更新约束在 <span class="arithmatex">\(r\)</span> 维子空间内。</p>
</div>

<figure id="fig:lora-decomposition"><a href="../../assets/figures/fig_011_lora-decomposition.png" target="_blank" rel="noopener" aria-label="查看图 1.10 原图" title="点击查看原图"><img src="../../assets/figures/fig_011_lora-decomposition.png" alt="LoRA 将权重更新 \(\Delta W\) 分解为两个小矩阵 \(B \times A\)。原始权重 \(W\) 保持冻结；只有 \(B\) 和 \(A\) 接收梯度。推理时，乘积 \(BA\) 可以零开销地合并到 \(W\) 中。" loading="lazy"></a><figcaption><span class="figure-number">图 1.10　</span>LoRA 将权重更新 <span class="arithmatex">\(\Delta W\)</span> 分解为两个小矩阵 <span class="arithmatex">\(B \times A\)</span>。原始权重 <span class="arithmatex">\(W\)</span> 保持冻结；只有 <span class="arithmatex">\(B\)</span> 和 <span class="arithmatex">\(A\)</span> 接收梯度。推理时，乘积 <span class="arithmatex">\(BA\)</span> 可以零开销地合并到 <span class="arithmatex">\(W\)</span> 中。</figcaption></figure>

<div class="admonition tip"><p class="admonition-title">为什么 <span class="arithmatex">\(\alpha/r\)</span> 缩放很重要</p>
<p>如果不做缩放，将秩 <span class="arithmatex">\(r\)</span> 翻倍大约会让 <span class="arithmatex">\(\Delta W = BA\)</span> 的幅度翻倍（<span class="arithmatex">\(B\)</span> 中更多列参与求和）。这意味着改变秩也会改变模型被扰动的幅度——你每次调整 <span class="arithmatex">\(r\)</span> 都得重新调学习率。</p>

<p><span class="arithmatex">\(\alpha/r\)</span> 因子<strong>对更新幅度进行归一化</strong>，使其无论秩为多少都大致保持恒定：</p>



<div class="arithmatex">\[W&#x27; = W + \frac{\alpha}{r} \cdot BA\]</div>

<ul>
<li><p><strong>固定 <span class="arithmatex">\(\alpha\)</span>，扫描 <span class="arithmatex">\(r\)</span></strong>：有效更新幅度无论秩为何都保持在 <span class="arithmatex">\(\sim\alpha\)</span> 附近。可以尝试 <span class="arithmatex">\(r \in \{8, 16, 32, 64\}\)</span> 而无需重新调学习率。</p></li>
<li><p><strong>常见做法</strong>：设 <span class="arithmatex">\(\alpha = r\)</span>（即 <span class="arithmatex">\(\alpha/r = 1\)</span>）或 <span class="arithmatex">\(\alpha = 2r\)</span>（即 <span class="arithmatex">\(\alpha/r = 2\)</span>）。这是缩放因子为小整数的便捷默认。</p></li>
<li><p><strong>为什么不直接调 LR？</strong> 可以，但 <span class="arithmatex">\(\alpha/r\)</span> 提供了一个<em>与秩无关</em>的旋钮。团队可以在不同秩的实验间共享 LR 配方。</p></li>
<li><p><strong>rsLoRA 洞察</strong> <a class="book-citation" id="cite-kalajdzievski2023rslora-1" href="../../references.md#ref-kalajdzievski2023rslora" aria-label="参考文献 176">[176]</a>：在高秩（<span class="arithmatex">\(r \geq 64\)</span>）下，经验证据表明 <span class="arithmatex">\(\alpha/\sqrt{r}\)</span> 比 <span class="arithmatex">\(\alpha/r\)</span> 更稳定，因为 <span class="arithmatex">\(BA\)</span> 的方差按 <span class="arithmatex">\(\sqrt{r}\)</span> 而非 <span class="arithmatex">\(r\)</span> 缩放。</p></li>
</ul>
</div>

### 1.9.2　LoRA 超参数 {#lora-hyperparameters}

<p>正确选择 LoRA 超参数至关重要——错误的秩或 alpha 要么欠拟合（约束过强），要么浪费内存（表达力过剩）。</p>

<p class="table-caption" id="table-1-12"><span class="table-number">表 1.12　</span>LoRA 超参数指南。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>超参数</strong></th><th scope="col"><strong>典型值</strong></th><th scope="col"><strong>建议</strong></th></tr></thead>
<tbody><tr><td><code>r</code>（秩）</td><td>8, 16, 32, 64</td><td>更高 = 更大容量但更多内存。从 16 开始。</td></tr>
<tr><td><code>lora_alpha</code></td><td>16, 32（通常 <span class="arithmatex">\(= r\)</span> 或 <span class="arithmatex">\(2r\)</span>）</td><td>通过 <span class="arithmatex">\(\alpha/r\)</span> 缩放控制更新幅度。</td></tr>
<tr><td><code>target_modules</code></td><td><code>q_proj, k_proj, v_proj, o_proj</code></td><td>所有 Attention 投影。添加 <code>gate_proj, up_proj, down_proj</code> 以实现完整覆盖。</td></tr>
<tr><td><code>lora_dropout</code></td><td>0.0–0.1</td><td>正则化。小数据集通常用 0.05。</td></tr>
<tr><td><code>bias</code></td><td><code>&quot;none&quot;</code></td><td>训练偏置项新增参数极少但很少有帮助。</td></tr>
<tr><td>学习率</td><td><span class="arithmatex">\(1\text{e-}4\)</span> 到 <span class="arithmatex">\(3\text{e-}4\)</span></td><td>高于全量微调（只更新适配器）。</td></tr></tbody>
</table>
</div>

<div class="admonition warning"><p class="admonition-title">秩选择经验法则</p>
<ul>
<li><p><strong>r=8</strong>：简单任务（单领域聊天、分类）。非常省内存。</p></li>
<li><p><strong>r=16</strong>：通用微调。良好的默认值。</p></li>
<li><p><strong>r=32–64</strong>：复杂任务（数学、代码、多轮推理）。接近全量微调质量。</p></li>
<li><p><strong>r=128+</strong>：收益递减；考虑全量微调或更高秩的 QLoRA。</p></li>
<li><p><strong>关键指标</strong>：若训练损失明显高于全量微调损失而停滞不前，则增大秩。</p></li>
</ul>
</div>

### 1.9.3　LoRA 变体 {#lora-variants}

<p class="table-caption" id="table-1-13"><span class="table-number">表 1.13　</span>LoRA 变体及其创新。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>关键创新</strong></th><th scope="col"><strong>适用场景</strong></th></tr></thead>
<tbody><tr><td><strong>QLoRA</strong> <a class="book-citation" id="cite-dettmers2023qlora-1" href="../../references.md#ref-dettmers2023qlora" aria-label="参考文献 75">[75]</a></td><td>4-bit 量化基座 + BF16 的 LoRA。NF4 数据类型 + 双重量化。</td><td>在单张 48GB GPU 上微调 70B。</td></tr>
<tr><td><strong>DoRA</strong> <a class="book-citation" id="cite-liu2024dora-1" href="../../references.md#ref-liu2024dora" aria-label="参考文献 228">[228]</a></td><td>将 <span class="arithmatex">\(W\)</span> 分解为幅度与方向；LoRA 只更新方向。</td><td>推理任务的泛化更好。</td></tr>
<tr><td><strong>LoRA+</strong> <a class="book-citation" id="cite-hayou2024loraplus-1" href="../../references.md#ref-hayou2024loraplus" aria-label="参考文献 136">[136]</a></td><td>为 <span class="arithmatex">\(A\)</span>/<span class="arithmatex">\(B\)</span> 使用不同的 LR。</td><td>免费的 2% 提升；无额外成本。</td></tr>
<tr><td><strong>AdaLoRA</strong> <a class="book-citation" id="cite-zhang2023adalora-1" href="../../references.md#ref-zhang2023adalora" aria-label="参考文献 424">[424]</a></td><td>跨层动态秩预算（基于 SVD 的重要性）。</td><td>算力预算非常紧张时。</td></tr>
<tr><td><strong>rsLoRA</strong> <a class="book-citation" id="cite-kalajdzievski2023rslora-2" href="../../references.md#ref-kalajdzievski2023rslora" aria-label="参考文献 176">[176]</a></td><td>用 <span class="arithmatex">\(\alpha/\sqrt{r}\)</span> 而非 <span class="arithmatex">\(\alpha/r\)</span> 缩放。在高秩下稳定。</td><td>使用 <span class="arithmatex">\(r \geq 64\)</span> 时。</td></tr>
<tr><td><strong>VeRA</strong> <a class="book-citation" id="cite-kopiczko2024vera-1" href="../../references.md#ref-kopiczko2024vera" aria-label="参考文献 190">[190]</a></td><td>共享冻结的随机 <span class="arithmatex">\(A, B\)</span>；只训练对角缩放。</td><td>极致的参数效率。</td></tr>
<tr><td><strong>LoRA-FA</strong></td><td>初始化后冻结 <span class="arithmatex">\(A\)</span>；只训练 <span class="arithmatex">\(B\)</span>。将 LoRA 内存减半。</td><td>内存受限场景。</td></tr></tbody>
</table>
</div>

#### 关键扩展详解 {#key-extensions-explained}

##### DoRA——权重分解低秩适配（Weight-Decomposed Low-Rank Adaptation，DoRA）。 {#dora-weight-decomposed-low-rank-adaptation.}

<p>DoRA <a class="book-citation" id="cite-liu2024dora-2" href="../../references.md#ref-liu2024dora" aria-label="参考文献 228">[228]</a>观察到，全量微调倾向于改变权重向量的<em>方向</em>多于幅度。标准 LoRA 把二者混在一起。DoRA 将每个权重列分解为幅度 <span class="arithmatex">\(m = \|W\|_\text{col}\)</span> 和方向 <span class="arithmatex">\(\hat{V} = W / \|W\|_\text{col}\)</span>，然后只对方向应用 LoRA：</p>



<div class="arithmatex">\[W&#x27; = m \odot \hat{V}&#x27;, \quad \hat{V}&#x27; = \frac{W + BA}{\|W + BA\|_\text{col}}\]</div>

<p>幅度 <span class="arithmatex">\(m\)</span> 是单独的可学习向量（每列一个标量）。它在推理与指令遵循基准上始终比 LoRA 高 1–3%，且无额外推理成本（部署时合并）。</p>

##### LoRA+——非对称学习率。 {#lora-asymmetric-learning-rates.}

<p>Hayou 等 <a class="book-citation" id="cite-hayou2024loraplus-2" href="../../references.md#ref-hayou2024loraplus" aria-label="参考文献 136">[136]</a>表明，LoRA 中的矩阵 <span class="arithmatex">\(A\)</span> 和 <span class="arithmatex">\(B\)</span> 拥有不同的最优学习率。由于 <span class="arithmatex">\(B\)</span> 初始化为零，它与从 <span class="arithmatex">\(\mathcal{N}(0, \sigma^2)\)</span> 初始化的 <span class="arithmatex">\(A\)</span> 处于截然不同的状态。设置 <span class="arithmatex">\(\eta_B \approx 16 \times \eta_A\)</span> 可提升收敛速度并将最终质量提高约 2%——一个仅需一行配置改动的免费收益：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># PEFT 中的 LoRA+：为每个矩阵设置不同的学习率</span>
<span class="n">optimizer_grouped_parameters</span> <span class="o">=</span> <span class="p">[</span>
    <span class="p">{</span><span class="s2">"params"</span><span class="p">:</span> <span class="p">[</span><span class="n">p</span> <span class="k">for</span> <span class="n">n</span><span class="p">,</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">model</span><span class="o">.</span><span class="n">named_parameters</span><span class="p">()</span> <span class="k">if</span> <span class="s2">"lora_B"</span> <span class="ow">in</span> <span class="n">n</span><span class="p">],</span>
     <span class="s2">"lr"</span><span class="p">:</span> <span class="mf">2e-4</span> <span class="o">*</span> <span class="mi">16</span><span class="p">},</span>   <span class="c1"># B 矩阵：更高的学习率</span>
    <span class="p">{</span><span class="s2">"params"</span><span class="p">:</span> <span class="p">[</span><span class="n">p</span> <span class="k">for</span> <span class="n">n</span><span class="p">,</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">model</span><span class="o">.</span><span class="n">named_parameters</span><span class="p">()</span> <span class="k">if</span> <span class="s2">"lora_A"</span> <span class="ow">in</span> <span class="n">n</span><span class="p">],</span>
     <span class="s2">"lr"</span><span class="p">:</span> <span class="mf">2e-4</span><span class="p">},</span>         <span class="c1"># A 矩阵：基础学习率</span>
<span class="p">]</span>
</code></pre></div>

##### VeRA——基于向量的随机矩阵适配（Vector-based Random Matrix Adaptation，VeRA）。 {#vera-vector-based-random-matrix-adaptation.}

<p>VeRA <a class="book-citation" id="cite-kopiczko2024vera-2" href="../../references.md#ref-kopiczko2024vera" aria-label="参考文献 190">[190]</a>把参数效率推向极致：它不学习 <span class="arithmatex">\(A\)</span> 和 <span class="arithmatex">\(B\)</span>，而是将它们<em>冻结</em>为所有层之间共享的随机矩阵，仅训练两个对角缩放向量 <span class="arithmatex">\(d_b \in \mathbb{R}^r\)</span> 和 <span class="arithmatex">\(d_a \in \mathbb{R}^d\)</span>：</p>



<div class="arithmatex">\[\Delta W = B \cdot \text{diag}(d_b) \cdot A \cdot \text{diag}(d_a)\]</div>

<p>这相比 LoRA 将可训练参数减少约 10<span class="arithmatex">\(\times\)</span>（每层只有 <span class="arithmatex">\(r + d\)</span> 个参数），同时达到 LoRA 90–95% 的质量。最适合需要数百个任务专用适配器、且存储极小化的场景。</p>

<div class="admonition example"><p class="admonition-title">QLoRA 内存节省</p>
<p><strong>70B 模型全量微调</strong>：140 GB（权重） + 280 GB（优化器） + 140 GB（梯度） = 560 GB（7<span class="arithmatex">\(\times\)</span> A100-80GB）。</p>

<p><strong>70B QLoRA（r=16，所有线性层）</strong>：</p>

<ul>
<li><p>NF4 格式的基座模型：<span class="arithmatex">\(70\text{B} \times 0.5 = 35\)</span> GB</p></li>
<li><p>BF16 格式的 LoRA 适配器：<span class="arithmatex">\(\sim\)</span>160 MB</p></li>
<li><p>优化器状态（仅适配器）：<span class="arithmatex">\(\sim\)</span>320 MB</p></li>
<li><p>激活值（梯度检查点）：<span class="arithmatex">\(\sim\)</span>8 GB</p></li>
<li><p><strong>总计：<span class="arithmatex">\(\sim\)</span>44 GB</strong>——可装入单张 48GB GPU！</p></li>
</ul>
</div>

<div class="highlight"><pre><span></span><code><span class="c1"># 使用 PEFT 的 QLoRA 配置</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">peft</span><span class="w"> </span><span class="kn">import</span> <span class="n">LoraConfig</span><span class="p">,</span> <span class="n">get_peft_model</span><span class="p">,</span> <span class="n">prepare_model_for_kbit_training</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">BitsAndBytesConfig</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="c1"># 4-bit 量化配置</span>
<span class="n">bnb_config</span> <span class="o">=</span> <span class="n">BitsAndBytesConfig</span><span class="p">(</span>
    <span class="n">load_in_4bit</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="n">bnb_4bit_quant_type</span><span class="o">=</span><span class="s2">"nf4"</span><span class="p">,</span>           <span class="c1"># NormalFloat4 - 对权重最优</span>
    <span class="n">bnb_4bit_compute_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span> <span class="c1"># 计算使用 BF16</span>
    <span class="n">bnb_4bit_use_double_quant</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>       <span class="c1"># 对量化常数再次量化</span>
<span class="p">)</span>

<span class="c1"># LoRA 配置</span>
<span class="n">lora_config</span> <span class="o">=</span> <span class="n">LoraConfig</span><span class="p">(</span>
    <span class="n">r</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span>
    <span class="n">lora_alpha</span><span class="o">=</span><span class="mi">32</span><span class="p">,</span>                        <span class="c1"># alpha/r = 2 倍缩放</span>
    <span class="n">target_modules</span><span class="o">=</span><span class="p">[</span><span class="s2">"q_proj"</span><span class="p">,</span> <span class="s2">"k_proj"</span><span class="p">,</span> <span class="s2">"v_proj"</span><span class="p">,</span> <span class="s2">"o_proj"</span><span class="p">,</span>
                    <span class="s2">"gate_proj"</span><span class="p">,</span> <span class="s2">"up_proj"</span><span class="p">,</span> <span class="s2">"down_proj"</span><span class="p">],</span>
    <span class="n">lora_dropout</span><span class="o">=</span><span class="mf">0.05</span><span class="p">,</span>
    <span class="n">bias</span><span class="o">=</span><span class="s2">"none"</span><span class="p">,</span>
    <span class="n">task_type</span><span class="o">=</span><span class="s2">"CAUSAL_LM"</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">prepare_model_for_kbit_training</span><span class="p">(</span><span class="n">model</span><span class="p">)</span>  <span class="c1"># 为 QLoRA 做准备</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">get_peft_model</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">lora_config</span><span class="p">)</span>       <span class="c1"># 添加 LoRA 适配器</span>
<span class="n">model</span><span class="o">.</span><span class="n">print_trainable_parameters</span><span class="p">()</span>
<span class="c1"># 输出：trainable params: 83,886,080 || all params: 70,553,706,496 || 0.12%</span>
</code></pre></div>

### 1.9.4　其他 PEFT 方法 {#other-peft-approaches}

<p>LoRA 主导了现代实践，但并非唯一的参数高效方法。为完整起见，列出主要的替代方案：</p>

<p class="table-caption" id="table-1-14"><span class="table-number">表 1.14　</span>参数高效微调（Parameter-Efficient Fine-Tuning，PEFT）方法家族。LoRA 是 LLM 微调的事实标准；其余方法列在此处用于历史背景和小众使用场景。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>机制</strong></th><th scope="col"><strong>优点 / 缺点</strong></th><th scope="col"><strong>现状</strong></th></tr></thead>
<tbody><tr><td><strong>LoRA</strong> <a class="book-citation" id="cite-hu2021lora-2" href="../../references.md#ref-hu2021lora" aria-label="参考文献 146">[146]</a>（及变体）</td><td>在现有权重上添加低秩矩阵</td><td>推理时可合并（零开销）；支持完善；适用于所有架构</td><td><strong>标准</strong></td></tr>
<tr><td><strong>Adapters</strong> <a class="book-citation" id="cite-houlsby2019adapters-1" href="../../references.md#ref-houlsby2019adapters" aria-label="参考文献 144">[144]</a></td><td>在层之间插入小型瓶颈 MLP</td><td>模块化；可堆叠；增加推理延迟（额外的串行层）</td><td>很少使用</td></tr>
<tr><td><strong>前缀微调</strong> <a class="book-citation" id="cite-li2021prefix-1" href="../../references.md#ref-li2021prefix" aria-label="参考文献 211">[211]</a></td><td>在每层 K/V 前拼接可学习的「虚拟 Token」</td><td>不修改权重；对生成任务有效；占用上下文长度</td><td>小众</td></tr>
<tr><td><strong>提示微调</strong> <a class="book-citation" id="cite-lester2021prompt-1" href="../../references.md#ref-lester2021prompt" aria-label="参考文献 204">[204]</a></td><td>在输入前拼接可学习的软提示 Embedding</td><td>参数极少（<span class="arithmatex">\(&lt;\)</span>0.01%）；复杂任务上弱于 LoRA</td><td>小众</td></tr>
<tr><td><strong>IA3</strong> <a class="book-citation" id="cite-liu2022ia3-1" href="../../references.md#ref-liu2022ia3" aria-label="参考文献 222">[222]</a></td><td>用学到的向量对 K、V 和 FFN 激活进行重缩放</td><td>参数比 LoRA 更少；可合并；容量有限</td><td>已弃用</td></tr>
<tr><td><strong>BitFit</strong> <a class="book-citation" id="cite-zaken2022bitfit-1" href="../../references.md#ref-zaken2022bitfit" aria-label="参考文献 417">[417]</a></td><td>只训练偏置项</td><td>参数接近于零；在简单任务上意外有效；表达力有限</td><td>历史方法</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 LoRA 胜出</p>
<p>LoRA 成为标准是因为它独特地兼具：(1) <strong>零推理开销</strong>——适配器可合并入基座权重，不像 Adapters 或前缀微调会增加延迟或消耗上下文；(2) <strong>可组合性</strong>——多个 LoRA 适配器可在服务时切换，用于多租户部署；(3) <strong>生态支持</strong>——HuggingFace PEFT、TRL、vLLM 以及所有主流框架都对 LoRA 有一等支持；(4) <strong>大规模验证</strong>——Meta、Google 以及 HuggingFace 上大多数开源微调都在生产中使用。除非你有 LoRA 无法满足的特定约束，否则它应当是你的默认选择。</p>
</div>

## 1.10　专家混合模型（Mixture of Experts，MoE） {#mixture-of-experts-moe}

<p>专家混合模型 <a class="book-citation" id="cite-shazeer2017outrageously-1" href="../../references.md#ref-shazeer2017outrageously" aria-label="参考文献 325">[325]</a><a class="book-citation" id="cite-jiang2024mixtral-1" href="../../references.md#ref-jiang2024mixtral" aria-label="参考文献 166">[166]</a> 通过对每个 Token 只激活参数的一个子集，从而在不按比例增加计算成本的情况下扩展模型容量。</p>

### 1.10.1　架构 {#architecture}

<div class="admonition note"><p class="admonition-title">MoE 层</p>
<p>在一个 MoE Transformer 中，每个块中的 FFN 层被替换为 <span class="arithmatex">\(N\)</span> 个并行的“专家” FFN，再加上一个 <strong>路由器（router）</strong> 来选择使用哪些专家：</p>



<div class="arithmatex">\[\text{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot E_i(x), \quad g(x) = \text{TopK}(\text{softmax}(W_r x))\]</div>

<ul>
<li><p><span class="arithmatex">\(E_i\)</span> 是专家网络（标准的 FFN 层）</p></li>
<li><p><span class="arithmatex">\(g_i(x)\)</span> 是来自路由器的门控权重（只有 top-<span class="arithmatex">\(K\)</span> 个非零）</p></li>
<li><p>通常每个 Token 在 <span class="arithmatex">\(N=8\)</span>–64 个专家中激活 <span class="arithmatex">\(K=2\)</span> 个</p></li>
<li><p>总参数量随 <span class="arithmatex">\(N\)</span> 增长；<strong>激活参数量</strong> 按 FFN 大小的 <span class="arithmatex">\(K/N\)</span> 比例增长</p></li>
</ul>
</div>

<figure id="figure-1-11"><a href="../../assets/figures/fig_012_fig12.png" target="_blank" rel="noopener" aria-label="查看图 1.11 原图" title="点击查看原图"><img src="../../assets/figures/fig_012_fig12.png" alt="具有 8 个专家和 Top-2 路由的 MoE 层。每个 Token 只计算门控值最高的两个专家；其余专家被完全跳过。" loading="lazy"></a><figcaption><span class="figure-number">图 1.11　</span>具有 8 个专家和 Top-2 路由的 MoE 层。每个 Token 只计算门控值最高的两个专家；其余专家被完全跳过。</figcaption></figure>

### 1.10.2　负载均衡 {#load-balancing}

<div class="admonition warning"><p class="admonition-title">负载均衡问题</p>
<p>在没有约束的情况下，路由器可能会把大多数 Token 都发送到相同的 1–2 个专家上（“专家坍缩（expert collapse）”）。这既浪费了容量，又造成了跨 GPU 的计算不均衡（每个专家通常位于不同的 GPU 上）。</p>

<p><strong>解决方案</strong>：添加一个辅助的负载均衡损失：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{bal}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot p_i\]</div>

<p>其中 <span class="arithmatex">\(f_i\)</span> 表示路由到专家 <span class="arithmatex">\(i\)</span> 的 Token 比例，<span class="arithmatex">\(p_i\)</span> 表示专家 <span class="arithmatex">\(i\)</span> 的平均路由概率。这鼓励了对专家的均匀利用。</p>
</div>

### 1.10.3　噪声 Top-K 门控（Noisy Top-K Gating）：让离散路由变得可训练 {#noisy-top-k-gating-making-discrete-routing-trainable}

<p>MoE 的核心挑战在于 <strong>top-<span class="arithmatex">\(k\)</span> 选择是不可微的</strong>——你无法通过一个硬性的“挑选前 2 名”操作反向传播。该领域发展出了两个关键技巧来解决这个问题：</p>

<div class="admonition tip"><p class="admonition-title">路由可微性问题</p>
<p>路由器为每个专家计算 logits <span class="arithmatex">\(h(x) = W_r \cdot x\)</span>，然后选择 top-<span class="arithmatex">\(k\)</span>。但是：</p>

<ul>
<li><p><em>被选中</em> 的专家会通过其门控权重获得梯度（在被选中者上做 softmax）</p></li>
<li><p><em>选择决策本身</em>（挑选哪 <span class="arithmatex">\(k\)</span> 个）的梯度为零</p></li>
<li><p>如果没有技巧，路由器可能会卡住：一个从未被选中的专家 <span class="arithmatex">\(\rightarrow\)</span> 永远得不到梯度信号 <span class="arithmatex">\(\rightarrow\)</span> 永远不会被选中</p></li>
</ul>
</div>

##### 方法 1：噪声 Top-K 门控（Noisy Top-K Gating） <a class="book-citation" id="cite-shazeer2017outrageously-2" href="../../references.md#ref-shazeer2017outrageously" aria-label="参考文献 325">[325]</a>。 {#approach-1-noisy-top-k-gating-.}

<p>在进行 top-<span class="arithmatex">\(k\)</span> 选择 <em>之前</em>，向路由器的 logits 添加可学习的高斯噪声：</p>



<div class="arithmatex">\[\begin{align*}
h(x) &amp;= W_g \cdot x \tag{clean logits} \\

  H(x) &amp;= h(x) + \epsilon \cdot \text{Softplus}(W_{\text{noise}} \cdot x), \quad \epsilon \sim \mathcal{N}(0, 1) \tag{noisy logits} \\

  \text{TopK}(v, k)_i &amp;= \begin{cases} v_i &amp; \text{if } v_i \text{ is in the top } k \\ -\infty &amp; \text{otherwise} \end{cases} \tag{1.11}\\

  g(x) &amp;= \text{softmax}\big(\text{TopK}(H(x),\, k)\big) \tag{sparse gates}
\end{align*}\]</div>

<ul>
<li><p><span class="arithmatex">\(W_{\text{noise}}\)</span> 是 <em>学习得到的</em> 噪声幅度——模型会学习每个专家需要多少探索量</p></li>
<li><p>在训练过程中，噪声偶尔会将“弱势”专家提升到 top-<span class="arithmatex">\(k\)</span> 中，使其获得梯度信号</p></li>
<li><p>在推理时去除噪声：使用干净的 logits <span class="arithmatex">\(h(x)\)</span> 进行确定性路由</p></li>
<li><p>Softplus 确保噪声尺度始终为正</p></li>
</ul>

##### 方法 2：Gumbel-Softmax 技巧（Gumbel-Softmax Trick，用于可微的离散采样）。 {#approach-2-gumbel-softmax-trick-for-differentiable-discrete-sampling.}

<p>来自变分推断文献的另一种方法 <a class="book-citation" id="cite-jang2017categorical-1" href="../../references.md#ref-jang2017categorical" aria-label="参考文献 161">[161]</a>。<strong>Gumbel-Max 技巧（Gumbel-Max trick）</strong> 提供了从类别分布中精确采样的方式：</p>



<div class="arithmatex">\[z = \arg\max_i \left[ \log \pi_i + G_i \right], \quad G_i \sim \text{Gumbel}(0,1)
\tag{1.12}\]</div>

<p>其中 Gumbel 噪声由 <span class="arithmatex">\(G_i = -\log(-\log(U_i)),\; U_i \sim \text{Uniform}(0,1)\)</span> 生成。</p>

<p>对于 <strong>top-<span class="arithmatex">\(k\)</span> 路由</strong>：取 <span class="arithmatex">\((\log \pi_i + G_i)\)</span> 的 top-<span class="arithmatex">\(k\)</span>，等价于从由 <span class="arithmatex">\(\pi\)</span> 定义的类别分布中 <em>无放回地</em> 抽取 <span class="arithmatex">\(k\)</span> 个样本。</p>

<p>由于 <span class="arithmatex">\(\arg\max\)</span> 是不可微的，<strong>Gumbel-Softmax</strong> 松弛将其替换为一个由温度控制的 softmax：</p>



<div class="arithmatex">\[\hat{g}_i = \frac{\exp\left((\log \pi_i + G_i) / \tau\right)}{\sum_j \exp\left((\log \pi_j + G_j) / \tau\right)}
\tag{1.13}\]</div>

<ul>
<li><p><span class="arithmatex">\(\tau \to 0\)</span>：趋近于硬 one-hot（精确但不可微）</p></li>
<li><p><span class="arithmatex">\(\tau \to \infty\)</span>：趋近于均匀分布（可微但无信息量）</p></li>
<li><p>实践中，在训练过程中将 <span class="arithmatex">\(\tau\)</span> 从 1.0 退火到 0.1–0.5</p></li>
<li><p><strong>直通估计器（Straight-through estimator）</strong>：前向传播中使用硬 top-<span class="arithmatex">\(k\)</span>，反向传播中使用 Gumbel-Softmax 梯度——两全其美</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">实际中使用的是哪种方法？</p>
<ul>
<li><p><strong>Sparsely-Gated MoE <a class="book-citation" id="cite-shazeer2017outrageously-3" href="../../references.md#ref-shazeer2017outrageously" aria-label="参考文献 325">[325]</a>、Mixtral <a class="book-citation" id="cite-jiang2024mixtral-2" href="../../references.md#ref-jiang2024mixtral" aria-label="参考文献 166">[166]</a>、DeepSeek-V2 <a class="book-citation" id="cite-deepseekv2-1" href="../../references.md#ref-deepseekv2" aria-label="参考文献 69">[69]</a></strong>：使用带高斯噪声的噪声 Top-K。简单、有效，在大规模上得到了充分验证。</p></li>
<li><p><strong>Switch Transformer <a class="book-citation" id="cite-fedus2022switch-1" href="../../references.md#ref-fedus2022switch" aria-label="参考文献 89">[89]</a></strong>：简化为不带噪声的 Top-1（仅依赖负载均衡损失）。</p></li>
<li><p><strong>研究型 / 小规模 MoE</strong>：一些工作使用 Gumbel-Softmax 实现完全可微的路由，尤其是当学习路由本身就是研究目标时。</p></li>
<li><p><strong>关键见解</strong>：两种方法都通过噪声注入解决了同一个问题（让离散选择可训练）。高斯噪声更简单；Gumbel 噪声在类别采样上有更强的理论保证。</p></li>
</ul>

<h4 id="heading-1-140">无辅助损失负载均衡。</h4>

<p>标准做法是在训练损失上加一个均衡惩罚项，但 DeepSeek <a class="book-citation" id="cite-wang2024auxfree-1" href="../../references.md#ref-wang2024auxfree" aria-label="参考文献 372">[372]</a> 证明这会与路由信号相互对抗：损失把利用率推向均匀，而路由器想要专业化。他们的解决方案把均衡完全移出梯度，改为一个基于观测利用率更新的逐专家路由<em>偏置（bias）</em>——同时实现了更好的质量<em>和</em>更高的专家专业化。Qwen 的后续洞见进一步揭示：<em>聚合范围</em>比机制本身更重要——按微批次（常见默认）计算均衡统计会悄无声息地摧毁专业化，因为微批次过于同质。像 MAI-Thinking-1 所证实的那样，在整个全局批次上聚合，才能恢复专家分化所需的多样性信号。</p>
</div>

### 1.10.4　值得关注的 MoE 模型 {#notable-moe-models}

<span id="table-1-inline-20"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模型</strong></th><th scope="col"><strong>总参数量</strong></th><th scope="col"><strong>激活参数量</strong></th><th scope="col"><strong>专家</strong></th><th scope="col"><strong>创新点</strong></th></tr></thead>
<tbody><tr><td>Switch Transformer <a class="book-citation" id="cite-fedus2022switch-2" href="../../references.md#ref-fedus2022switch" aria-label="参考文献 89">[89]</a></td><td>1.6T</td><td>100B</td><td>128, Top-1</td><td>首个大规模 MoE；简化的路由</td></tr>
<tr><td>Mixtral 8x7B <a class="book-citation" id="cite-jiang2024mixtral-3" href="../../references.md#ref-jiang2024mixtral" aria-label="参考文献 166">[166]</a></td><td>47B</td><td>13B</td><td>8, Top-2</td><td>开放权重；质量媲美 Llama-2 70B</td></tr>
<tr><td>DeepSeek-V2 <a class="book-citation" id="cite-deepseekv2-2" href="../../references.md#ref-deepseekv2" aria-label="参考文献 69">[69]</a></td><td>236B</td><td>21B</td><td>160, Top-6</td><td>带共享 + 路由专家的 DeepSeekMoE</td></tr>
<tr><td>Qwen-MoE <a class="book-citation" id="cite-qwen2024qwen25-1" href="../../references.md#ref-qwen2024qwen25" aria-label="参考文献 350">[350]</a></td><td>14.3B</td><td>2.7B</td><td>60, Top-4</td><td>为提升效率而设计的细粒度专家</td></tr>
<tr><td>DBRX <a class="book-citation" id="cite-databricks2024dbrx-1" href="../../references.md#ref-databricks2024dbrx" aria-label="参考文献 67">[67]</a></td><td>132B</td><td>36B</td><td>16, Top-4</td><td>每块 4 个专家的细粒度结构</td></tr></tbody>
</table>
</div>

## 1.11　LLM 训练中的多样性 {#diversity-in-llm-training}

<p>多样性——在训练数据、模型输出和优化轨迹上的多样性——对于防止模式坍缩并确保鲁棒的通用 LLM 至关重要。本节介绍适用于所有 LLM 训练阶段的关键多样性机制。</p>

### 1.11.1　采样多样性 {#sampling-diversity}

<div class="admonition note"><p class="admonition-title">用于多样化生成的采样策略</p>
<ul>
<li><p><strong>温度 <span class="arithmatex">\(\tau\)</span></strong>：<span class="arithmatex">\(P(x_i) \propto \exp(\text{logit}_i / \tau)\)</span>。更高的 <span class="arithmatex">\(\tau\)</span> = 更均匀的分布 = 更多样化。典型值：RLHF 生成中使用 <span class="arithmatex">\(\tau=0.7\)</span>–<span class="arithmatex">\(1.0\)</span>。</p></li>
<li><p><strong>Top-<span class="arithmatex">\(k\)</span></strong>：只从概率最高的 <span class="arithmatex">\(k\)</span> 个 Token 中采样。防止退化的低概率 Token。</p></li>
<li><p><strong>Top-<span class="arithmatex">\(p\)</span>（核采样，Nucleus Sampling）</strong>：从累计概率 <span class="arithmatex">\(\geq p\)</span> 的最小 Token 集合中采样。自适应：模型不确定时更多样。</p></li>
<li><p><strong>Min-<span class="arithmatex">\(p\)</span></strong>：仅保留满足 <span class="arithmatex">\(P \geq p_{\min} \times P_{\max}\)</span> 的 Token。比 top-<span class="arithmatex">\(k\)</span> 更具原理性。</p></li>
<li><p><strong>频率/出现惩罚（Frequency/presence penalty）</strong>：对已在回复中出现过的 Token 进行惩罚。鼓励词汇多样性。</p></li>
</ul>
</div>

### 1.11.2　训练数据多样性 {#training-data-diversity}

<ul>
<li><p><strong>提示多样性</strong>：覆盖不同的领域、难度等级和格式。Goldilocks 原则：提示的成功率应在 20–80% 之间。</p></li>
<li><p><strong>去重</strong>：移除近似重复的训练样本（MinHash、n-gram 重叠）。重复样本会导致对特定模式的过拟合。</p></li>
<li><p><strong>数据混合</strong>：使用温度加权采样或课程策略，在任务/领域之间进行平衡。</p></li>
</ul>

### 1.11.3　促进多样性的方法 {#diversity-promoting-methods}

<span id="table-1-inline-21"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>它如何促进多样性</strong></th></tr></thead>
<tbody><tr><td>温度缩放（Temperature scaling）</td><td>更高的 <span class="arithmatex">\(\tau\)</span> 会让分布变得更平坦；更多 Token 变得合理。</td></tr>
<tr><td>Top-<span class="arithmatex">\(p\)</span> / Min-<span class="arithmatex">\(p\)</span></td><td>自适应阈值允许模型在不确定时进行更广的采样。</td></tr>
<tr><td>频率惩罚（Frequency penalty）</td><td>惩罚重复出现的 Token，强制在一次回复内产生词汇多样性。</td></tr>
<tr><td>数据去重</td><td>从训练数据中移除近似重复样本，防止对特定模式的过拟合。</td></tr>
<tr><td>多领域混合</td><td>跨领域的温度加权采样确保了广泛的覆盖。</td></tr>
<tr><td>口头化采样（Verbalized sampling）</td><td>提示模型显式地用语言表达出对回复的概率分布 <a class="book-citation" id="cite-zhang2025verbalized-1" href="../../references.md#ref-zhang2025verbalized" aria-label="参考文献 422">[422]</a>。见 §<a href="../07-grpo/index.md#grpo-variants-and-extensions">7.5</a>。</td></tr></tbody>
</table>
</div>
