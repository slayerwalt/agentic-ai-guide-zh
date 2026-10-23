# 第 1 章 · 训练理论与实践 {#reading-reading-02}

## 1.5　LLM 训练的优化理论 {#optimization-theory-for-llm-training}

<p>训练一个大语言模型意味着寻找一组参数 <span class="arithmatex">\(\theta\)</span>（数十亿个权重），使损失函数 <span class="arithmatex">\(\mathcal{L}(\theta)\)</span> ——通常是下一个 token 的负对数似然——最小化。这是一个在极高维空间中的优化问题，用来在该空间中导航的算法直接决定了训练是成功、发散还是停滞。</p>

### 1.5.1　梯度下降：基础 {#gradient-descent-the-foundation}

##### 什么是梯度？ {#what-is-a-gradient}

<p>梯度 <span class="arithmatex">\(\nabla_\theta \mathcal{L}\)</span> 是一个指向损失函数<em>最陡上升方向</em>的向量。每个分量 <span class="arithmatex">\(\frac{\partial \mathcal{L}}{\partial \theta_i}\)</span> 告诉我们：如果对参数 <span class="arithmatex">\(\theta_i\)</span> 略微增加一点，损失会变化多少。为了<em>降低</em>损失，我们沿相反方向移动：</p>



<div class="arithmatex">\[\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
\tag{1.10}\]</div>

<p>其中 <span class="arithmatex">\(\eta &gt; 0\)</span> 是<strong>学习率（Learning Rate）</strong>——也就是步长。这就是<strong>梯度下降（Gradient Descent）</strong> <a class="book-citation" id="cite-rumelhart1986learning-1" href="../../references.md#ref-rumelhart1986learning" aria-label="参考文献 310">[310]</a>。</p>

<figure id="figure-1-8"><a href="../../assets/figures/fig_009_fig9.png" target="_blank" rel="noopener" aria-label="查看图 1.8 原图" title="点击查看原图"><img src="../../assets/figures/fig_009_fig9.png" alt="梯度下降：从随机初始化 \(\theta_0\) 开始，每一步都将参数沿降低损失的方向移动，步长由学习率 \(\eta\) 控制。该过程会向某个（局部）最小值收敛。" loading="lazy"></a><figcaption><span class="figure-number">图 1.8　</span>梯度下降：从随机初始化 <span class="arithmatex">\(\theta_0\)</span> 开始，每一步都将参数沿降低损失的方向移动，步长由学习率 <span class="arithmatex">\(\eta\)</span> 控制。该过程会向某个（局部）最小值收敛。</figcaption></figure>

##### 为何全梯度下降不可行。 {#why-full-gradient-descent-is-impractical.}

<p>计算精确梯度需要在<em>整个</em>训练集上评估损失（对 LLM 而言是万亿级 token），计算开销极其高昂——单步梯度更新就需要遍历全部数据。</p>

##### 随机梯度下降（Stochastic Gradient Descent，SGD）。 {#stochastic-gradient-descent-sgd.}

<p>解决办法是：用一个小的随机数据子集（<strong>小批量，mini-batch</strong>）来估计梯度 <a class="book-citation" id="cite-robbins1951stochastic-1" href="../../references.md#ref-robbins1951stochastic" aria-label="参考文献 308">[308]</a>：</p>



<div class="arithmatex">\[\nabla_\theta \mathcal{L}(\theta) \approx \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell(\theta; x_i)\]</div>

<p>其中 <span class="arithmatex">\(B\)</span> 是批大小（对 LLM 通常是 1K–4M 个 token）。小批量梯度是真实梯度的<em>含噪但无偏</em>的估计。</p>

<div class="admonition note"><p class="admonition-title">为何小批量 SGD 有效</p>
<ul>
<li><p><strong>计算效率</strong>：每步开销是 <span class="arithmatex">\(O(B)\)</span> 而非 <span class="arithmatex">\(O(N_{\text{total}})\)</span>。当 <span class="arithmatex">\(B = 4096\)</span> 而总 token 为 15T 时，每步开销约便宜 40 亿倍。</p></li>
<li><p><strong>噪声即正则化</strong>：随机噪声有助于逃离尖锐的局部极小值，找到泛化性更好的平坦区域。</p></li>
<li><p><strong>GPU 利用率</strong>：小批量足够大，能充分占满 GPU 并行度（矩阵乘法变为计算受限而非显存受限）。</p></li>
<li><p><strong>收敛性</strong>：理论上以 <span class="arithmatex">\(O(1/\sqrt{T})\)</span> 的速率收敛到局部极小值（比精确 GD 的 <span class="arithmatex">\(O(1/T)\)</span> 慢，但每步开销便宜数百万倍）。</p></li>
</ul>
</div>

##### 从 SGD 到自适应方法。 {#from-sgd-to-adaptive-methods.}

<p>带动量的 SGD 对视觉模型（CNN）效果不错，但 LLM 训练需要<strong>自适应优化器（Adaptive Optimizer）</strong>——即为每个参数维护各自学习率的算法。</p>

### 1.5.2　为何朴素 SGD 在 LLM 上失效 {#why-vanilla-sgd-fails-for-llms}

<p>随机梯度下降按下式更新权重：</p>



<div class="arithmatex">\[\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)\]</div>

<div class="admonition warning"><p class="admonition-title">SGD 在 LLM 上的问题</p>
<ul>
<li><p><strong>各层梯度尺度不同：</strong>Transformer 早期层的梯度远小于后期层（梯度消失）。单一学习率 <span class="arithmatex">\(\eta\)</span> 对一些参数过大，对另一些参数又过小。</p></li>
<li><p><strong>稀疏梯度：</strong>embedding 层只对当前批次出现的 token 有梯度，大部分行的梯度为零。带动量的 SGD 会在零梯度行上浪费动量。</p></li>
<li><p><strong>鞍点：</strong>高维损失曲面存在大量鞍点。SGD 可能停滞，而自适应方法能更快逃脱。</p></li>
<li><p><strong>对学习率敏感：</strong>SGD 需要仔细调参；<span class="arithmatex">\(\eta\)</span> 变化 2 倍就可能导致发散。</p></li>
</ul>
</div>

### 1.5.3　Adam — 自适应矩估计（Adaptive Moment Estimation） {#adam-adaptive-moment-estimation}

<p>Adam <a class="book-citation" id="cite-kingma2015adam-1" href="../../references.md#ref-kingma2015adam" aria-label="参考文献 185">[185]</a> 为每个参数维护梯度一阶矩（均值）与二阶矩（非中心化方差）的估计。</p>

<div class="admonition note"><p class="admonition-title">Adam 更新方程</p>
<p>给定梯度 <span class="arithmatex">\(g_t = \nabla_\theta \mathcal{L}(\theta_t)\)</span> 与超参数 <span class="arithmatex">\(\beta_1, \beta_2, \epsilon, \eta\)</span>：</p>

<p><strong>第 1 步 – 更新有偏一阶矩估计：</strong></p>



<div class="arithmatex">\[m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\]</div>

<p><strong>第 2 步 – 更新有偏二阶矩估计：</strong></p>



<div class="arithmatex">\[v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\]</div>

<p><strong>第 3 步 – 偏差校正：</strong></p>



<div class="arithmatex">\[\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}\]</div>

<p><strong>第 4 步 – 参数更新：</strong></p>



<div class="arithmatex">\[\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\]</div>

<p><strong>典型取值：</strong><span class="arithmatex">\(\beta_1 = 0.9\)</span>，<span class="arithmatex">\(\beta_2 = 0.95\)</span> 或 <span class="arithmatex">\(0.999\)</span>，<span class="arithmatex">\(\epsilon = 10^{-8}\)</span>，<span class="arithmatex">\(\eta = 10^{-4}\)</span> 至 <span class="arithmatex">\(10^{-5}\)</span>。</p>
</div>

<div class="admonition tip"><p class="admonition-title">每一项的作用</p>
<ul>
<li><p><span class="arithmatex">\(m_t\)</span>（<strong>动量</strong>）：梯度的指数移动平均，可平滑噪声梯度估计。<span class="arithmatex">\(\beta_1 = 0.9\)</span> 意味着当前梯度贡献 10%，历史贡献 90%。</p></li>
<li><p><span class="arithmatex">\(v_t\)</span>（<strong>自适应学习率</strong>）：梯度平方的 EMA。梯度持续较大的参数获得较小的有效学习率（<span class="arithmatex">\(\eta / \sqrt{v_t}\)</span>）；梯度较小的参数获得较大的有效学习率。这是应对各层梯度尺度差异的关键。</p></li>
<li><p><span class="arithmatex">\(\hat{m}_t, \hat{v}_t\)</span>（<strong>偏差校正</strong>）：在 <span class="arithmatex">\(t=1\)</span> 时，<span class="arithmatex">\(m_1 = (1-\beta_1)g_1\)</span> 远小于真实均值。除以 <span class="arithmatex">\((1-\beta_1^t)\)</span> 可以校正这种初始化偏差。如果不做校正，早期更新步长会过小。</p></li>
<li><p><span class="arithmatex">\(\epsilon\)</span>（<strong>数值稳定性</strong>）：防止除零；同时也作为有效学习率的下限。</p></li>
</ul>
</div>

### 1.5.4　AdamW — 解耦权重衰减（Decoupled Weight Decay, AdamW） {#adamw-decoupled-weight-decay}

<p>AdamW <a class="book-citation" id="cite-loshchilov2019adamw-1" href="../../references.md#ref-loshchilov2019adamw" aria-label="参考文献 238">[238]</a> 修正了权重衰减与自适应优化器交互时一个细微但重要的问题。</p>

<div class="admonition note"><p class="admonition-title">为何在 Adam 中 L2 正则 <span class="arithmatex">\(\neq\)</span> 权重衰减</p>
<p>在 L2 正则化下，损失变为 <span class="arithmatex">\(\mathcal{L} + \frac{\lambda}{2}\|\theta\|^2\)</span>，因此梯度为 <span class="arithmatex">\(g_t + \lambda \theta_t\)</span>。在 Adam 中，这部分正则化梯度会<em>被自适应因子</em> <span class="arithmatex">\(1/\sqrt{\hat{v}_t}\)</span> <em>缩放</em>：</p>



<div class="arithmatex">\[\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t + \lambda \theta_t}{\sqrt{\hat{v}_t} + \epsilon}\]</div>

<p><span class="arithmatex">\(v_t\)</span> 较大（梯度方差大）的参数受到的正则化反而<em>更弱</em>。这并不是我们想要的——权重衰减本应是均匀的。</p>
</div>

<div class="admonition note"><p class="admonition-title">AdamW – 解耦权重衰减</p>
<p>AdamW（Loshchilov &amp; Hutter, 2017）将权重衰减<em>直接</em>作用于参数本身，置于自适应缩放之外：</p>



<div class="arithmatex">\[\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  - \eta \lambda \theta_t\]</div>

<p>权重衰减项 <span class="arithmatex">\(\eta \lambda \theta_t\)</span> 不再被 <span class="arithmatex">\(\sqrt{\hat{v}_t}\)</span> 缩放。这样无论参数的梯度历史如何，所有参数都能获得均匀的正则化。</p>

<p><strong>典型取值：</strong>LLM 训练时 <span class="arithmatex">\(\lambda = 0.1\)</span>。</p>
</div>

<div class="admonition warning"><p class="admonition-title">对 LLM 始终使用 AdamW – 不要用纯 Adam</p>
<p>Adam 与 AdamW 的差别虽细微但很重要。使用 Adam + L2 时，梯度方差小的参数（如 bias、LayerNorm 参数）受到更强的权重衰减，梯度方差大的参数（如 attention 权重）受到的衰减更弱。AdamW 才能提供我们想要的均匀正则化。大多数框架默认使用 AdamW；务必确认你的优化器类。</p>
</div>

### 1.5.5　Muon：超越 AdamW {#sec:muon-optimizer}

<p>近十年来，AdamW 一直是神经网络训练无可争议的默认优化器。Muon <a class="book-citation" id="cite-liu2025muon-1" href="../../references.md#ref-liu2025muon" aria-label="参考文献 224">[224]</a>（Liu 等人，2025）是第一个严肃挑战这一统治地位的优化器。其核心洞见是：Adam 累积的动量缓冲区携带的方向信息会被自适应缩放步骤掩盖——而在应用动量之前先对其做<em>正交化</em>，能产生根本更好的更新方向。</p>

<div class="admonition note"><p class="admonition-title">Muon：正交化动量</p>
<p>Muon 用两步流程取代隐藏层权重矩阵上的标准 Adam 更新：</p>

<ol>
<li><p><strong>正交化动量</strong>：对动量缓冲区 <span class="arithmatex">\(M_t\)</span> 应用 Newton–Schulz 迭代，得到一个近似正交的矩阵 <span class="arithmatex">\(\tilde{M}_t = \text{NS}(M_t)\)</span>。</p></li>
<li><p><strong>重新缩放以匹配 AdamW 的 RMS</strong>：缩放 <span class="arithmatex">\(\tilde{M}_t\)</span>，使其均方根与对应 AdamW 更新的 RMS 一致，确保有效步长具有可比性。</p></li>
</ol>

<p>最终的更新直接作用于权重矩阵。Embedding 层、输出投影与标量参数仍继续使用 AdamW。</p>
</div>

<p>Newton–Schulz 迭代是一个矩阵多项式，收敛于 <span class="arithmatex">\(M_t\)</span> 极分解中的正交因子。具体而言，从 <span class="arithmatex">\(X_0 = M_t / \|M_t\|_F\)</span> 出发，迭代</p>



<div class="arithmatex">\[X_{k+1} = \frac{3}{2} X_k - \frac{1}{2} X_k X_k^\top X_k\]</div>

<p>会快速收敛（通常 5–10 步）到一个满足 <span class="arithmatex">\(\tilde{M}_t^\top \tilde{M}_t \approx I\)</span> 的矩阵 <span class="arithmatex">\(\tilde{M}_t\)</span>。正交性保证了更新方向在所有奇异值方向上最大程度地展开——防止优化器把更新坍缩到低秩子空间上，而这是 Adam 在奇异值谱高度倾斜的权重矩阵上的一个已知失败模式。</p>

##### 实证效率。 {#heading-1-82}

<p>Muon 声称相对 AdamW 约有 <strong>2<span class="arithmatex">\(\times\)</span> 的算力效率</strong>：以大约一半的梯度步数达到相同的验证损失。这不是边际改进——它代表着优化器在损失景观上样本效率的质变。</p>

##### 前沿模型中的采用。 {#heading-1-83}

<p>在大型预训练中，天平明显正在向 Muon 倾斜：</p>

<ul>
<li><p><strong>GLM-4.5 / GLM-5</strong>（智谱 AI） <a class="book-citation" id="cite-glm45_2025-1" href="../../references.md#ref-glm45_2025" aria-label="参考文献 434">[434]</a><a class="book-citation" id="cite-glm5_2026-1" href="../../references.md#ref-glm5_2026" aria-label="参考文献 435">[435]</a>：预训练采用 Muon</p></li>
<li><p><strong>Kimi K2</strong>（月之暗面） <a class="book-citation" id="cite-kimik2_2025-1" href="../../references.md#ref-kimik2_2025" aria-label="参考文献 17">[17]</a>：使用 <strong>MuonClip</strong>——一个加入 <em>QK-Clip</em> 的变体，通过重新缩放 query 与 key 投影来限制 attention logits 并防止损失尖峰。Kimi K2 在超过 15.5T token 上训练且零损失尖峰，是该规模模型的稳定性纪录。</p></li>
<li><p><strong>DeepSeek-V4</strong>（2026） <a class="book-citation" id="cite-deepseekv4_2026-1" href="../../references.md#ref-deepseekv4_2026" aria-label="参考文献 71">[71]</a>：预训练采用 Muon</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">AdamW 仍是安全默认</p>
<p>AdamW 仍是 MAI-Thinking-1、Qwen、Llama 以及大多数生产微调流水线的首选优化器。Muon 的优势在权重矩阵的大规模预训练中最为显著；它在 LoRA 适配器、小型微调与 RL 训练（PPO/GRPO）上的行为则尚未被充分刻画。除非你正在进行大规模预训练，且有基础设施在具体架构上验证 Muon 的行为，否则请使用 AdamW。</p>
</div>

### 1.5.6　学习率 — 最重要的超参数 {#learning-rate-the-most-important-hyperparameter}

<div class="admonition note"><p class="admonition-title">各训练阶段的典型学习率</p>
<span id="table-1-inline-11"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>阶段</strong></th><th scope="col"><strong>典型学习率</strong></th><th scope="col"><strong>备注</strong></th></tr></thead>
<tbody><tr><td>从零预训练</td><td><span class="arithmatex">\(1\text{e-}4\)</span> 至 <span class="arithmatex">\(3\text{e-}4\)</span></td><td>大模型、大批量</td></tr>
<tr><td>继续预训练</td><td><span class="arithmatex">\(1\text{e-}5\)</span> 至 <span class="arithmatex">\(1\text{e-}4\)</span></td><td>更小学习率以保留已有知识</td></tr>
<tr><td>SFT（监督微调）</td><td><span class="arithmatex">\(1\text{e-}5\)</span> 至 <span class="arithmatex">\(2\text{e-}5\)</span></td><td>标准区间</td></tr>
<tr><td>LoRA 微调</td><td><span class="arithmatex">\(1\text{e-}4\)</span> 至 <span class="arithmatex">\(3\text{e-}4\)</span></td><td>适配器权重需用更高学习率</td></tr></tbody>
</table>
</div>

<p><em>有关 RL 学习率（PPO、DPO、GRPO）请参见 §<a href="../11-training-infrastructure/index.md#sec:rl-optimizer-config">11.15</a>。</em></p>
</div>

### 1.5.7　学习率预热（Warmup） {#learning-rate-warmup}

<div class="admonition note"><p class="admonition-title">为何需要预热</p>
<p>训练开始时，<span class="arithmatex">\(v_t\)</span>（二阶矩估计）被初始化为零。偏差校正后：<span class="arithmatex">\(\hat{v}_t = v_t / (1 - \beta_2^t)\)</span>。在 <span class="arithmatex">\(t=1\)</span>、<span class="arithmatex">\(\beta_2 = 0.999\)</span> 时：<span class="arithmatex">\(\hat{v}_1 = v_1 / (1 - 0.999) = 1000 v_1\)</span>。这意味着有效学习率是 <span class="arithmatex">\(\eta / \sqrt{1000 v_1}\)</span>，远小于预期。</p>

<p>另一方面，如果第一步的梯度异常大（在初始化时常见），二阶矩估计会被这个离群值主导，导致早期更新极不稳定。预热（Warmup）通过从极小的学习率开始、逐步增大来缓解这一问题，让 <span class="arithmatex">\(v_t\)</span> 有时间积累一个可靠的估计。</p>
</div>

<ul>
<li><p><strong>线性预热：</strong><span class="arithmatex">\(\eta_t = \eta_{\max} \times t / T_{\text{warmup}}\)</span></p></li>
<li><p><strong>典型预热时长：</strong>预训练为总步数的 1–5%；微调为 3–10%（更短的训练任务通常需要按比例更长的预热）</p></li>
<li><p><strong>SFT 场景：</strong>通常预热 50–200 步</p></li>
</ul>

### 1.5.8　学习率调度策略 {#learning-rate-schedules}

<figure id="figure-1-9"><a href="../../assets/figures/fig_010_fig10.png" target="_blank" rel="noopener" aria-label="查看图 1.9 原图" title="点击查看原图"><img src="../../assets/figures/fig_010_fig10.png" alt="常见学习率调度。所有调度都包含一个线性预热阶段。WSD（Warmup-Stable-Decay）正成为预训练的事实标准。" loading="lazy"></a><figcaption><span class="figure-number">图 1.9　</span>常见学习率调度。所有调度都包含一个线性预热阶段。WSD（Warmup-Stable-Decay）正成为预训练的事实标准。</figcaption></figure>

##### (a) 恒定调度（Constant）。 {#a-constant.}

<p>最简单的调度。适合短期微调，以避免学习率过度衰减。风险在于：缺乏退火意味着模型可能无法收敛到最尖锐的最小值。</p>

##### (b) 余弦衰减（Cosine Decay）。 {#b-cosine-decay.}



<div class="arithmatex">\[\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
  \left(1 + \cos\!\left(\frac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}} \pi\right)\right)\]</div>

<p>预训练与 SFT 的标准选择。平滑衰减避免了学习率的突变。<span class="arithmatex">\(\eta_{\min}\)</span> 通常取 <span class="arithmatex">\(\eta_{\max} / 10\)</span>。</p>

##### (c) 线性衰减（Linear Decay）。 {#c-linear-decay.}

<p>比余弦更简单，经验效果相近。当你希望任意步的学习率都可预测时优先选用。</p>

##### (d) WSD — 预热-稳定-衰减（Warmup-Stable-Decay, WSD）。 {#d-wsd-warmup-stable-decay.}

<p>大规模预训练的新标准 <a class="book-citation" id="cite-hu2024minicpm-1" href="../../references.md#ref-hu2024minicpm" aria-label="参考文献 150">[150]</a><a class="book-citation" id="cite-grattafiori2024llama3-1" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>。包含三个阶段：</p>

<ol>
<li><p><strong>预热：</strong>线性升至 <span class="arithmatex">\(\eta_{\max}\)</span>（占总步数的 1–5%）</p></li>
<li><p><strong>稳定：</strong>在训练的大部分时间保持恒定的 <span class="arithmatex">\(\eta_{\max}\)</span></p></li>
<li><p><strong>衰减：</strong>以快速余弦或线性方式衰减到 <span class="arithmatex">\(\eta_{\min}\)</span>（最后 10–20% 的步数）</p></li>
</ol>

<p>核心优势在于：稳定阶段允许在任意时刻打 checkpoint 并继续训练；而衰减阶段可在任一训练任务结束时再施加。</p>

##### (e) 带重启的余弦调度 — 带重启的随机梯度下降（Stochastic Gradient Descent with Warm Restarts, SGDR）。 {#e-cosine-with-restarts-sgdr.}

<p>周期性重启会将学习率重置为 <span class="arithmatex">\(\eta_{\max}\)</span>，有助于逃离局部极小值。在 LLM 中并不常用，更适合较小的模型。</p>

### 1.5.9　梯度裁剪（Gradient Clipping） {#gradient-clipping}

<div class="admonition note"><p class="admonition-title">梯度裁剪（Gradient Clipping）</p>
<p>当梯度的全局范数超过某阈值时，梯度裁剪会对其重新缩放：</p>



<div class="arithmatex">\[g_t \leftarrow g_t \cdot \min\!\left(1,\; \frac{\tau}{\|g_t\|_2}\right)\]</div>

<p>其中 <span class="arithmatex">\(\tau\)</span> 是 <code>max_grad_norm</code>（通常取 1.0）。</p>
</div>

<div class="admonition tip"><p class="admonition-title">梯度裁剪 vs. 降低学习率</p>
<p>梯度裁剪与降低学习率都能限制参数更新的幅度。区别在于：裁剪保留梯度的<em>方向</em>（只是缩放幅度），而较小的学习率会对所有更新一视同仁地缩放。裁剪更适合处理偶发的大梯度，同时不会拖慢正常训练步的速度。</p>
</div>

#### 综合实践：HuggingFace 优化器配置 {#putting-it-together-huggingface-optimizer-configuration}

<p>下面的代码片段展示了本节中各概念——带解耦权重衰减的 AdamW（§1.6.6）、带线性预热的余弦学习率调度（§1.6.7）和梯度裁剪（§1.6.8）——如何借助 HuggingFace <code>transformers</code> 库在实践中结合起来。</p>

<p class="code-caption"><span class="code-number">代码 1.4　</span>将 AdamW、余弦调度与梯度裁剪整合在一起的完整优化器配置。</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">TrainingArguments</span><span class="p">,</span> <span class="n">Trainer</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_cosine_schedule_with_warmup</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="c1"># --- 方案 1：使用 TrainingArguments（推荐）---</span>
<span class="n">training_args</span> <span class="o">=</span> <span class="n">TrainingArguments</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./checkpoints"</span><span class="p">,</span>

    <span class="c1"># AdamW 优化器（解耦权重衰减，§1.6.6）</span>
    <span class="n">optim</span><span class="o">=</span><span class="s2">"adamw_torch"</span><span class="p">,</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">2e-5</span><span class="p">,</span>           <span class="c1"># 预热后的峰值学习率</span>
    <span class="n">adam_beta1</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>               <span class="c1"># 一阶矩衰减</span>
    <span class="n">adam_beta2</span><span class="o">=</span><span class="mf">0.999</span><span class="p">,</span>             <span class="c1"># 二阶矩衰减</span>
    <span class="n">adam_epsilon</span><span class="o">=</span><span class="mf">1e-8</span><span class="p">,</span>            <span class="c1"># 数值稳定项</span>
    <span class="n">weight_decay</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span>           <span class="c1"># 解耦 L2 惩罚</span>

    <span class="c1"># 学习率调度（§1.6.7）</span>
    <span class="n">lr_scheduler_type</span><span class="o">=</span><span class="s2">"cosine"</span><span class="p">,</span>  <span class="c1"># 余弦衰减至 0</span>
    <span class="n">warmup_ratio</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>            <span class="c1"># 10% 步数用于线性预热</span>

    <span class="c1"># 梯度裁剪（§1.6.8）</span>
    <span class="n">max_grad_norm</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>           <span class="c1"># 按全局 L2 范数裁剪</span>

    <span class="c1"># 混合精度（§1.6.9）</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>                   <span class="c1"># 在 Ampere+ GPU 上使用 BFloat16</span>

    <span class="c1"># 训练时长</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>  <span class="c1"># 有效批量 = 8*4 = 32</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">Trainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">training_args</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>

<span class="c1"># --- 方案 2：手动控制（适用于自定义训练循环）---</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">torch.optim</span><span class="w"> </span><span class="kn">import</span> <span class="n">AdamW</span>

<span class="c1"># 权重衰减分组（不对 bias / norm 做正则化）</span>
<span class="n">no_decay</span> <span class="o">=</span> <span class="p">[</span><span class="s2">"bias"</span><span class="p">,</span> <span class="s2">"LayerNorm.weight"</span><span class="p">,</span> <span class="s2">"layernorm.weight"</span><span class="p">]</span>
<span class="n">param_groups</span> <span class="o">=</span> <span class="p">[</span>
    <span class="p">{</span>
        <span class="s2">"params"</span><span class="p">:</span> <span class="p">[</span><span class="n">p</span> <span class="k">for</span> <span class="n">n</span><span class="p">,</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">model</span><span class="o">.</span><span class="n">named_parameters</span><span class="p">()</span>
                   <span class="k">if</span> <span class="ow">not</span> <span class="nb">any</span><span class="p">(</span><span class="n">nd</span> <span class="ow">in</span> <span class="n">n</span> <span class="k">for</span> <span class="n">nd</span> <span class="ow">in</span> <span class="n">no_decay</span><span class="p">)],</span>
        <span class="s2">"weight_decay"</span><span class="p">:</span> <span class="mf">0.01</span><span class="p">,</span>
    <span class="p">},</span>
    <span class="p">{</span>
        <span class="s2">"params"</span><span class="p">:</span> <span class="p">[</span><span class="n">p</span> <span class="k">for</span> <span class="n">n</span><span class="p">,</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">model</span><span class="o">.</span><span class="n">named_parameters</span><span class="p">()</span>
                   <span class="k">if</span> <span class="nb">any</span><span class="p">(</span><span class="n">nd</span> <span class="ow">in</span> <span class="n">n</span> <span class="k">for</span> <span class="n">nd</span> <span class="ow">in</span> <span class="n">no_decay</span><span class="p">)],</span>
        <span class="s2">"weight_decay"</span><span class="p">:</span> <span class="mf">0.0</span><span class="p">,</span>
    <span class="p">},</span>
<span class="p">]</span>

<span class="n">optimizer</span> <span class="o">=</span> <span class="n">AdamW</span><span class="p">(</span><span class="n">param_groups</span><span class="p">,</span> <span class="n">lr</span><span class="o">=</span><span class="mf">2e-5</span><span class="p">,</span> <span class="n">betas</span><span class="o">=</span><span class="p">(</span><span class="mf">0.9</span><span class="p">,</span> <span class="mf">0.999</span><span class="p">))</span>

<span class="c1"># 带线性预热的余弦调度</span>
<span class="n">total_steps</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">train_dataloader</span><span class="p">)</span> <span class="o">*</span> <span class="n">num_epochs</span>
<span class="n">warmup_steps</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="mf">0.1</span> <span class="o">*</span> <span class="n">total_steps</span><span class="p">)</span>
<span class="n">scheduler</span> <span class="o">=</span> <span class="n">get_cosine_schedule_with_warmup</span><span class="p">(</span>
    <span class="n">optimizer</span><span class="p">,</span>
    <span class="n">num_warmup_steps</span><span class="o">=</span><span class="n">warmup_steps</span><span class="p">,</span>
    <span class="n">num_training_steps</span><span class="o">=</span><span class="n">total_steps</span><span class="p">,</span>
<span class="p">)</span>

<span class="c1"># 带梯度裁剪的训练循环</span>
<span class="k">for</span> <span class="n">batch</span> <span class="ow">in</span> <span class="n">train_dataloader</span><span class="p">:</span>
    <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="o">**</span><span class="n">batch</span><span class="p">)</span>
    <span class="n">loss</span> <span class="o">=</span> <span class="n">outputs</span><span class="o">.</span><span class="n">loss</span>
    <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>

    <span class="c1"># 在 optimizer.step() 之前裁剪梯度</span>
    <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">clip_grad_norm_</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">max_norm</span><span class="o">=</span><span class="mf">1.0</span><span class="p">)</span>

    <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
    <span class="n">scheduler</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
    <span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>
</code></pre></div>

<div class="admonition note"><p class="admonition-title">实践要点</p>
<ul>
<li><p><strong>权重衰减排除</strong>：bias 项和 LayerNorm 权重不应被正则化——它们参数很少，对其加正则反而损害性能 <a class="book-citation" id="cite-loshchilov2019adamw-2" href="../../references.md#ref-loshchilov2019adamw" aria-label="参考文献 238">[238]</a>。</p></li>
<li><p><strong>预热比例</strong>：通常为总步数的 5–10%；预热不足配合高学习率会使早期训练不稳定。</p></li>
<li><p><strong>梯度累积</strong>：在显存受限时模拟更大批量；裁剪作用于<em>累积后</em>的梯度。</p></li>
<li><p><strong>BF16 vs. FP16</strong>：在 Ampere+ GPU 上优先 <code>bf16=True</code>（更宽的动态范围可免去损失缩放）；老硬件回退到 <code>fp16=True</code>。</p></li>
</ul>
</div>

### 1.5.10　混合精度训练（Mixed Precision Training） {#mixed-precision-training}

<div class="admonition note"><p class="admonition-title">BF16 vs. FP16</p>
<span id="table-1-inline-12"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>格式</strong></th><th scope="col"><strong>指数位数</strong></th><th scope="col"><strong>尾数位数</strong></th><th scope="col"><strong>动态范围</strong></th></tr></thead>
<tbody><tr><td>FP32</td><td>8</td><td>23</td><td><span class="arithmatex">\(\sim 10^{-38}\)</span> to <span class="arithmatex">\(10^{38}\)</span></td></tr>
<tr><td>BF16</td><td>8</td><td>7</td><td>与 FP32 相同（指数位相同）</td></tr>
<tr><td>FP16</td><td>5</td><td>10</td><td><span class="arithmatex">\(\sim 6 \times 10^{-5}\)</span> to <span class="arithmatex">\(65504\)</span></td></tr></tbody>
</table>
</div>
</div>

<div class="admonition tip"><p class="admonition-title">选 BF16 而不是 FP16：在 LLM 训练中范围比精度更重要</p>
<p>BF16 与 FP32 拥有相同的指数范围，能表示同样范围的数值（只是尾数精度更低）。FP16 的动态范围则小得多——超过 65504 的梯度或激活值会溢出（NaN/Inf）。这就是为什么 FP16 训练需要<em>损失缩放（Loss Scaling）</em>（把损失乘以一个大常数以让梯度保持在 FP16 范围内），而 BF16 训练通常不需要。A100 与 H100 原生支持 BF16；除非有特殊理由，否则请使用 BF16。</p>
</div>

##### 损失缩放（Loss Scaling，仅 FP16）。 {#loss-scaling-fp16-only.}

<ol>
<li><p>将损失乘以缩放因子 <span class="arithmatex">\(S\)</span>（例如 <span class="arithmatex">\(S = 2^{15}\)</span>）</p></li>
<li><p>用 FP16 计算梯度（梯度被 <span class="arithmatex">\(S\)</span> 缩放）</p></li>
<li><p>在 optimizer 步骤前将梯度除以 <span class="arithmatex">\(S\)</span></p></li>
<li><p>检查溢出（NaN/Inf）；若发现则跳过该步并减小 <span class="arithmatex">\(S\)</span></p></li>
<li><p>若连续 <span class="arithmatex">\(N\)</span> 步未溢出，则增大 <span class="arithmatex">\(S\)</span></p></li>
</ol>

##### FP32 主权重（FP32 Master Weights）。 {#fp32-master-weights.}

<p>在混合精度训练中，权重以 FP32 形式保存（主副本），仅在前向/反向传播时转换为 BF16/FP16。优化器步骤则在 FP32 下完成。这一点很重要，原因是：</p>

<ul>
<li><p>很小的梯度更新（<span class="arithmatex">\(\Delta\theta \ll \theta\)</span>）在 BF16 精度下会丢失（7 位尾数 <span class="arithmatex">\(\approx\)</span> 0.8% 的相对精度）</p></li>
<li><p>FP32 主权重可确保许多小更新在多步中的累积保持准确</p></li>
<li><p>显存代价：权重存储翻倍（FP32 + BF16 副本）</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">何时 FP32 主权重至关重要</p>
<p>FP32 主权重在以下情形下尤为重要：</p>

<ul>
<li><p>长时间训练（大量微小梯度更新需要累积）</p></li>
<li><p>学习率较小（每次更新相对权重幅度极小）</p></li>
</ul>

<p>对于学习率较大的短期 SFT，仅用 BF16（不保留 FP32 主权重）通常也能跑通且节省显存。但对于 RL 训练，FP32 主权重必不可少——参见 §<a href="../11-training-infrastructure/index.md#sec:rl-optimizer-config">11.15</a>。</p>
</div>

#### 混合精度实战：HuggingFace {#mixed-precision-in-practice-huggingface}

<p class="code-caption"><span class="code-number">代码 1.5　</span>使用 HuggingFace 与手写 PyTorch AMP 实现混合精度训练。</p>
<div class="highlight"><pre><span></span><code><span class="c1"># === HuggingFace TrainingArguments（最简方式）===</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">TrainingArguments</span>

<span class="c1"># Ampere+ GPU 上的 BF16（A100、H100、RTX 30xx/40xx）</span>
<span class="n">args_bf16</span> <span class="o">=</span> <span class="n">TrainingArguments</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./out"</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>               <span class="c1"># BF16 前向/反向；FP32 主权重</span>
    <span class="n">bf16_full_eval</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>     <span class="c1"># 评估时也使用 BF16</span>
    <span class="c1"># 无需 loss scaling —— BF16 的范围已与 FP32 等价</span>
<span class="p">)</span>

<span class="c1"># 老 GPU 上的 FP16（V100、T4、RTX 20xx）</span>
<span class="n">args_fp16</span> <span class="o">=</span> <span class="n">TrainingArguments</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./out"</span><span class="p">,</span>
    <span class="n">fp16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>               <span class="c1"># FP16 前向/反向</span>
    <span class="n">fp16_full_eval</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span>    <span class="c1"># 评估仍用 FP32 以保证精度</span>
    <span class="c1"># 损失缩放由 PyTorch GradScaler 自动完成</span>
<span class="p">)</span>

<span class="c1"># === 手写 PyTorch AMP（用于自定义训练循环）===</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="c1"># 初始化（PyTorch 2.x API）</span>
<span class="n">use_fp16</span> <span class="o">=</span> <span class="ow">not</span> <span class="n">torch</span><span class="o">.</span><span class="n">cuda</span><span class="o">.</span><span class="n">is_bf16_supported</span><span class="p">()</span>
<span class="n">scaler</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">amp</span><span class="o">.</span><span class="n">GradScaler</span><span class="p">(</span><span class="s2">"cuda"</span><span class="p">,</span> <span class="n">enabled</span><span class="o">=</span><span class="n">use_fp16</span><span class="p">)</span>  <span class="c1"># 仅 FP16 需要</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">optim</span><span class="o">.</span><span class="n">AdamW</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="n">lr</span><span class="o">=</span><span class="mf">2e-5</span><span class="p">)</span>
<span class="n">dtype</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">float16</span> <span class="k">if</span> <span class="n">use_fp16</span> <span class="k">else</span> <span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span>

<span class="k">for</span> <span class="n">batch</span> <span class="ow">in</span> <span class="n">train_dataloader</span><span class="p">:</span>
    <span class="n">optimizer</span><span class="o">.</span><span class="n">zero_grad</span><span class="p">()</span>

    <span class="c1"># autocast：以降精度执行前向</span>
    <span class="k">with</span> <span class="n">torch</span><span class="o">.</span><span class="n">autocast</span><span class="p">(</span><span class="s2">"cuda"</span><span class="p">,</span> <span class="n">dtype</span><span class="o">=</span><span class="n">dtype</span><span class="p">):</span>
        <span class="n">outputs</span> <span class="o">=</span> <span class="n">model</span><span class="p">(</span><span class="o">**</span><span class="n">batch</span><span class="p">)</span>
        <span class="n">loss</span> <span class="o">=</span> <span class="n">outputs</span><span class="o">.</span><span class="n">loss</span>

    <span class="k">if</span> <span class="n">use_fp16</span><span class="p">:</span>
        <span class="c1"># FP16 路径：缩放损失以防梯度下溢</span>
        <span class="n">scaler</span><span class="o">.</span><span class="n">scale</span><span class="p">(</span><span class="n">loss</span><span class="p">)</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">scaler</span><span class="o">.</span><span class="n">unscale_</span><span class="p">(</span><span class="n">optimizer</span><span class="p">)</span>          <span class="c1"># 裁剪前先反缩放</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">clip_grad_norm_</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="mf">1.0</span><span class="p">)</span>
        <span class="n">scaler</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">optimizer</span><span class="p">)</span>              <span class="c1"># 溢出时会跳过 step</span>
        <span class="n">scaler</span><span class="o">.</span><span class="n">update</span><span class="p">()</span>                     <span class="c1"># 调整缩放因子</span>
    <span class="k">else</span><span class="p">:</span>
        <span class="c1"># BF16 路径：无需缩放</span>
        <span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
        <span class="n">torch</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">utils</span><span class="o">.</span><span class="n">clip_grad_norm_</span><span class="p">(</span><span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">(),</span> <span class="mf">1.0</span><span class="p">)</span>
        <span class="n">optimizer</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>

    <span class="n">scheduler</span><span class="o">.</span><span class="n">step</span><span class="p">()</span>
</code></pre></div>

<div class="admonition tip"><p class="admonition-title">代码层面 BF16 与 FP16 的关键差异</p>
<ul>
<li><p><strong>BF16</strong>：只需用 <code>autocast(dtype=torch.bfloat16)</code> 包裹即可，无需 scaler。代码更简单，数值更稳定。</p></li>
<li><p><strong>FP16</strong>：需要 <code>GradScaler</code> 以防止梯度下溢。scaler 会动态调节乘子；若检测到溢出（NaN）就会跳过该 optimizer step 并降低缩放因子。</p></li>
<li><p><strong>梯度裁剪 + FP16</strong>：<em>必须</em>在 <code>clip_grad_norm_</code> 之前调用 <code>scaler.unscale_(optimizer)</code>，否则你裁剪的是被缩放后的梯度（阈值错误）。</p></li>
<li><p><strong>显存节省</strong>：激活显存减半（激活值以 16-bit 存储）；权重显存取决于是否保留 FP32 主副本。</p></li>
</ul>
</div>

##### FP8 与 FP4：下一代训练精度。 {#heading-1-98}

<p>DeepSeek-V3 <a class="book-citation" id="cite-deepseekv3-1" href="../../references.md#ref-deepseekv3" aria-label="参考文献 70">[70]</a> 证明：以 FP8（前向传播用 E4M3，反向传播用 E5M2）配合细粒度逐 tile 缩放训练 671B 参数模型，相对 BF16 仅损失不到 0.25%——而总训练成本约为 $5.6M。关键技术包括：用随机舍入（stochastic rounding）降低量化偏差、让缩放因子逐 tile 适应局部激活幅度，以及在量化敏感度最高的首尾层保持更高精度。NVIDIA 的 Nemotron 3 更进一步，采用 NVFP4（4-bit），通过在大约 15% 的层上维持 BF16，在超过 25T 训练 token 上证明了稳定性。其经济含义极为深远：精心的精度工程能把前沿训练成本降低 3–5<span class="arithmatex">\(\times\)</span>，让数十亿参数的训练不再只是最大实验室的专利。</p>

### 1.5.11　各训练阶段的优化器配置实践 {#practical-optimizer-settings-by-training-phase}

<div class="admonition note"><p class="admonition-title">优化器超参数参考表</p>
<span id="table-1-inline-13"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>阶段</strong></th><th scope="col"><strong>优化器</strong></th><th scope="col"><strong>学习率</strong></th><th scope="col"><strong>权重衰减</strong></th><th scope="col"><strong>预热</strong></th><th scope="col"><strong>调度</strong></th></tr></thead>
<tbody><tr><td>预训练</td><td>AdamW</td><td><span class="arithmatex">\(3\text{e-}4\)</span></td><td>0.1</td><td>2000 步</td><td>WSD 或 Cosine</td></tr>
<tr><td>SFT</td><td>AdamW</td><td><span class="arithmatex">\(2\text{e-}5\)</span></td><td>0.01</td><td>100 步</td><td>Cosine</td></tr>
<tr><td>LoRA SFT</td><td>AdamW</td><td><span class="arithmatex">\(2\text{e-}4\)</span></td><td>0.01</td><td>100 步</td><td>Cosine</td></tr></tbody>
</table>
</div>

<p><em>以上均使用：<span class="arithmatex">\(\beta_1{=}0.9\)</span>、<span class="arithmatex">\(\beta_2{=}0.95\)</span>、<span class="arithmatex">\(\epsilon{=}10^{-8}\)</span>、<code>max_grad_norm</code>=1.0、BF16。RL 相关设置参见 §<a href="../11-training-infrastructure/index.md#sec:rl-optimizer-config">11.15</a>。</em></p>
</div>

<div class="admonition example"><p class="admonition-title">诊断训练不稳定</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 监控以下指标以诊断优化器问题：</span>
<span class="c1"># 1. 梯度范数 —— 大多数时候应 &lt; max_grad_norm</span>
<span class="c1"># 2. 损失缩放因子（FP16）—— 应稳定，不应持续减小</span>
<span class="c1"># 3. 参数更新范数 —— 应远小于参数自身的范数</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="k">def</span><span class="w"> </span><span class="nf">log_optimizer_stats</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">optimizer</span><span class="p">,</span> <span class="n">step</span><span class="p">):</span>
    <span class="c1"># 梯度范数（裁剪前）</span>
    <span class="n">total_norm</span> <span class="o">=</span> <span class="mf">0.0</span>
    <span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">model</span><span class="o">.</span><span class="n">parameters</span><span class="p">():</span>
        <span class="k">if</span> <span class="n">p</span><span class="o">.</span><span class="n">grad</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">:</span>
            <span class="n">total_norm</span> <span class="o">+=</span> <span class="n">p</span><span class="o">.</span><span class="n">grad</span><span class="o">.</span><span class="n">data</span><span class="o">.</span><span class="n">norm</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span><span class="o">.</span><span class="n">item</span><span class="p">()</span> <span class="o">**</span> <span class="mi">2</span>
    <span class="n">total_norm</span> <span class="o">=</span> <span class="n">total_norm</span> <span class="o">**</span> <span class="mf">0.5</span>

    <span class="c1"># Adam 二阶矩统计（自适应学习率的代理指标）</span>
    <span class="n">v_norms</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="k">for</span> <span class="n">group</span> <span class="ow">in</span> <span class="n">optimizer</span><span class="o">.</span><span class="n">param_groups</span><span class="p">:</span>
        <span class="k">for</span> <span class="n">p</span> <span class="ow">in</span> <span class="n">group</span><span class="p">[</span><span class="s1">'params'</span><span class="p">]:</span>
            <span class="n">state</span> <span class="o">=</span> <span class="n">optimizer</span><span class="o">.</span><span class="n">state</span><span class="p">[</span><span class="n">p</span><span class="p">]</span>
            <span class="k">if</span> <span class="s1">'exp_avg_sq'</span> <span class="ow">in</span> <span class="n">state</span><span class="p">:</span>
                <span class="n">v_norms</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">state</span><span class="p">[</span><span class="s1">'exp_avg_sq'</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span><span class="o">.</span><span class="n">item</span><span class="p">())</span>

    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Step </span><span class="si">{</span><span class="n">step</span><span class="si">}</span><span class="s2">: grad_norm=</span><span class="si">{</span><span class="n">total_norm</span><span class="si">:</span><span class="s2">.3f</span><span class="si">}</span><span class="s2">, "</span>
          <span class="sa">f</span><span class="s2">"mean_v=</span><span class="si">{</span><span class="nb">sum</span><span class="p">(</span><span class="n">v_norms</span><span class="p">)</span><span class="o">/</span><span class="nb">len</span><span class="p">(</span><span class="n">v_norms</span><span class="p">)</span><span class="si">:</span><span class="s2">.6f</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

<span class="c1"># 红色警报：</span>
<span class="c1"># grad_norm 反复 &gt;&gt; 1.0 —— 降低学习率或加长预热</span>
<span class="c1"># grad_norm == 0.0 —— 梯度消失或 loss 写错</span>
<span class="c1"># loss_scale 持续下降 —— FP16 溢出，改用 BF16</span>
<span class="c1"># v 非常小 —— Adam 还没预热完，加长预热</span>
</code></pre></div>
</div>

<div class="admonition tip"><p class="admonition-title">学习率是最重要的超参数</p>
<p>实践中，把学习率调对比任何其他超参数都重要。LLM 微调的经验法则：</p>

<ul>
<li><p>从上表中的取值开始</p></li>
<li><p>若损失发散（初始下降后又上升）：学习率过高</p></li>
<li><p>若损失下降极慢且过早进入平台期：学习率过低</p></li>
<li><p>若损失不稳定（震荡）：学习率过高或预热过短</p></li>
</ul>

<p>第二重要的超参数是批大小（通过线性缩放规则影响梯度噪声和有效学习率）。其他都是次要的。</p>
</div>

## 1.6　FlashAttention——算法与硬件感知 {#flash-attention-algorithm-and-hardware-awareness}

<p>FlashAttention <a class="book-citation" id="cite-dao2022flashattention-1" href="../../references.md#ref-dao2022flashattention" aria-label="参考文献 65">[65]</a><a class="book-citation" id="cite-dao2023flashattention2-1" href="../../references.md#ref-dao2023flashattention2" aria-label="参考文献 64">[64]</a>是自 Transformer 本身以来深度学习领域最具影响力的算法创新之一。它不改变 Attention 的数学结果——计算出的输出与原始 Attention <em>完全相同</em>——但它重构了内存访问模式，让 GPU 上容量有限的高速 SRAM 承担所有重活，从而将高带宽内存（HBM）占用从 <span class="arithmatex">\(O(n^2)\)</span> 降至 <span class="arithmatex">\(O(n)\)</span>，并在典型工作负载上带来 2–4<span class="arithmatex">\(\times\)</span> 的端到端实际墙钟时间加速。</p>

### 1.6.1　标准 Attention 的内存问题 {#the-standard-attention-memory-problem}

<p>标准的缩放点积 Attention 为：</p>



<div class="arithmatex">\[\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V\]</div>

<div class="admonition note"><p class="admonition-title">标准 Attention 的内存复杂度</p>
<p>对于序列长度 <span class="arithmatex">\(n\)</span> 和头维度 <span class="arithmatex">\(d\)</span>：</p>

<ul>
<li><p><span class="arithmatex">\(Q, K, V \in \mathbb{R}^{n \times d}\)</span>：<span class="arithmatex">\(O(nd)\)</span> 内存</p></li>
<li><p><span class="arithmatex">\(S = QK^T \in \mathbb{R}^{n \times n}\)</span>：<strong><span class="arithmatex">\(O(n^2)\)</span> 内存</strong>——瓶颈所在</p></li>
<li><p><span class="arithmatex">\(P = \text{softmax}(S) \in \mathbb{R}^{n \times n}\)</span>：又一个 <span class="arithmatex">\(O(n^2)\)</span></p></li>
<li><p><span class="arithmatex">\(O = PV \in \mathbb{R}^{n \times d}\)</span>：<span class="arithmatex">\(O(nd)\)</span></p></li>
</ul>

<p>在 <span class="arithmatex">\(n=8192\)</span>、<span class="arithmatex">\(d=128\)</span>、BF16 条件下：仅 Attention 矩阵本身就达到 <span class="arithmatex">\(8192^2 \times 2 \approx 134\)</span> MB <em>每头</em>。若有 32 个头，单单一层的 Attention 分数就要占用 4.3 GB。</p>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 <span class="arithmatex">\(O(n^2)\)</span> 是灾难性的</p>
<p>Attention 矩阵必须写入 HBM（对长序列而言它放不进 SRAM），随后再读出来做 Softmax，然后再读一次用于 <span class="arithmatex">\(PV\)</span> 乘法。每一次 HBM 往返都很慢。当 <span class="arithmatex">\(n=32768\)</span>（32K 上下文）时，Attention 矩阵达 <span class="arithmatex">\(32768^2 \times 2 \approx 2\)</span> GB <em>每头</em>——完全无法存储。</p>
</div>

### 1.6.2　FlashAttention 的核心洞察——分块（Tiling）与在线 Softmax（Online Softmax） {#the-flash-attention-key-insight-tiling-and-online-softmax}

<p>核心洞察是：<strong>我们从不需要把完整的 <span class="arithmatex">\(n \times n\)</span> 矩阵一次性放入内存</strong>。如果使用<em>在线 Softmax</em>技巧，就可以按块逐步计算输出 <span class="arithmatex">\(O\)</span>。</p>

##### 在线 Softmax（Online Softmax）。 {#online-softmax.}

<p>回想一下，为了数值稳定性，Softmax 需要一个全局最大值：</p>



<div class="arithmatex">\[\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}, \quad m = \max_j x_j\]</div>

<p>诀窍在于：我们可以在处理新块时<em>更新</em>运行中的最大值和归一化因子，而无需在任何时刻物化整行。</p>

<div class="admonition note"><p class="admonition-title">在线 Softmax 更新规则</p>
<p>给定运行状态 <span class="arithmatex">\((m_{\text{old}}, \ell_{\text{old}}, O_{\text{old}})\)</span> 和一个新的分数块 <span class="arithmatex">\(s_{\text{new}}\)</span>：</p>

<ol>
<li><p><span class="arithmatex">\(m_{\text{new}} = \max(m_{\text{old}},\; \max(s_{\text{new}}))\)</span></p></li>
<li><p><span class="arithmatex">\(\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}}
        + \sum_j e^{s_{\text{new},j} - m_{\text{new}}}\)</span></p></li>
<li><p><span class="arithmatex">\(O_{\text{new}} = \frac{1}{\ell_{\text{new}}} \left(
        e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} \cdot O_{\text{old}}
        + e^{s_{\text{new}} - m_{\text{new}}} \cdot V_{\text{new}} \right)\)</span></p></li>
</ol>

<p>这在数学上等价于一次性对所有块计算 Softmax。</p>
</div>

### 1.6.3　FlashAttention 算法 {#the-flash-attention-algorithm}

<div class="admonition example"><p class="admonition-title">FlashAttention 前向传播——分块（Block Tiling）</p>
<p><strong>设定：</strong> SRAM 大小 <span class="arithmatex">\(M\)</span>。块大小 <span class="arithmatex">\(B_r = \lceil M / (4d) \rceil\)</span>，<span class="arithmatex">\(B_c = \min(\lceil M / (4d) \rceil, d)\)</span>。</p>

<ol>
<li><p>将 <span class="arithmatex">\(Q\)</span> 划分为 <span class="arithmatex">\(T_r = \lceil n / B_r \rceil\)</span> 块 <span class="arithmatex">\(Q_1, \ldots, Q_{T_r}\)</span></p></li>
<li><p>将 <span class="arithmatex">\(K, V\)</span> 划分为 <span class="arithmatex">\(T_c = \lceil n / B_c \rceil\)</span> 块 <span class="arithmatex">\(K_1, \ldots, K_{T_c}\)</span></p></li>
<li><p>初始化输出 <span class="arithmatex">\(O \in \mathbb{R}^{n \times d}\)</span>、运行最大值 <span class="arithmatex">\(m \in \mathbb{R}^n\)</span>、运行求和 <span class="arithmatex">\(\ell \in \mathbb{R}^n\)</span>（均在 HBM 中）</p></li>
<li><p>对 <span class="arithmatex">\(j = 1, \ldots, T_c\)</span> 进行 <strong>外循环</strong>：</p>

<ol>
<li><p>将 <span class="arithmatex">\(K_j, V_j\)</span> 从 HBM 加载到 SRAM</p></li>
<li><p>对 <span class="arithmatex">\(i = 1, \ldots, T_r\)</span> 进行 <strong>内循环</strong>：</p>

<ol>
<li><p>将 <span class="arithmatex">\(Q_i, O_i, m_i, \ell_i\)</span> 从 HBM 加载到 SRAM</p></li>
<li><p>计算 <span class="arithmatex">\(S_{ij} = Q_i K_j^T / \sqrt{d}\)</span>（保留在 SRAM 中）</p></li>
<li><p>应用在线 Softmax 更新得到新的 <span class="arithmatex">\(m_i, \ell_i, O_i\)</span></p></li>
<li><p>将 <span class="arithmatex">\(O_i, m_i, \ell_i\)</span> 写回 HBM</p></li>
</ol></li>
</ol></li>
<li><p>返回 <span class="arithmatex">\(O\)</span></p></li>
</ol>

<p><strong>关键：</strong> <span class="arithmatex">\(S_{ij}\)</span>（Attention 块）在 SRAM 中计算并丢弃。它<em>从不写入 HBM</em>。</p>
</div>

<div class="admonition note"><p class="admonition-title">FlashAttention 复杂度</p>
<span id="table-1-inline-14"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>标准 Attention</strong></th><th scope="col"><strong>FlashAttention</strong></th></tr></thead>
<tbody><tr><td>内存（HBM）</td><td><span class="arithmatex">\(O(n^2)\)</span></td><td><span class="arithmatex">\(O(n)\)</span></td></tr>
<tr><td>HBM 读写量</td><td><span class="arithmatex">\(O(n^2 d)\)</span></td><td><span class="arithmatex">\(O(n^2 d / M)\)</span></td></tr>
<tr><td>FLOPs</td><td><span class="arithmatex">\(O(n^2 d)\)</span></td><td><span class="arithmatex">\(O(n^2 d)\)</span>（相同）</td></tr>
<tr><td>加速比</td><td>1<span class="arithmatex">\(\times\)</span></td><td>2–4<span class="arithmatex">\(\times\)</span></td></tr></tbody>
</table>
</div>

<p>在前向传播中，总 FLOPs 仍为 <span class="arithmatex">\(O(n^2 d)\)</span>——与标准 Attention 完全相同。FlashAttention 的加速完全来自于削减缓慢的 HBM 流量，而不是减少运算。（反向传播由于重计算实际上执行了<em>更多</em> FLOPs，但墙钟时间仍然更短，因为节省下来的内存带宽占主导。）</p>
</div>

### 1.6.4　FlashAttention 2——更好的并行性 {#flash-attention-2-better-parallelism}

<p>FlashAttention 2 <a class="book-citation" id="cite-dao2023flashattention2-2" href="../../references.md#ref-dao2023flashattention2" aria-label="参考文献 64">[64]</a> 做出了三项关键改进：</p>

<ol>
<li><p><strong>减少非矩阵乘法 FLOPs：</strong> 原版 FA 在内循环中存在不必要的重缩放操作。FA2 重构循环以最小化这些操作。在 A100 上，Tensor Core 矩阵乘法比标量运算快约 16<span class="arithmatex">\(\times\)</span>，因此即便内循环中只有少量非矩阵乘法工作，也会成为延迟瓶颈。</p></li>
<li><p><strong>序列维度上更好的并行性：</strong> FA1 仅在 batch 和 head 上并行。FA2 还在 query 序列维度上并行，使长序列、小批量场景下的 GPU 利用率显著提升。</p></li>
<li><p><strong>因果掩码优化：</strong> 对于自回归（因果）Attention，大约一半的块是完全被掩码的。FA2 完全跳过这些块，相对于双向 Attention 带来约 2<span class="arithmatex">\(\times\)</span> 的加速。</p></li>
</ol>

### 1.6.5　FlashAttention 3——Hopper 架构 {#flash-attention-3-hopper-architecture}

<p>FlashAttention 3 <a class="book-citation" id="cite-shah2024flashattention3-1" href="../../references.md#ref-shah2024flashattention3" aria-label="参考文献 322">[322]</a> 专为 H100 设计，并利用了三项 Hopper 特有的特性：</p>

<ul>
<li><p><strong>TMA（Tensor Memory Accelerator，张量内存加速器）：</strong> H100 有一个专用硬件单元用于在 HBM 和 SRAM 之间进行异步批量数据搬移。FA3 利用 TMA 让数据加载与计算重叠，从而隐藏内存延迟。</p></li>
<li><p><strong>Warp 专用化（Warp-specialization）：</strong> FA3 给不同的 warp 分配不同角色（生产者 warp 通过 TMA 加载数据；消费者 warp 计算 MMA）。这是一种软件流水线技术，使内存系统和 Tensor Core 同时保持繁忙。</p></li>
<li><p><strong>FP8 支持：</strong> H100 以 BF16 两倍的吞吐量支持 FP8（E4M3/E5M2）Tensor Core 运算。FA3 支持 FP8 Attention，并通过逐块量化来保持精度。</p></li>
</ul>

<p>FA3 在 FP16 Attention 上可达 <strong>H100 理论峰值的 75%</strong>，相比之下 FA2 约为 35%。</p>

### 1.6.6　FlashAttention 4——Blackwell 架构 {#flash-attention-4-blackwell-architecture}

<p>FlashAttention 4 <a class="book-citation" id="cite-zadouri2026flashattention4-1" href="../../references.md#ref-zadouri2026flashattention4" aria-label="参考文献 415">[415]</a> 面向 NVIDIA 的 Blackwell GPU（B200/GB200），这类 GPU 将 Tensor Core 吞吐量提高到 2.25 PFLOP/s（BF16），但非矩阵乘法单元（指数运算、共享内存带宽）的扩展速度较慢。这种<em>非对称的硬件扩展</em>意味着瓶颈发生了转移：在 Blackwell 上，Attention 的限制因素不再是矩阵乘法，而是 Softmax 指数运算以及围绕它们的共享内存流量。</p>

<p>FA4 通过四项关键技术来应对这一问题：</p>

<ul>
<li><p><strong>完全异步的 MMA 流水线：</strong> Blackwell 的 MMA 指令是完全异步的（不像 Hopper 的 wgmma 仍会阻塞等待完成）。FA4 重新设计了流水线，在更大的分块尺寸上让 MMA、TMA 加载和 Softmax 重缩放相互重叠，使所有硬件单元保持饱和。</p></li>
<li><p><strong>软件模拟指数运算：</strong> FA4 不调用硬件的 <code>ex2</code> 单元（吞吐量瓶颈），而是用多项式近似在快得多的 Tensor Core 上模拟 <span class="arithmatex">\(e^x\)</span>。这相当于用额外的矩阵乘法指令换取消除指数单元的停顿。</p></li>
<li><p><strong>条件式 Softmax 重缩放：</strong> 标准 FlashAttention 每个分块都会重缩放运行 <span class="arithmatex">\(\max\)</span>。FA4 在新分块的最大值不超过当前运行最大值时（实际中很常见）跳过重缩放，从而节省寄存器搬移和同步屏障。</p></li>
<li><p><strong>Tensor Memory + 2-CTA MMA 模式（反向传播）：</strong> 反向传播使用 Blackwell 的<em>张量内存（Tensor Memory）</em>（一种比共享内存更大的、每 SM 私有的暂存区）以及 2-CTA 协作模式，在两个线程块簇之间融合 <span class="arithmatex">\(dQ\)</span> 累加，使共享内存往返次数减半。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">FA4 实现：CuTe-DSL</p>
<p>FA4 是首个用 <strong>CuTe-DSL</strong> 编写的 FlashAttention 版本，CuTe-DSL 是一种嵌入在 Python 中、面向 GPU 内核的领域特定语言（CUTLASS 4.x 的一部分）。CuTe-DSL 的编译速度比 C++ CUTLASS 模板快 20–30<span class="arithmatex">\(\times\)</span>，同时保留对寄存器分配和流水线调度的完全控制。这显著缩短了内核开发的迭代时间。</p>
</div>

##### 结果。 {#results.}

<p>在 B200 上，BF16、head 维度 128（因果，序列长度 8K）：</p>

<ul>
<li><p><strong>1613 TFLOP/s</strong>——达到 Blackwell 峰值利用率的 71%</p></li>
<li><p>比 cuDNN 9.13（NVIDIA 的专有融合内核）快 <strong>1.3<span class="arithmatex">\(\times\)</span></strong></p></li>
<li><p>在同一硬件上比 Triton 快 <strong>2.7<span class="arithmatex">\(\times\)</span></strong></p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">软硬件协同演化</p>
<p>FlashAttention 系列体现了一条关键原则：每一代 GPU 都会让瓶颈发生转移，因此需要新的算法思想而不仅是重新编译。A80 <span class="arithmatex">\(\to\)</span> 内存带宽受限（FA1/FA2：分块 + 重计算）。H100 <span class="arithmatex">\(\to\)</span> 数据搬移受限（FA3：TMA + warp 专用化）。B200 <span class="arithmatex">\(\to\)</span> 非矩阵乘法计算受限（FA4：软件模拟 exp + 条件式重缩放）。理解<em>硬件瓶颈究竟在哪</em>是编写高效内核的前提。</p>
</div>

## 1.7　预训练：最佳实践 {#sec:pretraining}

<p>预训练是 LLM 开发中代价最高的阶段——消耗数百万 GPU 小时，需要对数据、算力和超参数进行精心编排。本节提炼 Llama-3 <a class="book-citation" id="cite-grattafiori2024llama3-1--section-07" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>、Chinchilla <a class="book-citation" id="cite-hoffmann2022chinchilla-1" href="../../references.md#ref-hoffmann2022chinchilla" aria-label="参考文献 140">[140]</a> 和 GPT-4 <a class="book-citation" id="cite-openai2023gpt4-1" href="../../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a> 的关键经验。</p>

### 1.7.1　训练目标 {#training-objective}

<p>所有现代仅解码器 LLM 都使用<strong>因果语言建模（Causal Language Modeling，CLM）</strong>：</p>



<div class="arithmatex">\[\mathcal{L}_\text{CLM} = -\frac{1}{T}\sum_{t=1}^T \log P_\theta(x_t \mid x_{&lt;t})\]</div>

<p>这个简单的目标——在足够的数据和规模下——无需显式监督就能产生涌现能力（上下文学习、推理、指令遵循） <a class="book-citation" id="cite-brown2020language-1" href="../../references.md#ref-brown2020language" aria-label="参考文献 32">[32]</a>。</p>

### 1.7.2　数据流水线 {#data-pipeline}

<div class="admonition note"><p class="admonition-title">预训练数据配方</p>
<ul>
<li><p><strong>规模</strong>：前沿模型为 1–15 万亿 Token（Llama-3：15T Token）</p></li>
<li><p><strong>来源</strong>：网页抓取（80%）、代码（10%）、书籍/论文（5%）、精选数据（5%）</p></li>
<li><p><strong>去重</strong>：MinHash + 精确子串去重可降低记忆化 <a class="book-citation" id="cite-lee2022deduplicating-1" href="../../references.md#ref-lee2022deduplicating" aria-label="参考文献 203">[203]</a></p></li>
<li><p><strong>质量过滤</strong>：基于困惑度的分类器、启发式过滤器（长度、语种识别、毒性）</p></li>
<li><p><strong>数据混合</strong>：跨域温度加权采样；为推理能力提升代码与数学的权重</p></li>
</ul>
</div>

### 1.7.3　扩展律 {#scaling-laws}

<p>Hoffmann 等 <a class="book-citation" id="cite-hoffmann2022chinchilla-2" href="../../references.md#ref-hoffmann2022chinchilla" aria-label="参考文献 140">[140]</a>表明，算力最优的训练需要平衡模型大小 <span class="arithmatex">\(N\)</span> 和数据大小 <span class="arithmatex">\(D\)</span>：<span class="arithmatex">\(N_\text{opt} \propto C^{0.50}\)</span>，<span class="arithmatex">\(D_\text{opt} \propto C^{0.50}\)</span>。70B 模型的算力最优点约在 1.4T Token。实际中，模型通常被<em>过训练</em>（Token 数超过 Chinchilla 最优值），因为推理成本随模型大小而非训练 Token 数扩展——较小的过训练模型部署成本更低。</p>

### 1.7.4　关键超参数 {#key-hyperparameters}

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

### 1.7.5　常见失败模式 {#common-failure-modes}

<div class="admonition warning"><p class="admonition-title">预训练陷阱</p>
<ul>
<li><p><strong>损失尖峰</strong>：由坏数据批次或数值不稳定引发的损失突增。Llama-3 报告通过回滚检查点并跳过问题批次来处理。</p></li>
<li><p><strong>记忆化</strong>：模型逐字复述训练数据。修复：激进去重；监控抽取攻击。</p></li>
<li><p><strong>上下文长度</strong>：在短序列上训练却部署到长上下文会失败。应在长文档上做继续预训练 + RoPE 缩放。</p></li>
</ul>
</div>

### 1.7.6　中期训练：为强化学习做准备 {#sec:mid-training}

<p>在原始预训练与后训练之间，一个新的流水线阶段已经浮现。<em>中期训练（Mid-training）</em>（有时也叫“继续预训练（continued pretraining）”或“退火（annealing）”）在高质量精选数据上提高 STEM、数学与代码的权重，同时把上下文长度扩展到 128K–256K token。它的目的不是传授新知识，而是<em>让模型为 RL 做好准备</em>：植入验证、回溯、子目标分解这些认知行为，供之后的强化学习放大。</p>

<div class="admonition warning"><p class="admonition-title">中期训练决定 RL 就绪度</p>
<p>OctoThinker <a class="book-citation" id="cite-octothinker2025-1" href="../../references.md#ref-octothinker2025" aria-label="参考文献 236">[236]</a> 证明：中期训练的数据构成决定了 base 模型是否会对 RL 产生响应。同一套 GRPO 配方在 Qwen（其中期训练富含推理内容）上带来了显著的推理提升，在 Llama（缺少此类内容）上却完全失败。差异不在模型大小或架构——而在于 base 模型是否已经展现出 RL 可以强化的自我纠正行为。</p>
</div>

<p>典型的中期训练使用 100B–500B token 的精选高质量数据，搭配余弦衰减的学习率。Llama 3 的“退火尾巴（annealing tail）”、MiMo 的三阶段混合（把数学+代码提高到 70%），以及 MAI-Thinking-1 对 STEM 的显式加权，都是这一模式的实例。</p>

## 1.8　监督微调（Supervised Fine-Tuning，SFT） {#sec:sft}

<p>SFT 通过在精选的「提示——回复」对上训练，将预训练语言模型转化为指令遵循助手。它是从原始语言建模到 RLHF 的桥梁。</p>

### 1.8.1　SFT 目标 {#sft-objective}

<p>损失与 CLM 相同，但只在<strong>回复 Token</strong> 上计算：</p>



<div class="arithmatex">\[\mathcal{L}_\text{SFT} = -\frac{1}{|y|}\sum_{t=1}^{|y|} \log P_\theta(y_t \mid x_\text{prompt}, y_{&lt;t})\]</div>

<p>提示 Token 提供上下文但不接收梯度（标签设为 <span class="arithmatex">\(-100\)</span>）。</p>

### 1.8.2　数据质量：LIMA 原则 {#data-quality-the-lima-principle}

<p>Zhou 等 <a class="book-citation" id="cite-zhou2023lima-1" href="../../references.md#ref-zhou2023lima" aria-label="参考文献 439">[439]</a>证明，1,000 个精心策划的样例可以匹敌在 50K+ 嘈杂样例上训练的模型。关键要求：</p>

<ul>
<li><p><strong>多样性</strong>：涵盖问答、摘要、代码、数学、创意写作、多轮对话</p></li>
<li><p><strong>正确性</strong>：每条回复必须事实准确且格式良好</p></li>
<li><p><strong>长度平衡</strong>：混合短（单句）和长（多段落）回复</p></li>
<li><p><strong>去污染</strong>：移除与评估基准的重叠</p></li>
</ul>

### 1.8.3　训练配置 {#training-configuration}

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

### 1.8.4　高效训练方案 {#efficient-training-solutions}

<p>标准的 HuggingFace 训练存在显著的性能浪费。多个库为 SFT 工作负载提供了即插即用的效率提升：</p>

##### Liger Kernel <a class="book-citation" id="cite-hsu2024liger-1" href="../../references.md#ref-hsu2024liger" aria-label="参考文献 145">[145]</a>。 {#liger-kernel-.}

<p>LinkedIn 开源的一组 <strong>Triton 融合内核</strong>，在训练时替换标准的 PyTorch 算子。关键融合包括：</p>

<ul>
<li><p><strong>融合交叉熵</strong>：将最后的线性投影、Softmax 和损失计算合并为单个内核——避免物化完整的 <span class="arithmatex">\((\text{batch} \times \text{seq} \times \text{vocab})\)</span> logit 张量。</p></li>
<li><p><strong>融合 RMSNorm / SwiGLU / RoPE</strong>：消除常见 LLM 构建块的中间内存分配。</p></li>
<li><p><strong>分块操作</strong>：将大张量按块处理，从而限制峰值内存。</p></li>
</ul>

<p><strong>效果</strong>：仅需一行集成代码（<code>apply_liger_kernel_to_llama()</code>）即可获得 20% 的吞吐量提升和最高 60% 的内存降低。兼容 FSDP、DeepSpeed 和 LoRA。</p>

##### Unsloth <a class="book-citation" id="cite-unsloth2024-1" href="../../references.md#ref-unsloth2024" aria-label="参考文献 132">[132]</a>。 {#unsloth-.}

<p>一个专注于微调的库，将<strong>自定义 CUDA/Triton 内核</strong>与激进的内存优化相结合：</p>

<ul>
<li><p>对 LoRA 层进行手工反向传播（避免 autograd 开销）。</p></li>
<li><p>带融合反量化的 4-bit QLoRA——在单张 48 GB GPU 上训练 70B 模型。</p></li>
<li><p>针对每种架构（Llama、Mistral、Qwen、Gemma）的智能 RoPE 与 Attention 内核融合。</p></li>
</ul>

<p><strong>效果</strong>：比原版 HuggingFace + PEFT 快 2–5<span class="arithmatex">\(\times\)</span>，显存占用降低 60–70%。对单 GPU 与消费级硬件工作流尤为重要。</p>

##### torchtune <a class="book-citation" id="cite-torchtune2024-1" href="../../references.md#ref-torchtune2024" aria-label="参考文献 349">[349]</a>。 {#torchtune-.}

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

### 1.8.5　最佳实践 {#best-practices}

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
