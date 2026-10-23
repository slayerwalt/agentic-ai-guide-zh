# 6.9　DPO 扩展与变体 {#sec:dpo-variants}

<p>直接偏好优化（Direct Preference Optimization, DPO）通过推导 reward 函数与最优 policy 之间的解析映射，把 RLHF 重新表述为一个有监督学习问题。标准 DPO loss 为：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(q,y_w,y_l)}\!\left[
    \log \sigma\!\left(
      \beta \log \frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)}
      - \beta \log \frac{\pi_\theta(y_l|q)}{\pi_{\text{ref}}(y_l|q)}
    \right)
  \right],\]</div>

<p>其中 <span class="arithmatex">\(y_w\)</span> 为被偏好（获胜）的回答，<span class="arithmatex">\(y_l\)</span> 为被拒（失败）的回答，<span class="arithmatex">\(\beta\)</span> 控制 KL 惩罚强度。下面的小节涵盖最重要的扩展与变体。</p>

## 6.9.1　f-DPO——广义 f-Divergence DPO {#sec:fdpo}

<div class="admonition tip"><p class="admonition-title">超越反向 KL</p>
<p>标准 DPO 使用反向 KL 散度作为 policy 与 reference 之间的正则项。反向 KL 是<em>寻峰</em>的：它倾向于把概率质量集中在少数高 reward 回答上。前向 KL 则是<em>覆盖</em>的：它把概率质量分散去覆盖所有合理回答。f-DPO <a class="book-citation" id="cite-wang2023fdpo-1" href="../../references.md#ref-wang2023fdpo" aria-label="参考文献 363">[363]</a> 推广到任意 f-散度，允许实践者在这些行为间做权衡。</p>
</div>

<p>f-DPO loss 用 f-散度生成器的导数替换对数比：</p>



<div class="arithmatex">\[\mathcal{L}_{f\text{-DPO}} = -\mathbb{E}\!\left[
    f&#x27;\!\left(\frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)}\right)
    - f&#x27;\!\left(\frac{\pi_\theta(y_l|q)}{\pi_{\text{ref}}(y_l|q)}\right)
  \right],\]</div>

<p>其中 <span class="arithmatex">\(f&#x27;\)</span> 是 f-散度生成器函数的导数。</p>

<div class="admonition note"><p class="admonition-title">TRL 中的 f-Divergence 选项</p>
<ul>
<li><p><strong>reverse_kl</strong>：<span class="arithmatex">\(f&#x27;(u) = \log u\)</span>。即标准 DPO。寻峰。</p></li>
<li><p><strong>forward_kl</strong>：<span class="arithmatex">\(f&#x27;(u) = -1/u\)</span>。覆盖式。多样性更好。</p></li>
<li><p><strong>js_divergence</strong>：<span class="arithmatex">\(f&#x27;(u) = \log(2u/(u+1))\)</span>。寻峰/覆盖之间的平衡。</p></li>
<li><p><strong>alpha_divergence</strong>：<span class="arithmatex">\(f&#x27;(u) = u^{\alpha-1}\)</span>。在前向 KL 与反向 KL 之间插值。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 f-DPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="c1"># Jensen-Shannon 散度（平衡）</span>
<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">f_divergence_type</span><span class="o">=</span><span class="s2">"js_divergence"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="c1"># Alpha 散度（alpha=0：前向 KL；alpha=1：反向 KL）</span>
<span class="n">config_alpha</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">f_divergence_type</span><span class="o">=</span><span class="s2">"alpha_divergence"</span><span class="p">,</span>
    <span class="n">f_alpha_divergence_coef</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span>   <span class="c1"># alpha 参数</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 f-DPO</p>
<ul>
<li><p>当你想在多样性与质量之间取得平衡时，使用 <strong>JS 散度</strong>。</p></li>
<li><p>对多样性至关重要的创意任务，使用<strong>前向 KL</strong>。</p></li>
<li><p>对有唯一正确答案的任务，使用<strong>反向 KL</strong>（即标准 DPO）。</p></li>
<li><p>使用<strong>alpha 散度</strong>可连续插值并精调这一权衡。</p></li>
</ul>
</div>

## 6.9.2　Robust DPO {#sec:robust-dpo}

<div class="admonition tip"><p class="admonition-title">偏好数据中的噪声标签</p>
<p>人类偏好标注是有噪声的。标注者会意见分歧、会出错，有时会把“被偏好/被拒”的标签搞反。标准 DPO 把所有标签视为 ground truth，可能导致模型对噪声过拟合。Robust DPO <a class="book-citation" id="cite-chowdhury2024robustdpo-1" href="../../references.md#ref-chowdhury2024robustdpo" aria-label="参考文献 56">[56]</a> 在已知噪声模型下解析性地对 loss 去偏。</p>
</div>

<p>假设每个标签以概率 <span class="arithmatex">\(\epsilon\)</span>（噪声率）被翻转。去偏后的 loss 为：</p>



<div class="arithmatex">\[\boxed{
    \mathcal{L}_{\text{robust}} =
    \frac{(1-\epsilon)\,\mathcal{L}_{\text{DPO}}(y_w, y_l)
          - \epsilon\,\mathcal{L}_{\text{DPO}}(y_l, y_w)}{1 - 2\epsilon},
  }\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{L}_{\text{DPO}}(y_w, y_l)\)</span> 是把 <span class="arithmatex">\(y_w\)</span> 视为被偏好的标准 DPO loss，而 <span class="arithmatex">\(\mathcal{L}_{\text{DPO}}(y_l, y_w)\)</span> 是把标签翻转后的 loss。该修正去除了标签噪声引入的偏置。</p>

<div class="admonition note"><p class="admonition-title">Robust DPO 的直觉</p>
<p>该公式是一个线性组合，把翻转标签的贡献“减掉”。当 <span class="arithmatex">\(\epsilon=0\)</span> 时，它退化为标准 DPO；当 <span class="arithmatex">\(\epsilon=0.5\)</span> 时，分母趋于零——标签纯属噪声，无法学习。实际中，<span class="arithmatex">\(\epsilon \in [0.05, 0.2]\)</span> 覆盖了大多数真实标注噪声水平。</p>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 Robust DPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"robust"</span><span class="p">,</span>
    <span class="n">label_smoothing</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>   <span class="c1"># 对应 epsilon = 0.1</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

## 6.9.3　TR-DPO——信任域 DPO {#sec:tr-dpo}

<div class="admonition tip"><p class="admonition-title">过期 Reference 模型问题</p>
<p>标准 DPO 在整个训练过程中使用固定的 reference 模型 <span class="arithmatex">\(\pi_{\text{ref}}\)</span>。随着 policy <span class="arithmatex">\(\pi_\theta\)</span> 改进，KL 惩罚 <span class="arithmatex">\(\beta \log(\pi_\theta/\pi_{\text{ref}})\)</span> 不断增长，最终会主导 loss 并阻止进一步改进。TR-DPO <a class="book-citation" id="cite-gorbatenko2024trdpo-1" href="../../references.md#ref-gorbatenko2024trdpo" aria-label="参考文献 116">[116]</a> 周期性地更新 reference 模型，使其跟随当前 policy。</p>
</div>

<p>TR-DPO 使用指数滑动平均（EMA）更新 reference 模型：</p>



<div class="arithmatex">\[\pi_{\text{ref}}^{(t+1)} \leftarrow
    \alpha \cdot \pi_\theta^{(t)} + (1-\alpha) \cdot \pi_{\text{ref}}^{(t)},\]</div>

<p>其中 <span class="arithmatex">\(\alpha \in (0,1)\)</span> 是混合系数。每 <span class="arithmatex">\(T_{\text{sync}}\)</span> 个 gradient 步执行一次。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 TR-DPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"sigmoid"</span><span class="p">,</span>        <span class="c1"># 标准 DPO loss</span>
    <span class="n">sync_ref_model</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>        <span class="c1"># 启用 TR-DPO</span>
    <span class="n">ref_model_mixup_alpha</span><span class="o">=</span><span class="mf">0.6</span><span class="p">,</span>  <span class="c1"># alpha：混入当前 policy 的比例</span>
    <span class="n">ref_model_sync_steps</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span>   <span class="c1"># T_sync：每 512 步同步一次</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 TR-DPO</p>
<ul>
<li><p>Policy 已偏离初始 reference 较远的长训练任务。</p></li>
<li><p>当观察到 DPO loss 因 KL 惩罚主导而过早平台化时。</p></li>
<li><p>从当前 policy 收集新偏好数据的迭代 DPO 流水线。</p></li>
<li><p><span class="arithmatex">\(\alpha\)</span> 接近 1 表示快更新 reference；接近 0 表示慢更新。</p></li>
</ul>
</div>

## 6.9.4　EXO——精确优化 {#sec:exo}

<div class="admonition tip"><p class="admonition-title">DPO 的 KL 方向问题</p>
<p>DPO 是在反向 KL 约束下求解最优 policy 而推导出的。然而，由此得到的 loss 实际上在 reward 空间中优化的是一个<em>前向</em> KL 目标，这个方向是错的。EXO <a class="book-citation" id="cite-ji2024exo-1" href="../../references.md#ref-ji2024exo" aria-label="参考文献 163">[163]</a> 通过使用反向 KL 概率匹配修正了这一点，这正是对齐在理论上正确的目标。</p>
</div>

<p>EXO 最小化模型分布与目标（reward 最优）分布之间的反向 KL：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{EXO}} = \mathbb{E}_{y \sim \pi_\theta}\!\left[
    \log \frac{\pi_\theta(y|q)}{p^*(y|q)}
  \right],\]</div>

<p>其中 <span class="arithmatex">\(p^*(y|q) \propto \pi_{\text{ref}}(y|q) \exp(r(y,q)/\beta)\)</span> 即最优 policy。实践中，EXO 利用可用的偏好对来近似该量：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{EXO}} \approx -\mathbb{E}\!\left[
    \log \sigma\!\left(
      \beta \log \frac{\pi_{\text{ref}}(y_w|q)}{\pi_\theta(y_w|q)}
      - \beta \log \frac{\pi_{\text{ref}}(y_l|q)}{\pi_\theta(y_l|q)}
    \right)
  \right].\]</div>

<p>注意：相对于 DPO，<span class="arithmatex">\(\pi_\theta\)</span> 与 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 的角色被<em>对调</em>了。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 EXO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"exo_pair"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

## 6.9.5　NCA——噪声对比对齐 {#sec:nca}

<div class="admonition tip"><p class="admonition-title">DPO 的似然崩塌</p>
<p>DPO 已知的失败模式之一是<em>似然崩塌</em>：模型学到了降低被拒回答的概率，但同时也降低了被选回答的概率（因为 loss 只在意它们的<em>差值</em>）。NCA <a class="book-citation" id="cite-chen2024nca-1" href="../../references.md#ref-chen2024nca" aria-label="参考文献 43">[43]</a> 增加了一个绝对似然项以防止这一点。</p>
</div>

<p>NCA 把对齐重新表述为噪声对比估计。Loss 包含三项：</p>



<div class="arithmatex">\[\boxed{
    \mathcal{L}_{\text{NCA}} =
      -\log \sigma(r_w)
      - \tfrac{1}{2}\log \sigma(-r_w)
      - \tfrac{1}{2}\log \sigma(-r_l),
  }\]</div>

<p>其中 <span class="arithmatex">\(r_y = \beta \log(\pi_\theta(y|q)/\pi_{\text{ref}}(y|q))\)</span> 是隐式 reward。第一项鼓励 <span class="arithmatex">\(y_w\)</span> 获得高 reward；第二、三项则同时惩罚 <span class="arithmatex">\(y_w\)</span> 与 <span class="arithmatex">\(y_l\)</span> 上的高 reward（防止崩塌）。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 NCA</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"nca_pair"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.01</span><span class="p">,</span>   <span class="c1"># beta 较小：绝对似然项占主导</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 NCA</p>
<ul>
<li><p>当观察到 DPO 训练过程中被选回答的概率在下降时。</p></li>
<li><p>对于看重绝对回答质量、而不仅仅是相对排序的任务。</p></li>
<li><p>使用较小的 <span class="arithmatex">\(\beta\)</span>（如 0.01）让绝对似然项获得更高权重。</p></li>
</ul>
</div>

## 6.9.6　SLiC-HF——序列似然校准 {#sec:slic}

<div class="admonition tip"><p class="admonition-title">作为简化替代的 Hinge Loss</p>
<p>DPO 中的 log-sigmoid loss 虽然平滑，但当 margin 较大时收敛缓慢。SLiC-HF <a class="book-citation" id="cite-zhao2023slichf-1" href="../../references.md#ref-zhao2023slichf" aria-label="参考文献 432">[432]</a> 使用 hinge loss：当 margin 超过阈值时为零，否则为线性。它更简单、更快，且效果出乎意料地有竞争力。</p>
</div>

<p>SLiC-HF loss 为：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{SLiC}} = \max\!\left(0,\;
    \delta - \beta\log\frac{\pi_\theta(y_w|q)}{\pi_{\text{ref}}(y_w|q)}
    + \beta\log\frac{\pi_\theta(y_l|q)}{\pi_{\text{ref}}(y_l|q)}
  \right),\]</div>

<p>其中 <span class="arithmatex">\(\delta\)</span> 是 margin 阈值。当模型在被选与被拒回答之间已经赋予 <span class="arithmatex">\(\delta\)</span> 大小的 margin 时，loss 为零。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 SLiC-HF</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"hinge"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

## 6.9.7　Iterative RPO——推理偏好优化 {#sec:rpo}

<div class="admonition tip"><p class="admonition-title">DPO 会“忘记”如何生成</p>
<p>标准 DPO 训练模型去<em>判别</em>被选与被拒回答的优劣。但对于推理任务，模型还需要<em>生成</em>正确的推理轨迹。一个能判别却不能生成的模型在推理时毫无用处。RPO 在被选回答上加入一个 NLL（负对数似然）项，确保模型学会生成它。</p>
</div>

<p>RPO loss 将 DPO 与 SFT 结合：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{RPO}} =
    \lambda_1 \mathcal{L}_{\text{DPO}}(y_w, y_l)
    + \lambda_2 \mathcal{L}_{\text{NLL}}(y_w),\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{L}_{\text{NLL}}(y_w) = -\log \pi_\theta(y_w|q)\)</span> 是作用于被选回答的标准语言建模 loss。</p>

<div class="admonition example"><p class="admonition-title">TRL 中的 Iterative RPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"sigmoid"</span><span class="p">,</span>            <span class="c1"># 标准 DPO loss</span>
    <span class="n">rpo_alpha</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>                  <span class="c1"># NLL 正则权重（RPO）</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>

<div class="admonition note"><p class="admonition-title">何时使用 RPO</p>
<ul>
<li><p>模型必须生成逐步解答的推理任务（数学、代码、逻辑）。</p></li>
<li><p>当 DPO 训练导致模型流畅度或生成质量下降时。</p></li>
<li><p>迭代式流水线：生成 rollout、打标签、用 RPO 训练、再重复。</p></li>
<li><p>NLL 项起到正则化作用，防止 policy 崩塌。</p></li>
</ul>
</div>

## 6.9.8　SimPO——简化版偏好优化 {#sec:simpo}

<div class="admonition tip"><p class="admonition-title">无参考模型的偏好学习</p>
<p>DPO 需要一个 reference 模型来计算隐式 reward，这会让显存翻倍并增加复杂度。SimPO <a class="book-citation" id="cite-meng2024simpo-1" href="../../references.md#ref-meng2024simpo" aria-label="参考文献 252">[252]</a> 取消了 reference 模型，转而用回答的<em>平均对数概率</em>作为隐式 reward，并通过长度归一化项防止模型偏好短回答。</p>
</div>

<p>SimPO 将隐式 reward 定义为：</p>



<div class="arithmatex">\[r_{\text{SimPO}}(y|q) = \frac{\beta}{|y|} \log \pi_\theta(y|q),\]</div>

<p>而 loss 为：</p>



<div class="arithmatex">\[\boxed{
    \mathcal{L}_{\text{SimPO}} = -\mathbb{E}\!\left[
      \log \sigma\!\left(
        \frac{\beta}{|y_w|}\log\pi_\theta(y_w|q)
        - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|q)
        - \gamma
      \right)
    \right],
  }\]</div>

<p>其中 <span class="arithmatex">\(\gamma &gt; 0\)</span> 是目标 reward margin，确保被选回答的 reward 至少比被拒回答高出 <span class="arithmatex">\(\gamma\)</span>。</p>

<div class="admonition note"><p class="admonition-title">SimPO vs DPO vs ORPO</p>
<ul>
<li><p><strong>DPO</strong>：使用 reference 模型；基于比率的隐式 reward。</p></li>
<li><p><strong>ORPO</strong>：无 reference；在 SFT loss 上加入 odds-ratio 项。</p></li>
<li><p><strong>SimPO</strong>：无 reference；长度归一化的对数概率 reward + margin。</p></li>
<li><p>SimPO 比 DPO 简单（无 reference 模型），且比 ORPO 更具理论性。</p></li>
<li><p>SimPO 中的长度归一化至关重要：若没有它，模型会偏好长回答。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的 SimPO</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"simpo"</span><span class="p">,</span>
    <span class="n">simpo_gamma</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span>   <span class="c1"># 目标 reward margin gamma</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">2.5</span><span class="p">,</span>          <span class="c1"># 长度归一化系数</span>
    <span class="c1"># 无需 ref_model！</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="kc">None</span><span class="p">,</span>    <span class="c1"># SimPO 是无 reference 的</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>
</code></pre></div>
</div>
