# 7.2　算法 {#algorithm}

<ol>
<li><p>对每个 prompt <span class="arithmatex">\(x\)</span>，采样 <span class="arithmatex">\(G\)</span> 条 completion：<span class="arithmatex">\(\{y_1, \ldots, y_G\} \sim \pi_\theta(\cdot|x)\)</span></p></li>
<li><p>对每条打分：<span class="arithmatex">\(r_i = R(x, y_i)\)</span></p></li>
<li><p>组内归一化：<span class="arithmatex">\(\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}\)</span>，其中 <span class="arithmatex">\(\mu_G = \frac{1}{G}\sum_j r_j\)</span>，<span class="arithmatex">\(\sigma_G = \text{std}(\{r_j\})\)</span></p></li>
<li><p>用这些 advantage 应用 PPO 风格的 clipped 更新</p></li>
</ol>



<div class="arithmatex">\[\boxed{\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}, \qquad L = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_i,\; \text{clip}(r_t(\theta), 1{\pm}\epsilon)\hat{A}_i\right)\right] - \beta D_\text{KL}[\pi_\theta\|\pi_\text{ref}]}
\tag{7.1}\]</div>

<div class="admonition tip"><p class="admonition-title">为什么组内归一化有效</p>
<p><strong>组均值近似 <span class="arithmatex">\(V(s)\)</span></strong>：若对同一 prompt 采样足够多的响应，其平均 reward 就是期望 reward 的蒙特卡洛估计，即 value function。</p>

<p><strong>高于均值 = 好动作</strong>：<span class="arithmatex">\(\hat{A}_i &gt; 0\)</span> 表示该响应对该 prompt 比平均更好，应强化。</p>

<p><strong>低于均值 = 坏动作</strong>：<span class="arithmatex">\(\hat{A}_i &lt; 0\)</span> 表示比平均更差，应抑制。</p>

<p><strong>归一化</strong>：除以 <span class="arithmatex">\(\sigma_G\)</span> 确保 advantage 在不同 reward 量级的 prompt 之间具有尺度不变性。</p>

<p><strong>DeepSeek-R1 突破</strong> <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a>：纯 GRPO 配合二值正确性 reward（答对 <span class="arithmatex">\(r=1\)</span>，答错 <span class="arithmatex">\(r=0\)</span>）在数学/代码上训练时，模型自发涌现出思维链（Chain-of-Thought，CoT）推理、自我验证和错误纠正——完全没有被显式指示这么做。</p>
</div>

<figure id="figure-7-1"><a href="../../assets/figures/fig_029_fig29.png" target="_blank" rel="noopener" aria-label="查看图 7.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_029_fig29.png" alt="GRPO 实战：对一个数学 prompt 采样 \(G{=}5\) 条响应。三条正确（\(r{=}1\)），两条错误（\(r{=}0\)）。组均值 \(\mu_G{=}0.6\) 充当 baseline；正确响应获得正 advantage（强化），错误响应获得负 advantage（抑制）。" loading="lazy"></a><figcaption><span class="figure-number">图 7.1　</span>GRPO 实战：对一个数学 prompt 采样 <span class="arithmatex">\(G{=}5\)</span> 条响应。三条正确（<span class="arithmatex">\(r{=}1\)</span>），两条错误（<span class="arithmatex">\(r{=}0\)</span>）。组均值 <span class="arithmatex">\(\mu_G{=}0.6\)</span> 充当 baseline；正确响应获得正 advantage（强化），错误响应获得负 advantage（抑制）。</figcaption></figure>
