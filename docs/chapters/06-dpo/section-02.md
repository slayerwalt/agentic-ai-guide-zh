# 6.2　数学推导 {#mathematical-derivation}

<p><strong>第 1 步</strong>：RLHF 目标：<span class="arithmatex">\(\max_\pi \mathbb{E}_{x,y\sim\pi}[r(x,y)] - \beta D_\text{KL}[\pi\|\pi_\text{ref}]\)</span></p>

<p><strong>第 2 步</strong>：最优解为：<span class="arithmatex">\(\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)\)</span></p>

<p><strong>第 3 步</strong>：重排式子，用 policy 表达 reward：<span class="arithmatex">\(r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)\)</span></p>

<p><strong>第 4 步</strong>：代入 Bradley-Terry 偏好模型 <span class="arithmatex">\(P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))\)</span>。<span class="arithmatex">\(Z(x)\)</span> 项相互抵消！</p>



<div class="arithmatex">\[\boxed{\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]}
\tag{6.1}\]</div>

<div class="admonition tip"><p class="admonition-title">DPO 实际在做什么</p>
<p>将<strong>隐式 reward</strong> 定义为 <span class="arithmatex">\(\hat{r}(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\)</span>。</p>

<p>DPO 在最小化一个交叉熵 loss，其中“标签”是：被选回答的隐式 reward 应高于被拒回答。Margin 由 <span class="arithmatex">\(\beta\)</span> 控制：</p>

<ul>
<li><p><span class="arithmatex">\(\beta\)</span> 较大：需要较大的 margin <span class="arithmatex">\(\rightarrow\)</span> policy 移动得更激进 <span class="arithmatex">\(\rightarrow\)</span> 有遗忘风险</p></li>
<li><p><span class="arithmatex">\(\beta\)</span> 较小：较小的 margin 就够了 <span class="arithmatex">\(\rightarrow\)</span> policy 更靠近 reference <span class="arithmatex">\(\rightarrow\)</span> 更保守</p></li>
</ul>

<p>Reference 模型起到正则化作用：policy 必须用偏好对齐来“证明”自己偏离 reference 的合理性。</p>
</div>
