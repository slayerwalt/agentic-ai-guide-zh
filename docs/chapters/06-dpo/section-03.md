# 6.3　Gradient 分析 {#gradient-analysis}

<p>DPO 的 gradient 可分解为：</p>



<div class="arithmatex">\[\nabla_\theta \mathcal{L} = -\beta \cdot \underbrace{\sigma(-\hat{r}_w + \hat{r}_l)}_{\text{weight: higher when model is wrong}} \cdot \left[\nabla_\theta \log\pi_\theta(y_w|x) - \nabla_\theta \log\pi_\theta(y_l|x)\right]
\tag{6.2}\]</div>

<p><strong>解读</strong>：Gradient 会提升“被选”项的概率，降低“被拒”项的概率。当模型当前更偏好错误答案时，权重最大——也就是说，它把学习重点放在“易混淆”的成对样本上。</p>

<div class="admonition example"><p class="admonition-title">一个具体的 DPO 例子</p>
<p><strong>Prompt</strong>：“向一个 10 岁的小孩解释量子纠缠。”</p>

<p><strong>被选</strong>（<span class="arithmatex">\(y_w\)</span>）：“想象你有两枚魔法硬币。你抛其中一枚，如果它是正面，另一枚不管离多远都会立刻变成反面！”<br><span class="arithmatex">\(\log\pi_\theta(y_w|x) = -15.3\)</span>，<span class="arithmatex">\(\log\pi_\text{ref}(y_w|x) = -16.1\)</span></p>

<p><strong>被拒</strong>（<span class="arithmatex">\(y_l\)</span>）：“量子纠缠是一种现象，其中两个粒子相互关联，以致于一个粒子的量子态无法被独立描述。”<br><span class="arithmatex">\(\log\pi_\theta(y_l|x) = -12.8\)</span>，<span class="arithmatex">\(\log\pi_\text{ref}(y_l|x) = -12.5\)</span></p>

<p><strong>隐式 reward</strong>：<span class="arithmatex">\(\hat{r}_w = 0.1 \times ((-15.3) - (-16.1)) = 0.08\)</span>，<span class="arithmatex">\(\hat{r}_l = 0.1 \times ((-12.8) - (-12.5)) = -0.03\)</span></p>

<p><strong>Loss 输入</strong>：<span class="arithmatex">\(\sigma(0.08 - (-0.03)) = \sigma(0.11) = 0.527\)</span></p>

<p><strong>Loss</strong>：<span class="arithmatex">\(-\log(0.527) = 0.64\)</span>——模型仅略微偏好被选回答。Gradient 会用力推动。</p>

<p>训练之后：被选回答的概率提升、被拒回答的概率下降，直到 margin 稳定在约 <span class="arithmatex">\(1/(2\beta)\)</span>。</p>
</div>
