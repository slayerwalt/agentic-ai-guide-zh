# 5.2　Clipped 目标 {#the-clipped-objective}

<p>PPO 的核心创新是一个 clipped surrogate 目标：它能阻止破坏性的大幅 policy 更新，同时保持实现简单。</p>



<div class="arithmatex">\[\boxed{L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon)\hat{A}_t\right)\right]}
\tag{5.1}\]</div>

<p>其中 <span class="arithmatex">\(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}\)</span> 是概率比。</p>

<div class="admonition tip"><p class="admonition-title">Clipping 直觉——关键洞察</p>
<p><span class="arithmatex">\(\min\)</span> 算子构造了一个<strong>悲观下界</strong>：</p>

<ul>
<li><p><strong>好的 action（<span class="arithmatex">\(\hat{A} &gt; 0\)</span>）</strong>：我们希望提升其概率。Surrogate <span class="arithmatex">\(r\hat{A}\)</span> 会随 <span class="arithmatex">\(r\)</span> 增大而增大，但 clip 将收益上限锁在 <span class="arithmatex">\(r = 1 + \epsilon\)</span>。<em>“别因为一个好例子就贪心。”</em></p></li>
<li><p><strong>坏的 action（<span class="arithmatex">\(\hat{A} &lt; 0\)</span>）</strong>：我们希望降低其概率。<span class="arithmatex">\(r\hat{A}\)</span> 会随 <span class="arithmatex">\(r\)</span> 减小而改善，但 clip 将收益上限锁在 <span class="arithmatex">\(r = 1 - \epsilon\)</span>。<em>“别因为一个坏例子就遗忘得太激进。”</em></p></li>
</ul>

<p>净效应：每次更新步中 policy 的变化最多在 <span class="arithmatex">\(\pm\)</span>20% 之内，既能防止灾难性崩溃，也能防止过度自信地走向特化。</p>
</div>
