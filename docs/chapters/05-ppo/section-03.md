# 5.3　完整的 PPO Loss {#full-ppo-loss}



<div class="arithmatex">\[L = L^{\text{CLIP}} - c_1 \underbrace{(V_\theta(s_t) - V^{\text{target}}_t)^2}_{\text{value loss}} + c_2 \underbrace{H[\pi_\theta(\cdot|s_t)]}_{\text{entropy bonus}}
\tag{5.2}\]</div>

<ul>
<li><p><strong>Value loss</strong>（<span class="arithmatex">\(c_1 = 0.1\)</span>）：训练 critic 去预测 return；同样被 clip 以保持稳定。</p></li>
<li><p><strong>熵奖励</strong>（<span class="arithmatex">\(c_2 = 0.01\)</span>）：防止过早收敛到确定性 policy，对探索至关重要。</p></li>
</ul>
