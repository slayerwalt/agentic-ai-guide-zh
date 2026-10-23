# 3.4　时序差分（Temporal Difference，TD）学习 {#temporal-difference-td-learning}

<p>TD 学习 <a class="book-citation" id="cite-sutton1988learning-1" href="../../references.md#ref-sutton1988learning" aria-label="参考文献 346">[346]</a>采用自举（bootstrap）思想——它使用其他价值估计来更新价值估计，无需等待完整 episode 结束。</p>

## 3.4.1　理解 TD 误差：以「惊讶」作为学习信号 {#understanding-td-error-surprise-as-a-learning-signal}

<p><strong>TD 误差（TD Error）</strong>衡量智能体对未来奖励的<strong>当前估计</strong>与采取一步后<strong>新估计</strong>之间的差异。简单来说，就是智能体<em>原本以为</em>会发生的事，与<em>实际</em>发生的事加上对未来的新预期之间的差。它代表了智能体的「惊讶」。</p>

<div class="admonition example"><p class="admonition-title">直觉：开车类比</p>
<p>设想你正在开车回家，预计需要 30 分钟。</p>

<ul>
<li><p><strong>预测</strong>：总共 30 分钟。</p></li>
<li><p><strong>现实变化</strong>：10 分钟后，你遇到意外的道路施工。GPS 更新提示，你还需要 35 分钟。</p></li>
<li><p><strong>TD 误差</strong>：总预期时间现在是 45 分钟（已过 10 分钟 + 剩余 35 分钟）。新估计（45 分钟）与旧估计（30 分钟）之差就是 <strong>+15 分钟的 TD 误差</strong>。你下次会用这个「惊讶」来调整路线。</p></li>
</ul>

<p><strong>正的 TD 误差</strong>意味着结果好于预期 <span class="arithmatex">\(\rightarrow\)</span> 提升该状态的价值。<br><strong>负的 TD 误差</strong>意味着结果差于预期 <span class="arithmatex">\(\rightarrow\)</span> 降低该状态的价值。</p>
</div>

## 3.4.2　TD 误差公式 {#the-td-error-formula}



<div class="arithmatex">\[\boxed{\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)}
\tag{3.9}\]</div>

<ul>
<li><p><span class="arithmatex">\(R_{t+1}\)</span>：采取动作后获得的<strong>即时奖励</strong>。</p></li>
<li><p><span class="arithmatex">\(\gamma V(S_{t+1})\)</span>：下一个状态的估计<strong>折扣价值</strong>（智能体从下一个状态起预期获得的回报，按折扣因子 <span class="arithmatex">\(\gamma\)</span> 缩放）。</p></li>
<li><p><span class="arithmatex">\(V(S_t)\)</span>：当前状态价值的<strong>原始估计</strong>。</p></li>
</ul>

<p>组合项 <span class="arithmatex">\((R_{t+1} + \gamma V(S_{t+1}))\)</span> 被称为 <strong>TD 目标（TD Target）</strong>。因此：</p>



<div class="arithmatex">\[\text{TD Error} = \text{TD Target} - \text{Old Estimate}
\tag{3.10}\]</div>

## 3.4.3　智能体如何使用 TD 误差 {#how-the-agent-uses-td-error}

<p>智能体调整其价值函数以将 TD 误差驱向零：</p>



<div class="arithmatex">\[\boxed{V(S_t) \leftarrow V(S_t) + \alpha \cdot \delta_t}
\tag{3.11}\]</div>

<ul>
<li><p>若 <span class="arithmatex">\(\delta_t &gt; 0\)</span>：结果好于预测 <span class="arithmatex">\(\rightarrow\)</span> 增加 <span class="arithmatex">\(V(S_t)\)</span>，使智能体倾向追求该状态。</p></li>
<li><p>若 <span class="arithmatex">\(\delta_t &lt; 0\)</span>：结果差于预测 <span class="arithmatex">\(\rightarrow\)</span> 降低 <span class="arithmatex">\(V(S_t)\)</span>，使智能体避开该状态。</p></li>
<li><p>若 <span class="arithmatex">\(\delta_t = 0\)</span>：预测完美 <span class="arithmatex">\(\rightarrow\)</span> 无需更新（已收敛）。</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">TD 与 Monte Carlo 的比较</p>
<p><strong>Monte Carlo</strong>：等到 episode 结束，使用实际回报 <span class="arithmatex">\(G_t\)</span>。无偏但方差高（单条完整轨迹可能不具代表性）。</p>

<p><strong>TD</strong>：每一步使用估计的未来价值 <span class="arithmatex">\(\gamma V(s_{t+1})\)</span> 进行更新。有偏（依赖 <span class="arithmatex">\(V\)</span> 的准确性），但方差低得多（单步更新，不会复合噪声）。</p>

<p><strong>TD(<span class="arithmatex">\(\lambda\)</span>)</strong>：在 TD(0) 与 Monte Carlo 之间插值。<span class="arithmatex">\(\lambda=0\)</span>：纯 TD；<span class="arithmatex">\(\lambda=1\)</span>：纯 MC。这正是 GAE 在 PPO 中所做的事情（取 <span class="arithmatex">\(\lambda=0.95\)</span>）。</p>
</div>

<p><strong>TD 目标</strong>：<span class="arithmatex">\(y_t = r_t + \gamma V(s_{t+1})\)</span>——我们要靠近的「更好的估计」。</p>

<p><strong>多步 TD</strong>（n 步回报）：</p>



<div class="arithmatex">\[G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
\tag{3.12}\]</div>
