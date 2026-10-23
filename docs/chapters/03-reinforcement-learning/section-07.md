# 3.7　Actor-Critic 方法 {#actor-critic-methods}

<p>将策略梯度（actor）与学习得到的价值函数（critic）结合，以在保持策略优化灵活性的同时降低方差。</p>

<p><strong>结构</strong>：</p>

<ul>
<li><p><strong>Actor</strong> <span class="arithmatex">\(\pi_\theta(a|s)\)</span>：策略，提议动作。</p></li>
<li><p><strong>Critic</strong> <span class="arithmatex">\(V_\phi(s)\)</span> 或 <span class="arithmatex">\(Q_\phi(s,a)\)</span>：评估状态/动作的好坏，提供低方差基线。</p></li>
</ul>

<p><strong>Actor 更新</strong>（使用 critic 提供的优势）：</p>



<div class="arithmatex">\[\nabla_\theta J = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t\right], \quad \hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
\tag{3.21}\]</div>

<p><strong>Critic 更新</strong>（最小化 TD 误差）：</p>



<div class="arithmatex">\[\mathcal{L}_\text{critic} = \mathbb{E}\left[(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2\right]
\tag{3.22}\]</div>

<div class="admonition note"><p class="admonition-title">面向 LLM 的 PPO 演进路径</p>
<ol>
<li><p><strong>REINFORCE</strong> <a class="book-citation" id="cite-williams1992simple-1" href="../../references.md#ref-williams1992simple" aria-label="参考文献 379">[379]</a>：高方差、无自举 <span class="arithmatex">\(\rightarrow\)</span> 对 LLM 不可行</p></li>
<li><p><strong>A2C/A3C</strong> <a class="book-citation" id="cite-mnih2016asynchronous-1" href="../../references.md#ref-mnih2016asynchronous" aria-label="参考文献 259">[259]</a>（Advantage Actor-Critic）：使用基于 TD 的优势，方差更低，但步长无界。</p></li>
<li><p><strong>TRPO</strong> <a class="book-citation" id="cite-schulman2015trust-1" href="../../references.md#ref-schulman2015trust" aria-label="参考文献 317">[317]</a>：约束策略更新前后的 KL 散度。稳定但代价高（二阶方法）。</p></li>
<li><p><strong>PPO</strong> <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a>：通过裁剪策略比率，仅用一阶优化就达到与 TRPO 类似的稳定性。是 LLM RL 训练的标准方法。</p></li>
<li><p><strong>GRPO</strong>：完全去掉 critic，使用组内统计量作为基线。更简单，对可验证奖励有效。</p></li>
</ol>
</div>
