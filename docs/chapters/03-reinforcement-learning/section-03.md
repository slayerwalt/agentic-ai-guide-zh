# 3.3　强化学习方法的分类 {#taxonomy-of-rl-methods}

<p>强化学习算法可以从多个维度进行分类。理解这一分类体系有助于针对特定问题选择合适的方法。</p>

<figure id="figure-3-2"><a href="../../assets/figures/fig_021_fig21.png" target="_blank" rel="noopener" aria-label="查看图 3.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_021_fig21.png" alt="" loading="lazy"></a><figcaption><span class="figure-number">图 3.2　</span></figcaption></figure>

<div class="admonition note"><p class="admonition-title">关键分类区分</p>
<p><strong>无模型（Model-Free） vs 基于模型（Model-Based）</strong>：</p>

<ul>
<li><p><strong>Model-Free</strong>：直接从交互经验中学习策略或价值函数；无需建模环境的转移机制。对 LLM 最实用（语言的转移机制难以建模）。</p></li>
<li><p><strong>Model-Based</strong>：学习或使用环境转移模型 <span class="arithmatex">\(P(s&#x27;|s,a)\)</span>。可以提前规划。样本效率更高，但需要准确的模型。</p></li>
</ul>

<p><strong>基于价值（Value-Based） vs 基于策略（Policy-Based）</strong>：</p>

<ul>
<li><p><strong>Value-Based</strong>：学习 <span class="arithmatex">\(Q(s,a)\)</span> 或 <span class="arithmatex">\(V(s)\)</span>，并通过 <span class="arithmatex">\(\arg\max_a Q(s,a)\)</span> 得到策略。适合离散、小规模动作空间（如 Atari），在连续或大规模动作空间上表现不佳。</p></li>
<li><p><strong>Policy-Based</strong>：直接参数化并优化 <span class="arithmatex">\(\pi_\theta(a|s)\)</span>。天然适合连续或高维动作空间，对 LLM 至关重要（词表 = 32K–128K 个动作）。</p></li>
<li><p><strong>Actor-Critic</strong>：两者结合——策略（actor）提议动作，价值函数（critic）评估动作。用于 LLM 的 PPO 即属于 actor-critic 方法。</p></li>
</ul>

<p><strong>On-Policy vs Off-Policy</strong>：</p>

<ul>
<li><p><strong>On-Policy</strong>：从<em>当前</em>策略生成的数据中学习。每次更新后必须重新生成数据。例如：REINFORCE、PPO、A2C。更稳定但样本效率较低。</p></li>
<li><p><strong>Off-Policy</strong>：从<em>任意</em>策略生成的数据中学习（包括旧版本或其他智能体）。可以复用过往经验。例如：Q-Learning、DQN、SAC。样本效率高但更难稳定。</p></li>
</ul>
</div>
