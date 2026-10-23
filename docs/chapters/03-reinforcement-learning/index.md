# 第 3 章 · 强化学习导论 {#sec:rl}

<p>强化学习（Reinforcement Learning，RL）是一种范式，其中<strong>智能体（agent）</strong>通过与<strong>环境（environment）</strong>交互、接收<strong>奖励（rewards）</strong>作为反馈，并优化其<strong>策略（policy）</strong>以最大化长期累积奖励来学习做出序贯决策 <a class="book-citation" id="cite-sutton2018reinforcement-1" href="../../references.md#ref-sutton2018reinforcement" aria-label="参考文献 347">[347]</a>。与监督学习（需要带标签的输入-输出对）不同，RL 通过<em>试错（trial and error）</em>来发现最优行为。</p>

<figure id="figure-3-1"><a href="../../assets/figures/fig_020_fig20.png" target="_blank" rel="noopener" aria-label="查看图 3.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_020_fig20.png" alt="强化学习概览：智能体与环境交互，接收奖励作为反馈，并通过试错更新其策略。与从带标签数据对学习的监督学习不同，RL 通过经验最大化奖励来学习应当做什么。" loading="lazy"></a><figcaption><span class="figure-number">图 3.1　</span>强化学习概览：智能体与环境交互，接收奖励作为反馈，并通过试错更新其策略。与从带标签数据对学习的监督学习不同，RL 通过经验最大化奖励来学习应当做什么。</figcaption></figure>

## 本章内容

<ul class="chapter-outline">
<li id="the-markov-decision-process-mdp"><a href="section-01.md">3.1　马尔可夫决策过程（Markov Decision Process，MDP）</a></li>
<li id="core-concepts-and-definitions"><a href="section-02.md">3.2　核心概念与定义</a></li>
<li id="taxonomy-of-rl-methods"><a href="section-03.md">3.3　强化学习方法的分类</a></li>
<li id="temporal-difference-td-learning"><a href="section-04.md">3.4　时序差分（Temporal Difference，TD）学习</a></li>
<li id="q-learning"><a href="section-05.md">3.5　Q-Learning</a></li>
<li id="policy-gradient-methods-reinforce"><a href="section-06.md">3.6　策略梯度方法——REINFORCE</a></li>
<li id="actor-critic-methods"><a href="section-07.md">3.7　Actor-Critic 方法</a></li>
<li id="generalized-advantage-estimation-gae"><a href="section-08.md">3.8　广义优势估计（Generalized Advantage Estimation，GAE）</a></li>
<li id="on-policy-vs-off-policy-detailed-comparison"><a href="section-09.md">3.9　同策略 vs 异策略——详细对比</a></li>
<li id="model-based-vs-model-free"><a href="section-10.md">3.10　基于模型 vs 无模型</a></li>
<li id="reward-shaping"><a href="section-11.md">3.11　奖励塑形（Reward Shaping）</a></li>
</ul>
