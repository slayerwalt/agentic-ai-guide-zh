# 3.2　核心概念与定义 {#core-concepts-and-definitions}

<p><strong>策略（Policy）</strong> <span class="arithmatex">\(\pi(a|s)\)</span>：从状态到动作概率的映射。确定性策略：<span class="arithmatex">\(a = \pi(s)\)</span>；随机性策略：<span class="arithmatex">\(a \sim \pi(\cdot|s)\)</span>。</p>

<p><strong>回报（Return）</strong>（累积折扣奖励）：</p>



<div class="arithmatex">\[G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\tag{3.1}\]</div>

<p><strong>价值函数（Value Function）</strong>（在策略 <span class="arithmatex">\(\pi\)</span> 下从状态 <span class="arithmatex">\(s\)</span> 出发的期望回报）：</p>



<div class="arithmatex">\[V^\pi(s) = \mathbb{E}_\pi\left[G_t \mid s_t = s\right] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \mid s_t = s\right]
\tag{3.2}\]</div>

<p><strong>动作价值函数（Action-Value Function）</strong>（从状态 <span class="arithmatex">\(s\)</span> 出发、采取动作 <span class="arithmatex">\(a\)</span>、其后遵循 <span class="arithmatex">\(\pi\)</span> 的期望回报）：</p>



<div class="arithmatex">\[Q^\pi(s, a) = \mathbb{E}_\pi\left[G_t \mid s_t = s, a_t = a\right]
\tag{3.3}\]</div>

<p><strong>优势函数（Advantage Function）</strong>（动作 <span class="arithmatex">\(a\)</span> 相对于平均水平好多少）：</p>



<div class="arithmatex">\[A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
\tag{3.4}\]</div>

<p><strong>贝尔曼方程（Bellman Equations）</strong>（递归关系）：</p>



<div class="arithmatex">\[\begin{align*}
V^\pi(s) &amp;= \sum_a \pi(a|s) \sum_{s&#x27;} P(s&#x27;|s,a)\left[R(s,a,s&#x27;) + \gamma V^\pi(s&#x27;)\right] \tag{3.5}\\

Q^\pi(s,a) &amp;= \sum_{s&#x27;} P(s&#x27;|s,a)\left[R(s,a,s&#x27;) + \gamma \sum_{a&#x27;} \pi(a&#x27;|s&#x27;) Q^\pi(s&#x27;, a&#x27;)\right]
\tag{3.6}
\end{align*}\]</div>

<div class="admonition note"><p class="admonition-title">最优策略与贝尔曼最优性</p>
<p>最优策略 <span class="arithmatex">\(\pi^*\)</span> 满足：</p>



<div class="arithmatex">\[V^*(s) = \max_a \sum_{s&#x27;} P(s&#x27;|s,a)\left[R(s,a,s&#x27;) + \gamma V^*(s&#x27;)\right]
\tag{3.7}\]</div>



<div class="arithmatex">\[Q^*(s,a) = \sum_{s&#x27;} P(s&#x27;|s,a)\left[R(s,a,s&#x27;) + \gamma \max_{a&#x27;} Q^*(s&#x27;, a&#x27;)\right]
\tag{3.8}\]</div>

<p>一旦得到 <span class="arithmatex">\(Q^*\)</span>，最优策略就直接是：<span class="arithmatex">\(\pi^*(s) = \arg\max_a Q^*(s,a)\)</span>。</p>
</div>
