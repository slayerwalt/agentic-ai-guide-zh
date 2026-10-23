# 第 3 章 · 强化学习导论 {#sec:rl}

<p>强化学习（Reinforcement Learning，RL）是一种范式，其中<strong>智能体（agent）</strong>通过与<strong>环境（environment）</strong>交互、接收<strong>奖励（rewards）</strong>作为反馈，并优化其<strong>策略（policy）</strong>以最大化长期累积奖励来学习做出序贯决策 <a class="book-citation" id="cite-sutton2018reinforcement-1" href="../../references.md#ref-sutton2018reinforcement" aria-label="参考文献 347">[347]</a>。与监督学习（需要带标签的输入-输出对）不同，RL 通过<em>试错（trial and error）</em>来发现最优行为。</p>

<figure id="figure-3-1"><a href="../../assets/figures/fig_020_fig20.png" target="_blank" rel="noopener" aria-label="查看图 3.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_020_fig20.png" alt="强化学习概览：智能体与环境交互，接收奖励作为反馈，并通过试错更新其策略。与从带标签数据对学习的监督学习不同，RL 通过经验最大化奖励来学习应当做什么。" loading="lazy"></a><figcaption><span class="figure-number">图 3.1　</span>强化学习概览：智能体与环境交互，接收奖励作为反馈，并通过试错更新其策略。与从带标签数据对学习的监督学习不同，RL 通过经验最大化奖励来学习应当做什么。</figcaption></figure>

## 3.1　马尔可夫决策过程（Markov Decision Process，MDP） {#the-markov-decision-process-mdp}

<p>MDP 是一个五元组 <span class="arithmatex">\((S, A, P, R, \gamma)\)</span>：</p>

<ul>
<li><p><span class="arithmatex">\(S\)</span>：状态空间——环境所有可能的配置</p></li>
<li><p><span class="arithmatex">\(A\)</span>：动作空间——智能体可用的所有动作</p></li>
<li><p><span class="arithmatex">\(P(s&#x27;|s, a)\)</span>：转移函数——在状态 <span class="arithmatex">\(s\)</span> 下采取动作 <span class="arithmatex">\(a\)</span> 后到达状态 <span class="arithmatex">\(s&#x27;\)</span> 的概率</p></li>
<li><p><span class="arithmatex">\(R(s, a, s&#x27;)\)</span>：奖励函数——一次状态转移获得的即时标量反馈</p></li>
<li><p><span class="arithmatex">\(\gamma \in [0, 1]\)</span>：折扣因子——未来奖励相对于即时奖励的权重</p></li>
</ul>

<p><strong>马尔可夫性质（Markov Property）</strong>：未来只依赖于当前状态，而不依赖于历史：</p>



<div class="arithmatex">\[P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)\]</div>

<p>这一性质使问题变得可处理。</p>

<div class="admonition tip"><p class="admonition-title">智能体-环境交互循环</p>
<p>在每个时间步 <span class="arithmatex">\(t\)</span>：</p>

<ol>
<li><p>智能体观察状态 <span class="arithmatex">\(s_t\)</span></p></li>
<li><p>智能体根据策略 <span class="arithmatex">\(\pi(a|s)\)</span> 选择动作 <span class="arithmatex">\(a_t\)</span></p></li>
<li><p>环境转移到 <span class="arithmatex">\(s_{t+1} \sim P(\cdot|s_t, a_t)\)</span></p></li>
<li><p>智能体接收奖励 <span class="arithmatex">\(r_t = R(s_t, a_t, s_{t+1})\)</span></p></li>
<li><p>重复直到终止状态或时间步上限 <span class="arithmatex">\(T\)</span></p></li>
</ol>
</div>

## 3.2　核心概念与定义 {#core-concepts-and-definitions}

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

## 3.3　强化学习方法的分类 {#taxonomy-of-rl-methods}

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

## 3.4　时序差分（Temporal Difference，TD）学习 {#temporal-difference-td-learning}

<p>TD 学习 <a class="book-citation" id="cite-sutton1988learning-1" href="../../references.md#ref-sutton1988learning" aria-label="参考文献 346">[346]</a>采用自举（bootstrap）思想——它使用其他价值估计来更新价值估计，无需等待完整 episode 结束。</p>

### 3.4.1　理解 TD 误差：以「惊讶」作为学习信号 {#understanding-td-error-surprise-as-a-learning-signal}

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

### 3.4.2　TD 误差公式 {#the-td-error-formula}



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

### 3.4.3　智能体如何使用 TD 误差 {#how-the-agent-uses-td-error}

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

## 3.5　Q-Learning {#q-learning}

<p>Q-Learning <a class="book-citation" id="cite-watkins1989learning-1" href="../../references.md#ref-watkins1989learning" aria-label="参考文献 373">[373]</a>是基础性的<strong>异策略、基于价值</strong>的算法。它直接学习最优 <span class="arithmatex">\(Q^*\)</span>，与所遵循的策略无关。</p>

<p><strong>更新规则</strong>：</p>



<div class="arithmatex">\[\boxed{Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_t + \gamma \max_{a&#x27;} Q(s_{t+1}, a&#x27;) - Q(s_t, a_t)\right]}
\tag{3.13}\]</div>

<div class="admonition tip"><p class="admonition-title">为什么 Q-Learning 是异策略的</p>
<p>更新使用 <span class="arithmatex">\(\max_{a&#x27;} Q(s_{t+1}, a&#x27;)\)</span>——下一状态下<em>最优</em>动作的价值，与智能体实际采取的动作无关。这意味着目标始终在最优策略下计算，即便行为策略以随机方式探索（<span class="arithmatex">\(\epsilon\)</span>-贪心）。</p>

<p>这就是 Q-Learning 可以从重放缓冲区（Replay Buffer）、演示数据或任何经验来源中学习的原因。数据无需来自当前策略。</p>
</div>

<p><strong>SARSA</strong> <a class="book-citation" id="cite-rummery1994online-1" href="../../references.md#ref-rummery1994online" aria-label="参考文献 311">[311]</a>（同策略替代方案）：使用<em>实际采取的</em>动作而非最大值：</p>



<div class="arithmatex">\[Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha\left[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)\right]
\tag{3.14}\]</div>

<p><strong>深度 Q 网络（Deep Q-Networks，DQN）</strong> <a class="book-citation" id="cite-mnih2015human-1" href="../../references.md#ref-mnih2015human" aria-label="参考文献 258">[258]</a>：用神经网络 <span class="arithmatex">\(Q_\theta(s,a)\)</span> 取代表格化的 <span class="arithmatex">\(Q(s,a)\)</span>。关键创新包括：经验重放缓冲区（异策略数据复用）、目标网络（稳定性）、<span class="arithmatex">\(\epsilon\)</span>-贪心探索。</p>

<p><strong>DQN 损失函数</strong>：网络通过最小化从重放缓冲区采样的 mini-batch 上的均方 TD 误差来训练：</p>



<div class="arithmatex">\[\boxed{\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s&#x27;) \sim \mathcal{B}}\!\left[\left(r + \gamma \max_{a&#x27;} Q_{\bar{\theta}}(s&#x27;, a&#x27;) - Q_\theta(s, a)\right)^2\right]}
\tag{3.15}\]</div>

<p>其中 <span class="arithmatex">\(Q_{\bar{\theta}}\)</span> 是<strong>目标网络（target network）</strong>——<span class="arithmatex">\(Q_\theta\)</span> 的冻结副本，每 <span class="arithmatex">\(C\)</span> 步才更新一次（如 <span class="arithmatex">\(C = 10{,}000\)</span>）。这避免了「移动目标」问题：没有它的话，预测与目标会同时变化，导致发散。</p>

<p><strong>梯度更新</strong>：对损失关于 <span class="arithmatex">\(\theta\)</span> 求梯度（注意：目标 <span class="arithmatex">\(y\)</span> 被视为常量——梯度不流经 <span class="arithmatex">\(\bar{\theta}\)</span>）：</p>



<div class="arithmatex">\[\nabla_\theta \mathcal{L} = -\mathbb{E}\!\left[\underbrace{\left(r + \gamma \max_{a&#x27;} Q_{\bar{\theta}}(s&#x27;, a&#x27;) - Q_\theta(s, a)\right)}_{\text{TD error } \delta}\; \nabla_\theta Q_\theta(s, a)\right]
\tag{3.16}\]</div>



<div class="arithmatex">\[\theta \leftarrow \theta - \alpha \cdot \delta \cdot \nabla_\theta Q_\theta(s, a)
\tag{3.17}\]</div>

<p><strong>学习流程</strong>（每个训练步）：</p>

<ol>
<li><p><strong>行动</strong>：通过 <span class="arithmatex">\(\epsilon\)</span>-贪心选择动作：以概率 <span class="arithmatex">\(\epsilon\)</span> 选择随机动作，否则 <span class="arithmatex">\(a = \arg\max_a Q_\theta(s, a)\)</span>。在前 100 万步内将 <span class="arithmatex">\(\epsilon\)</span> 从 1.0 衰减到 0.01。</p></li>
<li><p><strong>存储</strong>：将转移 <span class="arithmatex">\((s, a, r, s&#x27;, d)\)</span> 存入重放缓冲区 <span class="arithmatex">\(\mathcal{B}\)</span>（容量约 100 万）。</p></li>
<li><p><strong>采样</strong>：从 <span class="arithmatex">\(\mathcal{B}\)</span> 中均匀采样 32 条转移作为 mini-batch。</p></li>
<li><p><strong>计算目标</strong>：<span class="arithmatex">\(y = r + \gamma(1 - d)\max_{a&#x27;} Q_{\bar{\theta}}(s&#x27;, a&#x27;)\)</span>（若为终止状态则未来价值为零）。</p></li>
<li><p><strong>更新</strong>：对 <span class="arithmatex">\((y - Q_\theta(s,a))^2\)</span> 进行梯度下降。将梯度裁剪到 <span class="arithmatex">\([-1, 1]\)</span>（Huber 损失变体）。</p></li>
<li><p><strong>同步目标网络</strong>：每 <span class="arithmatex">\(C\)</span> 步执行 <span class="arithmatex">\(\bar{\theta} \leftarrow \theta\)</span>。</p></li>
</ol>

### 3.5.1　理解重放缓冲区 {#understanding-replay-buffers}

<p><strong>重放缓冲区（Replay Buffer）</strong> <a class="book-citation" id="cite-lin1992self-1" href="../../references.md#ref-lin1992self" aria-label="参考文献 219">[219]</a>（经验回放）是一种数据存储机制，保存过往经验以便智能体后续再学习。智能体不会在执行动作后立刻丢弃数据，而是将转移存入记忆库，并随机采样 mini-batch 用于训练。</p>

<p><strong>存储内容</strong>：每条转移是一个元组：</p>



<div class="arithmatex">\[e_t = (s_t, a_t, r_t, s_{t+1}, d_t)
\tag{3.18}\]</div>

<p>其中 <span class="arithmatex">\(d_t\)</span> 是布尔标志，表示 episode 是否结束。</p>

<div class="admonition note"><p class="admonition-title">为何重放缓冲区不可或缺</p>
<ul>
<li><p><strong>打破数据相关性</strong>：连续步之间高度相关。神经网络在序列数据上泛化较差。从缓冲区随机采样可使训练样本近似独立同分布（i.i.d.）。</p></li>
<li><p><strong>防止灾难性遗忘</strong>：没有缓冲区，智能体在通过某个困难关卡后，若在后续 10K 步中持续在更后面的关卡上失败，可能会忘记如何通过先前那关。缓冲区可确保它持续练习旧的场景。</p></li>
<li><p><strong>提升样本效率</strong>：运行环境可能很慢。重放缓冲区允许同一条转移用于多次权重更新，从每一步中榨取更多价值。</p></li>
</ul>
</div>

<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">random</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">collections</span><span class="w"> </span><span class="kn">import</span> <span class="n">deque</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ReplayBuffer</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">capacity</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">buffer</span> <span class="o">=</span> <span class="n">deque</span><span class="p">(</span><span class="n">maxlen</span><span class="o">=</span><span class="n">capacity</span><span class="p">)</span>  <span class="c1"># 有界队列</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">push</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">state</span><span class="p">,</span> <span class="n">action</span><span class="p">,</span> <span class="n">reward</span><span class="p">,</span> <span class="n">next_state</span><span class="p">,</span> <span class="n">done</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">buffer</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="n">state</span><span class="p">,</span> <span class="n">action</span><span class="p">,</span> <span class="n">reward</span><span class="p">,</span> <span class="n">next_state</span><span class="p">,</span> <span class="n">done</span><span class="p">))</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">sample</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">):</span>
        <span class="c1"># 通过随机选取经验打破相关性</span>
        <span class="k">return</span> <span class="n">random</span><span class="o">.</span><span class="n">sample</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">buffer</span><span class="p">,</span> <span class="n">batch_size</span><span class="p">)</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__len__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="k">return</span> <span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">buffer</span><span class="p">)</span>
</code></pre></div>

<div class="admonition tip"><p class="admonition-title">优先级经验回放（Prioritized Experience Replay，PER）</p>
<p>在标准缓冲区中，所有经验的采样概率相等。但有些经验信息量大得多。<strong>PER</strong> <a class="book-citation" id="cite-schaul2016prioritized-1" href="../../references.md#ref-schaul2016prioritized" aria-label="参考文献 314">[314]</a>按 <strong>TD 误差幅度</strong>对采样概率进行加权——如果某条转移引发了巨大「惊讶」（<span class="arithmatex">\(|\delta_t|\)</span> 高），智能体就更频繁地采样它，从而更快修正模型。在 Atari 基准上可将学习速度加快 2–3 倍。</p>
</div>

<div class="admonition warning"><p class="admonition-title">为什么 Q-Learning 对 LLM 不适用</p>
<p>语言生成中的动作空间是整个词表（<span class="arithmatex">\(|A| = 32\text{K}\)</span>–<span class="arithmatex">\(128\text{K}\)</span>），状态空间是所有可能的 token 序列（无限）。在每个 token 位置对 128K 个动作计算 <span class="arithmatex">\(\max_a Q(s,a)\)</span> 是不可行的。这就是 LLM RL 使用<strong>基于策略</strong>方法（PPO、GRPO）的原因。</p>
</div>

## 3.6　策略梯度方法——REINFORCE {#policy-gradient-methods-reinforce}

<p>与其学习价值函数再推导策略，不如直接优化策略参数 <span class="arithmatex">\(\theta\)</span> 以最大化期望回报 <a class="book-citation" id="cite-williams1992simple-1" href="../../references.md#ref-williams1992simple" aria-label="参考文献 379">[379]</a>。</p>

<p><strong>目标</strong>：<span class="arithmatex">\(J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T r_t\right]\)</span></p>

<p><strong>策略梯度定理（Policy Gradient Theorem）</strong>：</p>



<div class="arithmatex">\[\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]}
\tag{3.19}\]</div>

<div class="admonition note"><p class="admonition-title">策略梯度定理——形式化推导（5 步）</p>
<p><strong>第 1 步</strong>：定义目标。我们希望最大化期望回报：</p>



<div class="arithmatex">\[J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T r_t\right] = \sum_\tau P(\tau|\theta) R(\tau)\]</div>

<p>其中 <span class="arithmatex">\(P(\tau|\theta) = p(s_0)\prod_{t=0}^T \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t, a_t)\)</span> 是轨迹概率。</p>

<p><strong>第 2 步</strong>：求梯度。只有 <span class="arithmatex">\(\pi_\theta\)</span> 项依赖于 <span class="arithmatex">\(\theta\)</span>（状态转移 <span class="arithmatex">\(p\)</span> 与 <span class="arithmatex">\(\theta\)</span> 无关）：</p>



<div class="arithmatex">\[\nabla_\theta J = \sum_\tau \nabla_\theta P(\tau|\theta)\, R(\tau)\]</div>

<p><strong>第 3 步</strong>：应用<strong>对数求导技巧（log-derivative trick）</strong>：<span class="arithmatex">\(\nabla_\theta P(\tau|\theta) = P(\tau|\theta)\, \nabla_\theta \log P(\tau|\theta)\)</span>：</p>



<div class="arithmatex">\[\nabla_\theta J = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\nabla_\theta \log P(\tau|\theta)\, R(\tau)\right]\]</div>

<p><strong>第 4 步</strong>：展开 <span class="arithmatex">\(\log P(\tau|\theta)\)</span>。<span class="arithmatex">\(\log p(s_0)\)</span> 与 <span class="arithmatex">\(\log p(s_{t+1}|s_t,a_t)\)</span> 在 <span class="arithmatex">\(\nabla_\theta\)</span> 下消失：</p>



<div class="arithmatex">\[\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)\]</div>

<p><strong>第 5 步</strong>：合并。未来奖励不依赖于过去动作（因果性），因此每个 <span class="arithmatex">\(\nabla\log\pi\)</span> 仅与未来回报 <span class="arithmatex">\(G_t = \sum_{t&#x27;=t}^T r_{t&#x27;}\)</span> 配对：</p>



<div class="arithmatex">\[\boxed{\nabla_\theta J = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]}\]</div>
</div>

<div class="admonition tip"><p class="admonition-title">这一结果为何优美</p>
<p>该梯度<strong>不需要对环境的状态转移</strong> <span class="arithmatex">\(p(s&#x27;|s,a)\)</span> <strong>求导</strong>。对数求导技巧将其转化为一个期望，只需<em>运行策略并观察奖励</em>就能估计。把 <span class="arithmatex">\(G_t\)</span> 替换为优势 <span class="arithmatex">\(\hat{A}_t = G_t - V(s_t)\)</span> 可以在不引入偏差的前提下降低方差（因为对任何仅依赖状态的基线 <span class="arithmatex">\(b(s)\)</span>，都有 <span class="arithmatex">\(\mathbb{E}[\nabla\log\pi \cdot b(s)] = 0\)</span>）。</p>
</div>

<p><strong>REINFORCE 算法（REINFORCE）</strong> <a class="book-citation" id="cite-williams1992simple-2" href="../../references.md#ref-williams1992simple" aria-label="参考文献 379">[379]</a>（Williams, 1992）：</p>

<ol>
<li><p>在 <span class="arithmatex">\(\pi_\theta\)</span> 下采样完整轨迹 <span class="arithmatex">\(\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)\)</span></p></li>
<li><p>为每个时间步计算回报 <span class="arithmatex">\(G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}\)</span></p></li>
<li><p>更新：<span class="arithmatex">\(\theta \leftarrow \theta + \alpha \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\)</span></p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">REINFORCE 直觉——「奖励加权的最大似然」</p>
<p><span class="arithmatex">\(\nabla_\theta \log \pi_\theta(a_t|s_t)\)</span> 是提升动作 <span class="arithmatex">\(a_t\)</span> 概率的方向。乘以 <span class="arithmatex">\(G_t\)</span> 意味着：</p>

<ul>
<li><p>高奖励轨迹：提升所有所采取动作的概率（<span class="arithmatex">\(G_t\)</span> 为正）</p></li>
<li><p>低奖励轨迹：降低所采取动作的概率（扣除基线后 <span class="arithmatex">\(G_t\)</span> 为负）</p></li>
</ul>

<p>这是一种监督学习，「标签」是你采取的动作，并按结果好坏加权。</p>
</div>

<p><strong>基线带来的方差缩减</strong>：</p>



<div class="arithmatex">\[\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]
\tag{3.20}\]</div>

<p>任何与 <span class="arithmatex">\(a_t\)</span> 无关的基线 <span class="arithmatex">\(b(s_t)\)</span> 都能在保持梯度无偏的同时降低方差。最佳选择：<span class="arithmatex">\(b(s_t) = V^\pi(s_t)\)</span>。此时 <span class="arithmatex">\(G_t - V(s_t) \approx A^\pi(s_t, a_t)\)</span> = 优势。</p>

<div class="admonition warning"><p class="admonition-title">REINFORCE 的局限</p>
<ul>
<li><p><strong>方差高</strong>：每次梯度仅使用一条轨迹，需要数千样本才能保证更新稳定。</p></li>
<li><p><strong>无自举</strong>：必须等待完整 episode（没有部分回报信号）。</p></li>
<li><p><strong>样本效率低</strong>：数据使用一次即丢弃（同策略）。</p></li>
<li><p><strong>缺乏步长控制</strong>：策略更新步可能灾难性地过大。</p></li>
</ul>

<p>这些局限催生了如下演进路径：REINFORCE <span class="arithmatex">\(\to\)</span> Actor-Critic <span class="arithmatex">\(\to\)</span> TRPO <span class="arithmatex">\(\to\)</span> <strong>PPO</strong>。</p>
</div>

## 3.7　Actor-Critic 方法 {#actor-critic-methods}

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
<li><p><strong>REINFORCE</strong> <a class="book-citation" id="cite-williams1992simple-1--section-07" href="../../references.md#ref-williams1992simple" aria-label="参考文献 379">[379]</a>：高方差、无自举 <span class="arithmatex">\(\rightarrow\)</span> 对 LLM 不可行</p></li>
<li><p><strong>A2C/A3C</strong> <a class="book-citation" id="cite-mnih2016asynchronous-1" href="../../references.md#ref-mnih2016asynchronous" aria-label="参考文献 259">[259]</a>（Advantage Actor-Critic）：使用基于 TD 的优势，方差更低，但步长无界。</p></li>
<li><p><strong>TRPO</strong> <a class="book-citation" id="cite-schulman2015trust-1" href="../../references.md#ref-schulman2015trust" aria-label="参考文献 317">[317]</a>：约束策略更新前后的 KL 散度。稳定但代价高（二阶方法）。</p></li>
<li><p><strong>PPO</strong> <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a>：通过裁剪策略比率，仅用一阶优化就达到与 TRPO 类似的稳定性。是 LLM RL 训练的标准方法。</p></li>
<li><p><strong>GRPO</strong>：完全去掉 critic，使用组内统计量作为基线。更简单，对可验证奖励有效。</p></li>
</ol>
</div>

## 3.8　广义优势估计（Generalized Advantage Estimation，GAE） {#generalized-advantage-estimation-gae}

<p><strong>动机</strong>：Actor-Critic 框架需要对优势 <span class="arithmatex">\(A(s,a) = Q(s,a) - V(s)\)</span> 进行良好估计——这个动作比平均水平好多少？但这里存在根本性的张力：</p>

<ul>
<li><p><strong>1 步 TD 优势</strong>（<span class="arithmatex">\(r_t + \gamma V(s_{t+1}) - V(s_t)\)</span>）：方差低（仅含一步随机性），但<strong>有偏</strong>——若价值函数 <span class="arithmatex">\(V\)</span> 不准确，优势估计就会系统性偏离。</p></li>
<li><p><strong>Monte Carlo 优势</strong>（<span class="arithmatex">\(G_t - V(s_t)\)</span>）：无偏（使用实际回报），但<strong>方差高</strong>——许多随机奖励之和在不同 episode 间波动剧烈。</p></li>
</ul>

<p>GAE <a class="book-citation" id="cite-schulman2016high-1" href="../../references.md#ref-schulman2016high" aria-label="参考文献 318">[318]</a>（Schulman 等，2016）通过单一参数 <span class="arithmatex">\(\lambda \in [0, 1]\)</span> 在两种极端之间提供<strong>平滑插值</strong>。它对所有 <span class="arithmatex">\(n\)</span> 的 <span class="arithmatex">\(n\)</span> 步优势估计进行指数加权平均，给出一种在偏差与方差之间权衡的原则化方式。</p>

<p><strong>核心思想</strong>：在每个时间步计算 1 步 TD 误差 <span class="arithmatex">\(\delta_t\)</span>，然后用指数衰减权重 <span class="arithmatex">\((\gamma\lambda)^l\)</span> 进行混合——近期 TD 误差权重完整，远期的被降权：</p>



<div class="arithmatex">\[\boxed{\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)}
\tag{3.23}\]</div>

<figure id="figure-3-3"><a href="../../assets/figures/fig_022_fig22.png" target="_blank" rel="noopener" aria-label="查看图 3.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_022_fig22.png" alt="GAE 数据流：每个 TD 残差 \(\delta_{t+l}^V\) 在求和前被乘以权重 \((\gamma\lambda)^l\)。\(\lambda\) 越大，纳入的未来残差越多（偏差更低，方差更高）。" loading="lazy"></a><figcaption><span class="figure-number">图 3.3　</span>GAE 数据流：每个 TD 残差 <span class="arithmatex">\(\delta_{t+l}^V\)</span> 在求和前被乘以权重 <span class="arithmatex">\((\gamma\lambda)^l\)</span>。<span class="arithmatex">\(\lambda\)</span> 越大，纳入的未来残差越多（偏差更低，方差更高）。</figcaption></figure>

<div class="admonition tip"><p class="admonition-title"><span class="arithmatex">\(\lambda\)</span> 控制什么——偏差-方差权衡</p>
<ul>
<li><p><span class="arithmatex">\(\lambda = 0\)</span>：<span class="arithmatex">\(\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\)</span>。完全信任价值函数。方差低，但若 <span class="arithmatex">\(V\)</span> 不准确则有偏。</p></li>
<li><p><span class="arithmatex">\(\lambda = 1\)</span>：<span class="arithmatex">\(\hat{A}_t = \sum_l \gamma^l r_{t+l} - V(s_t)\)</span>。完整 Monte Carlo 回报减去基线。无偏但方差极高。</p></li>
<li><p><span class="arithmatex">\(\lambda = 0.95\)</span>（标准取值）：黄金平衡点。主要信任 <span class="arithmatex">\(V\)</span>，但在远期效应上用实际回报修正。之所以可行，是因为价值头在初始训练后会变得准确。</p></li>
</ul>

<p>针对 LLM 的常用取值：<span class="arithmatex">\(\gamma = 1.0\)</span>（无时间折扣——单轮对话中所有 token 权重相等），<span class="arithmatex">\(\lambda = 0.95\)</span>。</p>
</div>

### 3.8.1　GAE 中偏差与方差的直观映射 {#intuitive-mapping-of-bias-and-variance-in-gae}

<p>在监督学习中，偏差与方差源于模型的结构性假设。而在通过 GAE 进行的强化学习中，它们源于<strong>你信任有缺陷的模型有多深 vs 你信任混乱的环境有多深</strong>：</p>

<ul>
<li><p><strong>偏差（系统性偏离）</strong>：当估计器依赖于价值网络 <span class="arithmatex">\(V_\theta\)</span> 的结构性假设与不完美预测时产生。如果 <span class="arithmatex">\(\theta\)</span> 训练不足或容量不够，基线猜测就会系统性出错。</p></li>
<li><p><strong>方差（样本抖动）</strong>：当估计器依赖于长且无约束的环境轨迹时产生。随机转移、随机种子与策略执行噪声在长时间步上累积，导致不同 rollout 之间的实际样本奖励剧烈波动。</p></li>
</ul>

### 3.8.2　架构光谱：边界情形分析 {#the-architectural-spectrum-boundary-analysis}

<figure id="figure-3-4"><a href="../../assets/figures/fig_023_fig23.png" target="_blank" rel="noopener" aria-label="查看图 3.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_023_fig23.png" alt="GAE 中的偏差 vs 方差：\(\lambda\) 控制这一权衡。较小的 \(\lambda\)（左）通过自举得到高偏差/低方差；较大的 \(\lambda\)（右）使用完整 Monte Carlo 回报得到低偏差/高方差。最优选择（\(\lambda \in [0.9, 0.95]\)）在稳定训练与准确的长程信用分配之间取得平衡。" loading="lazy"></a><figcaption><span class="figure-number">图 3.4　</span>GAE 中的偏差 vs 方差：<span class="arithmatex">\(\lambda\)</span> 控制这一权衡。较小的 <span class="arithmatex">\(\lambda\)</span>（左）通过自举得到高偏差/低方差；较大的 <span class="arithmatex">\(\lambda\)</span>（右）使用完整 Monte Carlo 回报得到低偏差/高方差。最优选择（<span class="arithmatex">\(\lambda \in [0.9, 0.95]\)</span>）在稳定训练与准确的长程信用分配之间取得平衡。</figcaption></figure>

<p>超参数 <span class="arithmatex">\(\lambda\)</span> 充当了两种基本估计范式之间的「滑尺」。</p>

<div class="admonition note"><p class="admonition-title">高偏差/低方差的限制（<span class="arithmatex">\(\lambda = 0\)</span>）</p>


<div class="arithmatex">\[\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)
\tag{3.24}\]</div>

<ul>
<li><p><strong>行为</strong>：优势在很大程度上由参数 <span class="arithmatex">\(\theta\)</span> 的当前状态决定。</p></li>
<li><p><strong>直觉</strong>：高度<strong>有偏</strong>，因为网络在 1 步窗口内为自己打分；若 <span class="arithmatex">\(V_\theta\)</span> 不准确，梯度步就会被破坏。<strong>方差低</strong>，因为它忽略 <span class="arithmatex">\(t+1\)</span> 步之后的未来随机事件，从而产生平滑、稳定的参数更新。</p></li>
<li><p><strong>风险</strong>：策略陷入次优局部极小——永远发现不了复杂的延迟奖励序列。</p></li>
</ul>
</div>

<div class="admonition note"><p class="admonition-title">低偏差/高方差的限制（<span class="arithmatex">\(\lambda = 1\)</span>）</p>
<p>当 <span class="arithmatex">\(\lambda = 1\)</span> 时，中间的价值项相互裂项相消，GAE 退化为 Monte Carlo 回报减去基线：</p>



<div class="arithmatex">\[\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V_\theta(s_t)
\tag{3.25}\]</div>

<ul>
<li><p><strong>行为</strong>：舍弃自举式前瞻，将整个 episode 的实际现实进行累加。</p></li>
<li><p><strong>直觉</strong>：对真实的环境转移完全<strong>无偏</strong>——衡量的是实际奖励而非神经网络近似。然而表现出极端的<strong>高方差</strong>：episode 早期的微小扰动可能导致完全不同的总回报，使策略更新变得不稳定。</p></li>
<li><p><strong>风险</strong>：破坏性梯度更新；训练爆炸。</p></li>
</ul>
</div>

### 3.8.3　权衡矩阵 {#the-trade-off-matrix}

<p>通过选择 <span class="arithmatex">\(\lambda \in [0.95, 0.99]\)</span>，GAE 可以最小化优势估计的总均方误差：</p>

<p class="table-caption" id="table-3-1"><span class="table-number">表 3.1　</span>GAE 参数选择的操作性对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>配置</strong></th><th scope="col"><strong>统计特性</strong></th><th scope="col"><strong>核心依赖</strong></th><th scope="col"><strong>实际风险</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(\lambda = 0\)</span></td><td>高偏差、低方差</td><td>模型参数（<span class="arithmatex">\(\theta\)</span>）</td><td>策略陷入次优局部极小</td></tr>
<tr><td><span class="arithmatex">\(\lambda \in [0.95, 0.99]\)</span></td><td>均衡（最优 MSE）</td><td>混合融合</td><td>需要根据环境随机性进行调参</td></tr>
<tr><td><span class="arithmatex">\(\lambda = 1\)</span></td><td>低偏差、高方差</td><td>经验性环境 rollout</td><td>破坏性梯度更新；训练爆炸</td></tr></tbody>
</table>
</div>

### 3.8.4　<span class="arithmatex">\(\lambda\)</span> 调参诊断 {#diagnostics-for-tuning-lambda}

<p>监控训练曲线能直接揭示当前主导因素是偏差还是方差：</p>

<ol>
<li><p><strong>高方差指标</strong>：策略熵骤降，同时价值函数的解释方差变得高度为负或剧烈波动 <span class="arithmatex">\(\rightarrow\)</span> 策略更新噪声大。<strong>对策</strong>：降低 <span class="arithmatex">\(\lambda\)</span> 以平滑目标更新。</p></li>
<li><p><strong>高偏差指标</strong>：智能体在早期获得稳定训练，但完全无法发现复杂的延迟奖励序列 <span class="arithmatex">\(\rightarrow\)</span> 由于自举导致对长程依赖的低估。<strong>对策</strong>：将 <span class="arithmatex">\(\lambda\)</span> 提高至接近 <span class="arithmatex">\(1.0\)</span>，让策略接触真实的下游轨迹信号。</p></li>
</ol>

## 3.9　同策略 vs 异策略——详细对比 {#on-policy-vs-off-policy-detailed-comparison}

<span id="table-3-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>On-Policy</strong></th><th scope="col"><strong>Off-Policy</strong></th></tr></thead>
<tbody><tr><td><strong>数据来源</strong></td><td>仅当前策略 <span class="arithmatex">\(\pi_\theta\)</span></td><td>任意策略（重放缓冲区）</td></tr>
<tr><td><strong>更新之后</strong></td><td>旧数据失效，必须重新生成</td><td>旧数据仍可使用</td></tr>
<tr><td><strong>样本效率</strong></td><td>低（数据仅用一次）</td><td>高（数据可多次复用）</td></tr>
<tr><td><strong>稳定性</strong></td><td>更稳定（分布一致）</td><td>可能发散（分布不匹配）</td></tr>
<tr><td><strong>示例</strong></td><td>REINFORCE、PPO、A2C、GRPO</td><td>Q-Learning、DQN、SAC、DPO</td></tr>
<tr><td><strong>用于 LLM</strong></td><td>PPO、GRPO（每步重新生成）</td><td>DPO（静态偏好数据集）</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">RLHF 方法的同/异策略属性</p>
<p><strong>PPO/GRPO 是同策略</strong>：用当前策略生成回复，计算优势，更新，丢弃数据，再次生成。这就是为何生成占据 60% 算力——每一步都要重新生成。</p>

<p><strong>DPO 是异策略</strong>：在固定的偏好数据集上训练。训练过程中无需生成。便宜得多，但会遭受分布偏移问题（随策略变化数据逐渐过时）。</p>

<p><strong>在线 DPO 是混合体</strong>：生成新鲜数据（同策略生成），但使用 DPO 的监督损失（异策略式优化）。兼具两者优势。</p>

<p><strong>PPO 的巧妙之处</strong>：使用裁剪比率 <span class="arithmatex">\(r = \pi_\text{new}/\pi_\text{old}\)</span>，从同一批同策略数据中榨取多步梯度更新（通常 4 个 epoch），以可控方式使其变得「略带异策略」。</p>
</div>

## 3.10　基于模型 vs 无模型 {#model-based-vs-model-free}

<span id="table-3-inline-3"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>Model-Free</strong></th><th scope="col"><strong>Model-Based</strong></th></tr></thead>
<tbody><tr><td><strong>所学内容</strong></td><td>直接学习策略 <span class="arithmatex">\(\pi\)</span> 和/或价值 <span class="arithmatex">\(V\)</span>/<span class="arithmatex">\(Q\)</span></td><td>环境模型 <span class="arithmatex">\(\hat{P}(s&#x27;|s,a)\)</span></td></tr>
<tr><td><strong>规划能力</strong></td><td>无规划，反应式决策</td><td>可以模拟未来轨迹</td></tr>
<tr><td><strong>样本效率</strong></td><td>低（必须经历所有情况）</td><td>高（可在想象中规划）</td></tr>
<tr><td><strong>准确性</strong></td><td>无模型偏差</td><td>模型误差会累积</td></tr>
<tr><td><strong>何时使用</strong></td><td>复杂/未知的转移</td><td>简单转移，追求效率</td></tr>
<tr><td><strong>示例</strong></td><td>PPO、DQN、SAC <a class="book-citation" id="cite-haarnoja2018soft-1" href="../../references.md#ref-haarnoja2018soft" aria-label="参考文献 129">[129]</a></td><td>MuZero <a class="book-citation" id="cite-schrittwieser2020mastering-1" href="../../references.md#ref-schrittwieser2020mastering" aria-label="参考文献 316">[316]</a>、Dreamer <a class="book-citation" id="cite-hafner2020dream-1" href="../../references.md#ref-hafner2020dream" aria-label="参考文献 130">[130]</a>、AlphaGo <a class="book-citation" id="cite-silver2016mastering-1" href="../../references.md#ref-silver2016mastering" aria-label="参考文献 331">[331]</a></td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 LLM RL 是无模型的</p>
<p>语言生成的状态转移是平凡的（将 token 追加到序列——确定性转移）。环境「模型」并非瓶颈。难的是<strong>奖励</strong>——预测人类的偏好。这使得 model-based 方法对 LLM RL 来说没有必要。</p>

<p>RLHF 中的奖励模型在某种意义上也可以被看作一种「模型」（它预测人类偏好），但它被用作奖励信号，而非用于规划/模拟。LLM RL 本质上是无模型的策略优化。</p>
</div>

## 3.11　奖励塑形（Reward Shaping） {#reward-shaping}

<p><strong>奖励塑形（Reward Shaping）</strong> <a class="book-citation" id="cite-ng1999policy-1" href="../../references.md#ref-ng1999policy" aria-label="参考文献 265">[265]</a>是一种由开发者修改或补充环境原始奖励函数的技术。其主要目的是将<strong>稀疏奖励</strong>场景（智能体仅在任务最终完成时才获得反馈）转换为带有中间反馈信号的<strong>稠密奖励</strong>场景，以加速收敛。</p>

### 3.11.1　数学框架 {#the-mathematical-framework}

<p>设时间步 <span class="arithmatex">\(t\)</span> 的原始奖励为 <span class="arithmatex">\(R_t(s, a, s&#x27;)\)</span>。重塑后的奖励添加了一个辅助塑形函数 <span class="arithmatex">\(F\)</span>：</p>



<div class="arithmatex">\[\boxed{R&#x27;_t(s, a, s&#x27;) = R_t(s, a, s&#x27;) + F(s, a, s&#x27;)}
\tag{3.26}\]</div>

<div class="admonition warning"><p class="admonition-title">朴素塑形的风险：奖励黑客（Reward Hacking）</p>
<p>如果 <span class="arithmatex">\(F(s, a, s&#x27;)\)</span> 被随意设计，智能体会找到结构性漏洞来最大化辅助信号，同时忽略全局目标。</p>

<p><strong>示例</strong>：一个因到达中间地标而获得奖励的导航智能体可能学会无限绕着某个检查点循环以累积无限奖励——而永远不会到达目的地。</p>

<p>在 LLM 中：若模型因「听起来自信」而获得奖励，它可能学会无论准确与否都以「Absolutely!」开头。</p>
</div>

### 3.11.2　基于势函数的奖励塑形（Potential-Based Reward Shaping，PBRS） {#potential-based-reward-shaping-pbrs}

<p>为了从数学上保证塑形<strong>不会</strong>改变最优策略，可使用<strong>基于势函数的奖励塑形（PBRS）</strong>。塑形函数 <span class="arithmatex">\(F\)</span> 被约束为跨状态的标量势函数 <span class="arithmatex">\(\Phi\)</span> 之差：</p>



<div class="arithmatex">\[\boxed{F(s, a, s&#x27;) = \gamma\, \Phi(s&#x27;) - \Phi(s)}
\tag{3.27}\]</div>

<p>其中 <span class="arithmatex">\(\Phi: \mathcal{S} \to \mathbb{R}\)</span> 是一个实值势函数，衡量某状态相对目标的接近程度，<span class="arithmatex">\(\gamma\)</span> 是折扣因子。</p>

<p>完整的 PBRS 奖励：</p>



<div class="arithmatex">\[R&#x27;(s, a, s&#x27;) = R(s, a, s&#x27;) + \gamma\, \Phi(s&#x27;) - \Phi(s)
\tag{3.28}\]</div>

### 3.11.3　理论保证 {#theoretical-guarantees}

<div class="admonition note"><p class="admonition-title">PBRS 策略不变性定理</p>
<ul>
<li><p><strong>策略不变性</strong>：在重塑奖励 <span class="arithmatex">\(R&#x27;\)</span> 下的最优策略 <span class="arithmatex">\(\pi^*\)</span> 与原始奖励 <span class="arithmatex">\(R\)</span> 下的最优策略<strong>完全相同</strong>。塑形不会引入次优行为。</p></li>
<li><p><strong>回路免疫</strong>：任何从某状态出发又回到同一状态的循环轨迹，其净势能变化恰好为零（<span class="arithmatex">\(\Phi(s) - \Phi(s) = 0\)</span>）。智能体无法利用回路来作弊式刷奖励。</p></li>
<li><p><strong>收敛加速</strong>：尽管最优策略不变，塑形后的奖励提供了更稠密的梯度信号，使智能体在稀疏奖励环境下收敛速度提升 5–50 倍。</p></li>
</ul>
</div>
