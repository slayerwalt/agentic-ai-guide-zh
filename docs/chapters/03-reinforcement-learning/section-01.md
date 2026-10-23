# 3.1　马尔可夫决策过程（Markov Decision Process，MDP） {#the-markov-decision-process-mdp}

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
