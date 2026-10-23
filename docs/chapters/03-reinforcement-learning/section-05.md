# 3.5　Q-Learning {#q-learning}

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

## 3.5.1　理解重放缓冲区 {#understanding-replay-buffers}

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
