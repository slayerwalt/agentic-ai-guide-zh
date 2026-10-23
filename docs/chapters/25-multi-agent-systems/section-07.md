# 25.7　用强化学习训练多智能体系统 {#subsec:mas-rl-training}

<p>用 RL 训练多智能体系统会引入超出单 Agent RL 的挑战。根本难点在于：每个 Agent 的环境都包含其他正在学习的 Agent，从任一单 Agent 的视角看，环境是<strong>非平稳的（non-stationary）</strong>。</p>

## 25.7.1　数学形式化 {#mathematical-formulation}

<p>多智能体系统可形式化为一个<strong>马尔可夫博弈（Markov Game）</strong>，也称随机博弈（stochastic game） <a class="book-citation" id="cite-shapley1953stochastic-1" href="../../references.md#ref-shapley1953stochastic" aria-label="参考文献 324">[324]</a>：</p>



<div class="arithmatex">\[\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}_{i \in \mathcal{N}}, \mathcal{T}, \{R^i\}_{i \in \mathcal{N}}, \gamma \rangle
\tag{25.4}\]</div>

<p>其中 <span class="arithmatex">\(\mathcal{N} = \{1, \ldots, n\}\)</span> 是 Agent 集合，<span class="arithmatex">\(\mathcal{S}\)</span> 是共享状态空间，<span class="arithmatex">\(\mathcal{A}^i\)</span> 是 Agent <span class="arithmatex">\(i\)</span> 的动作空间，<span class="arithmatex">\(\mathcal{T}: \mathcal{S} \times \mathcal{A}^1 \times \cdots \times \mathcal{A}^n \to \Delta(\mathcal{S})\)</span> 是状态转移函数，<span class="arithmatex">\(R^i: \mathcal{S} \times \mathcal{A}^1 \times \cdots \times \mathcal{A}^n \to \mathbb{R}\)</span> 是 Agent <span class="arithmatex">\(i\)</span> 的 Reward 函数，<span class="arithmatex">\(\gamma\)</span> 是折扣因子。</p>

<p>每个 Agent <span class="arithmatex">\(i\)</span> 都力图最大化其期望折扣回报：</p>



<div class="arithmatex">\[J^i(\pi^1, \ldots, \pi^n) = \mathbb{E}_{\pi^1,\ldots,\pi^n}\left[\sum_{t=0}^{\infty} \gamma^t R^i(s_t, a_t^1, \ldots, a_t^n)\right]
\tag{25.5}\]</div>

## 25.7.2　独立学习（Independent Learning） {#independent-learning}

<p>最简单的方法：每个 Agent <span class="arithmatex">\(i\)</span> 将其他 Agent 视为环境的一部分，使用标准单 Agent RL（例如 PPO、REINFORCE）独立优化自己的 Policy <span class="arithmatex">\(\pi^i\)</span>。</p>



<div class="arithmatex">\[\nabla_{\theta^i} J^i \approx \mathbb{E}\left[\nabla_{\theta^i} \log \pi^i(a^i_t | o^i_t) \cdot \hat{A}^i_t\right]
\tag{25.6}\]</div>

<div class="admonition warning"><p class="admonition-title">非平稳性问题</p>
<p>独立学习违反了马尔可夫性假设：随着其他 Agent 更新其 Policy，Agent <span class="arithmatex">\(i\)</span> 看到的状态转移和 Reward 分布都会发生变化。这会导致训练不稳定、震荡乃至无法收敛。独立学习在简单合作任务上实际可行，但在竞争或复杂合作场景下表现挣扎。</p>
</div>

## 25.7.3　集中训练、去中心化执行（CTDE） {#centralized-training-decentralized-execution-ctde}

<p>集中训练、去中心化执行（Centralized Training, Decentralized Execution, CTDE） <a class="book-citation" id="cite-lowe2017multi-1" href="../../references.md#ref-lowe2017multi" aria-label="参考文献 239">[239]</a><a class="book-citation" id="cite-rashid2018qmix-1" href="../../references.md#ref-rashid2018qmix" aria-label="参考文献 306">[306]</a> 是合作型多 Agent RL 的主流范式。训练阶段，集中式 critic 可访问全局状态 <span class="arithmatex">\(s\)</span> 与所有 Agent 的动作 <span class="arithmatex">\(\mathbf{a} = (a^1, \ldots, a^n)\)</span>。执行阶段，每个 Agent 仅基于其局部观测 <span class="arithmatex">\(o^i\)</span> 行动。</p>

<p>Agent <span class="arithmatex">\(i\)</span> 的集中式 critic：</p>



<div class="arithmatex">\[Q^i_\phi(s, \mathbf{a}) = Q^i_\phi(s, a^1, \ldots, a^n)
\tag{25.7}\]</div>

<p>Agent <span class="arithmatex">\(i\)</span> 的去中心化 actor：</p>



<div class="arithmatex">\[\pi^i_{\theta^i}(a^i | o^i)
\tag{25.8}\]</div>

<p>带有集中式 critic 的策略梯度：</p>



<div class="arithmatex">\[\nabla_{\theta^i} J^i = \mathbb{E}\left[\nabla_{\theta^i} \log \pi^i(a^i | o^i) \cdot Q^i_\phi(s, \mathbf{a})\right]
\tag{25.9}\]</div>

<p>CTDE 在训练时解决了非平稳性问题（集中式 critic 看到完整的联合状态），同时在推理时保留了去中心化执行（无需通信）。</p>

## 25.7.4　通信学习（Communication Learning） {#communication-learning}

<p>与使用固定通信协议不同，Agent 可以<strong>学习要通信什么</strong>。在可微通信框架 <a class="book-citation" id="cite-sukhbaatar2016learning-1" href="../../references.md#ref-sukhbaatar2016learning" aria-label="参考文献 343">[343]</a><a class="book-citation" id="cite-das2019tarmac-1" href="../../references.md#ref-das2019tarmac" aria-label="参考文献 66">[66]</a> 中，Agent 产生连续的通信向量 <span class="arithmatex">\(m^i_t\)</span> 并传递给其他 Agent：</p>



<div class="arithmatex">\[a^i_t, m^i_t = \pi^i_{\theta^i}(o^i_t, \{m^j_{t-1}\}_{j \neq i})
\tag{25.10}\]</div>

<p>通信向量通过联合 Reward 信号的反向传播进行端到端优化。对于 LLM Agent，这一思想被近似为训练 Agent 产生最大化任务表现的结构化自然语言消息。</p>

## 25.7.5　涌现通信（Emergent Communication） {#emergent-communication}

<p>当 Agent 仅依靠 Reward 信号（没有预定义语言）从零训练时，它们可以发展出<strong>涌现通信协议</strong> <a class="book-citation" id="cite-lazaridou2020emergent-1" href="../../references.md#ref-lazaridou2020emergent" aria-label="参考文献 199">[199]</a>：编码任务相关信息的共享符号系统。虽然在科学上令人着迷，但 LLM 系统中的涌现通信通常并不可取——我们希望 Agent 使用人类可理解的语言进行通信。</p>

## 25.7.6　自我对弈（Self-Play） {#self-play}

<p>在竞争或混合动机场景下，<strong>自我对弈（self-play）</strong> <a class="book-citation" id="cite-silver2017mastering-1" href="../../references.md#ref-silver2017mastering" aria-label="参考文献 332">[332]</a> 通过让 Agent 与自身的副本对抗来训练。这会自动产生课程式学习：随着 Agent 的提升，其对手（自身的早期版本）变得越来越难以击败。</p>

<p>对于 LLM Agent，自我对弈被用于：</p>

<ul>
<li><p>红队 vs. 蓝队训练</p></li>
<li><p>辩论训练（Agent 相互辩论）</p></li>
<li><p>谈判训练（Agent 相互谈判）</p></li>
</ul>

## 25.7.7　基于种群的训练（Population-Based Training） {#population-based-training}

<p><strong>基于种群的训练（Population-Based Training, PBT）</strong> <a class="book-citation" id="cite-jaderberg2019human-1" href="../../references.md#ref-jaderberg2019human" aria-label="参考文献 158">[158]</a> 维护一个具有不同 Policy、超参数和专长的多样化 Agent 种群。Agent 会被定期评估；表现不佳的 Agent 会被表现优秀 Agent 的变异副本所替换。</p>

<p>对于多 Agent LLM 系统，PBT 实现了：</p>

<ul>
<li><p>有效角色专长的自动发现</p></li>
<li><p>对个别 Agent 故障的鲁棒性（多样化种群）</p></li>
<li><p>通过种群多样性避开局部最优</p></li>
</ul>

## 25.7.8　社会福利与 Nash 均衡 {#social-welfare-and-nash-equilibrium}

<p>在多 Agent 场景下，最优性的概念比单 Agent 场景更复杂。两个关键的解概念：</p>

<p><strong>Nash 均衡（Nash Equilibrium）</strong>：一个联合 Policy <span class="arithmatex">\((\pi^{1*}, \ldots, \pi^{n*})\)</span>，使得没有任何 Agent 可以通过单方面偏离来提高其期望回报：</p>



<div class="arithmatex">\[J^i(\pi^{i*}, \pi^{-i*}) \geq J^i(\pi^i, \pi^{-i*}) \quad \forall i, \forall \pi^i
\tag{25.11}\]</div>

<p>其中 <span class="arithmatex">\(\pi^{-i}\)</span> 表示除 <span class="arithmatex">\(i\)</span> 之外所有 Agent 的联合 Policy。</p>

<p><strong>社会福利最大化（Social Welfare Maximization）</strong>：优化所有 Agent 回报之和：</p>



<div class="arithmatex">\[\max_{\pi^1,\ldots,\pi^n} \sum_{i=1}^{n} J^i(\pi^1, \ldots, \pi^n)
\tag{25.12}\]</div>

<p>在完全合作场景下（所有 Agent 共享同一 Reward），社会福利最大化是合适的目标。在竞争场景下，Nash 均衡是相关的解概念。大多数现实世界的多 Agent LLM 系统是<strong>混合动机（mixed-motive）</strong>的：Agent 之间的目标部分一致、部分冲突。</p>

<div class="admonition note"><p class="admonition-title">延伸阅读：多 Agent RL 的博弈论基础</p>
<p>对于有兴趣了解多智能体系统博弈论基础的读者：</p>

<ul>
<li><p><strong>Shoham &amp; Leyton-Brown</strong> <a class="book-citation" id="cite-shoham2008multiagent-1" href="../../references.md#ref-shoham2008multiagent" aria-label="参考文献 328">[328]</a> —— 全面的教科书，涵盖 Agent 系统的 Nash 均衡、机制设计与社会选择理论。</p></li>
<li><p><strong>Zhang et al.</strong> <a class="book-citation" id="cite-zhang2021multiagent-1" href="../../references.md#ref-zhang2021multiagent" aria-label="参考文献 423">[423]</a> —— 综述了在合作、竞争与混合场景下具有收敛保证的多 Agent RL 算法。</p></li>
<li><p><strong>Nisan et al.</strong> <a class="book-citation" id="cite-nisan2007algorithmic-1" href="../../references.md#ref-nisan2007algorithmic" aria-label="参考文献 267">[267]</a> —— 算法博弈论的权威参考，涵盖拍卖、均衡计算与无政府代价（price of anarchy）。</p></li>
</ul>
</div>
