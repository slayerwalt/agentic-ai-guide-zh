# 13.6　关键方法及其数学基础 {#subsec:reasoning_methods}

## 13.6.1　用于推理的蒙特卡洛树搜索 {#subsubsec:mcts_reasoning}

<p>蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）为将推理视作树搜索提供了一个有原则的框架。在 AlphaProof <a class="book-citation" id="cite-alphaproof2024-1" href="../../references.md#ref-alphaproof2024" aria-label="参考文献 68">[68]</a> 等相关系统中，MCTS 作用于推理步骤而非棋类走子。</p>

#### 状态与动作空间 {#state-and-action-space}

<ul>
<li><p><strong>状态</strong> <span class="arithmatex">\(s_k\)</span>：部分推理链 <span class="arithmatex">\((q, r_1, r_2, \ldots, r_k)\)</span>，其中 <span class="arithmatex">\(r_i\)</span> 为推理步骤</p></li>
<li><p><strong>动作</strong> <span class="arithmatex">\(a\)</span>：下一个推理步骤（一句话或一段话）</p></li>
<li><p><strong>终止状态</strong>：包含最终答案的状态</p></li>
<li><p><strong>奖励</strong>：<span class="arithmatex">\(R(s_{\text{terminal}}) = r_{\text{acc}}\)</span>（见公式 [eq:r1_accuracy_reward]）</p></li>
</ul>

#### 部分解的价值函数 {#value-function-for-partial-solutions}

<p>价值函数 <span class="arithmatex">\(V(s_k)\)</span> 估计从部分状态 <span class="arithmatex">\(s_k\)</span> 出发达到正确答案的概率：</p>



<div class="arithmatex">\[V(s_k) = P(\text{correct answer} \mid s_k) \approx \frac{1}{M} \sum_{m=1}^{M} R(\text{rollout}_m(s_k))
\tag{13.30}\]</div>

<p>其中 <span class="arithmatex">\(\text{rollout}_m(s_k)\)</span> 是从 <span class="arithmatex">\(s_k\)</span> 出发、使用当前 Policy 走到终止状态的一次蒙特卡洛 rollout。</p>

#### UCB 探索 {#ucb-exploration}

<p>节点选择使用为推理场景改编的置信上界（Upper Confidence Bound, UCB）公式：</p>



<div class="arithmatex">\[\text{UCB}(s_k, a) = Q(s_k, a) + c_{\text{puct}} \cdot \pi_\theta(a \mid s_k) \cdot \frac{\sqrt{N(s_k)}}{1 + N(s_k, a)}
\tag{13.31}\]</div>

<p>其中：</p>

<ul>
<li><p><span class="arithmatex">\(Q(s_k, a) = \frac{1}{N(s_k,a)} \sum_{\text{visits}} V(s_{k+1})\)</span> 为子状态的平均价值</p></li>
<li><p><span class="arithmatex">\(\pi_\theta(a \mid s_k)\)</span> 为 Policy 先验（语言模型对步骤 <span class="arithmatex">\(a\)</span> 的概率）</p></li>
<li><p><span class="arithmatex">\(N(s_k)\)</span> 为状态 <span class="arithmatex">\(s_k\)</span> 的访问次数</p></li>
<li><p><span class="arithmatex">\(N(s_k, a)\)</span> 为边 <span class="arithmatex">\((s_k, a)\)</span> 的访问次数</p></li>
<li><p><span class="arithmatex">\(c_{\text{puct}}\)</span> 为探索常数</p></li>
</ul>

#### MCTS 引导的训练 {#mcts-guided-training}

<p>MCTS 可被用于生成高质量训练数据：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{MCTS}}(\theta) = -\sum_{k} \sum_{a} \pi_{\text{MCTS}}(a \mid s_k) \log \pi_\theta(a \mid s_k)
\tag{13.32}\]</div>

<p>其中 <span class="arithmatex">\(\pi_{\text{MCTS}}(a \mid s_k) \propto N(s_k, a)^{1/\tau}\)</span> 是 MCTS Policy（带温度 <span class="arithmatex">\(\tau\)</span> 的访问计数分布）。</p>

## 13.6.2　过程奖励模型 {#subsubsec:prm_methods}

#### Math-Shepherd：自动化 PRM 训练 {#math-shepherd-automated-prm-training}

<p>Math-Shepherd <a class="book-citation" id="cite-wang2024mathshepherd-1" href="../../references.md#ref-wang2024mathshepherd" aria-label="参考文献 365">[365]</a> 提出了一种无需人工步骤级标注、自动训练 PRM 的方法。其核心洞见是使用<strong>基于结果的估计（outcome-based estimation）</strong>：如果存在从 <span class="arithmatex">\(s_k\)</span> 出发的某一补全能够到达正确答案，则将步骤 <span class="arithmatex">\(s_k\)</span> 标注为正确。</p>

<p>形式化地，对于部分解 <span class="arithmatex">\((s_1, \ldots, s_k)\)</span>：</p>



<div class="arithmatex">\[\hat{r}_k = \mathbf{1}\!\left[\exists\, (s_{k+1}, \ldots, s_K) : \text{verify}(s_K, y^*) = 1\right]
\tag{13.33}\]</div>

<p>在实践中，这通过从 <span class="arithmatex">\(s_k\)</span> 采样 <span class="arithmatex">\(M\)</span> 个补全并检查是否有任何一个正确来估计：</p>



<div class="arithmatex">\[\hat{r}_k \approx \mathbf{1}\!\left[\sum_{m=1}^{M} \text{verify}(\text{complete}_m(s_k), y^*) &gt; 0\right]
\tag{13.34}\]</div>

<p>随后用二元交叉熵训练 PRM：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{PRM}}(\phi) = -\sum_{k=1}^{K} \left[ \hat{r}_k \log r_\phi(s_k) + (1-\hat{r}_k) \log(1 - r_\phi(s_k)) \right]
\tag{13.35}\]</div>

#### 用 PRM 进行 Best-of-N 选择 {#prm-for-best-of-n-selection}

<p>PRM 的一个主要应用是 <strong>Best-of-N 选择</strong>：生成 <span class="arithmatex">\(N\)</span> 个候选解，选择 PRM 分数最高的那一个：</p>



<div class="arithmatex">\[y^* = \arg\max_{y \in \{y_1, \ldots, y_N\}} R_{\text{PRM}}(q, y)
\tag{13.36}\]</div>

<p>这比（使用 ORM 的）多数投票更有效，因为 PRM 能够区分通过不同质量推理路径达到相同答案的解。</p>

## 13.6.3　结果奖励模型与多数投票 {#outcome-reward-models-and-majority-voting}

#### 多数投票（自洽性，Self-Consistency） {#majority-voting-self-consistency}

<p>测试时扩展（Test-Time Scaling）最简单的形式是多数投票 <a class="book-citation" id="cite-wang2023selfconsistency-1" href="../../references.md#ref-wang2023selfconsistency" aria-label="参考文献 367">[367]</a>：生成 <span class="arithmatex">\(N\)</span> 个解并返回最常见的答案：</p>



<div class="arithmatex">\[y^* = \arg\max_{a} \sum_{i=1}^{N} \mathbf{1}[y_i = a]
\tag{13.37}\]</div>

<p>在每个解独立正确概率为 <span class="arithmatex">\(p &gt; 0.5\)</span> 的假设下，多数投票正确的概率为：</p>



<div class="arithmatex">\[P(\text{majority correct}) = \sum_{k=\lceil N/2 \rceil}^{N} \binom{N}{k} p^k (1-p)^{N-k} \xrightarrow{N \to \infty} 1
\tag{13.38}\]</div>

#### 用 ORM 加权的多数投票 {#weighted-majority-voting-with-orm}

<p>ORM 可以通过按置信度加权来改进多数投票：</p>



<div class="arithmatex">\[y^* = \arg\max_{a} \sum_{i=1}^{N} R_{\text{ORM}}(q, y_i) \cdot \mathbf{1}[y_i = a]
\tag{13.39}\]</div>

## 13.6.4　用于推理的自博弈 {#self-play-for-reasoning}

<p>自博弈（self-play）方法让模型同时扮演<em>生成器</em>和<em>验证器</em>两种角色，从而生成训练数据。</p>

#### STaR：自学推理者（Self-Taught Reasoner） {#star-self-taught-reasoner}

<p>STaR <a class="book-citation" id="cite-zelikman2022star-1" href="../../references.md#ref-zelikman2022star" aria-label="参考文献 418">[418]</a> 通过迭代方式自举推理能力：</p>

<ol>
<li><p>为问题集生成推理链</p></li>
<li><p>保留导向正确答案的推理链（拒绝采样）</p></li>
<li><p>在保留的推理链上微调</p></li>
<li><p>用改进后的模型重复上述过程</p></li>
</ol>

<p>其核心洞见是模型可以为正确答案进行<em>合理化（rationalize）</em>：即便它无法从零开始解决一个问题，给定答案后它仍能生成一条合理的推理链，这条推理链可作为训练数据。</p>

#### 自博弈 RL {#self-play-rl}

<p>在用于推理的自博弈 RL 中，模型同时生成问题与解答：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{self-play}}(\theta) = \mathbb{E}_{q \sim \pi_\theta^{\text{gen}}} \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot|q)} \left[ r(y, y^*) \right]
\tag{13.40}\]</div>

<p>其中 <span class="arithmatex">\(\pi_\theta^{\text{gen}}\)</span> 生成问题，<span class="arithmatex">\(\pi_\theta^{\text{solve}}\)</span> 解答它们。生成器因产生具有挑战性但可解的问题而获得奖励。</p>

## 13.6.5　可验证奖励的强化学习（RLVR） {#subsubsec:rlvr}

<p>可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR） <a class="book-citation" id="cite-lambert2024tulu3-1" href="../../references.md#ref-lambert2024tulu3" aria-label="参考文献 198">[198]</a> 是一个使用<strong>真值验证（ground-truth verification）</strong>作为奖励信号的框架，适用于任何可以自动检查正确性的领域。</p>

#### 可验证领域 {#verifiable-domains}

<ul>
<li><p><strong>数学</strong>：通过 SymPy、Lean 或 Isabelle 进行符号验证</p></li>
<li><p><strong>代码</strong>：单元测试执行</p></li>
<li><p><strong>形式逻辑</strong>：证明检查</p></li>
<li><p><strong>事实问答</strong>：数据库查找</p></li>
<li><p><strong>游戏</strong>：胜负结果</p></li>
</ul>

#### RLVR 目标 {#rlvr-objective}



<div class="arithmatex">\[\mathcal{L}_{\text{RLVR}}(\theta) = -\mathbb{E}_{(q, y^*) \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_\theta(\cdot|q)} \left[ \text{verify}(y, y^*) \right] + \beta \mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta \,\|\, \pi_{\text{ref}}\right]
\tag{13.41}\]</div>

<p>RLVR 相对于基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）的关键优势在于<strong>不存在奖励模型误差</strong>：由于奖励由确定性验证器而非学习得到的模型计算，因此不存在针对有缺陷奖励模型的奖励作弊。唯一的失败模式是模型找到了能通过验证、但并非真正正确的解（例如，利用代码评估中测试用例的弱点）。</p>

<div class="admonition example"><p class="admonition-title">用于代码的 RLVR：奖励作弊挑战</p>
<p>在代码生成中，验证器是一个测试套件。用 RLVR 训练的模型可能学会：</p>

<ul>
<li><p><strong>硬编码测试输出</strong>：对每个测试输入返回预期输出，而不实现真正的算法</p></li>
<li><p><strong>利用薄弱测试</strong>：通过所有提供的测试，但在边界情形下失败</p></li>
</ul>

<p>缓解措施包括：使用大规模、多样化的测试套件；引入对抗性测试用例；使用基于执行的奖励来惩罚硬编码（例如，检查解是否在 <span class="arithmatex">\(O(n \log n)\)</span> 时间内运行）。</p>
</div>

## 13.6.6　旅程学习（Journey Learning） {#journey-learning}

<p>旅程学习（Journey Learning） <a class="book-citation" id="cite-qin2024o1journey-1" href="../../references.md#ref-qin2024o1journey" aria-label="参考文献 298">[298]</a> 主张在<strong>完整推理轨迹</strong>上训练，包括失败的尝试和纠正，而不仅仅是成功的最终解。</p>

#### 动机 {#motivation-dup4}

<p>标准的拒绝采样会丢弃失败尝试。但失败尝试包含有价值的信息：</p>

<ul>
<li><p>哪些方法行不通（负样本）</p></li>
<li><p>如何识别错误并从中恢复（纠正模式）</p></li>
<li><p>问题空间的结构（探索数据）</p></li>
</ul>

#### Journey Learning 目标 {#journey-learning-objective}

<p>给定一条可能包含回溯的轨迹 <span class="arithmatex">\(\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)\)</span>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{journey}}(\theta) = -\sum_{t=0}^{T} w_t \log \pi_\theta(a_t \mid s_t)
\tag{13.42}\]</div>

<p>其中权重 <span class="arithmatex">\(w_t\)</span> 被设计为强调：</p>

<ul>
<li><p>最终导向成功的步骤（<span class="arithmatex">\(w_t &gt; 1\)</span>）</p></li>
<li><p>错误之后的纠正步骤（<span class="arithmatex">\(w_t &gt; 1\)</span>）</p></li>
<li><p>失败分支中的步骤（<span class="arithmatex">\(w_t &lt; 1\)</span>，但 <span class="arithmatex">\(&gt; 0\)</span>）</p></li>
</ul>

## 13.6.7　Quiet-STaR：在每个 Token 上推理 {#quiet-star-reasoning-at-every-token}

<p>Quiet-STaR <a class="book-citation" id="cite-zelikman2024quietstar-1" href="../../references.md#ref-zelikman2024quietstar" aria-label="参考文献 419">[419]</a> 将推理范式扩展到<strong>每一个 Token 位置</strong>：模型不仅在最终答案之前生成一条推理链，而是在每个 Token 位置都生成一段“思考（thought）”。</p>

#### 形式化 {#formulation}

<p>对于每个 Token 位置 <span class="arithmatex">\(t\)</span>，模型在预测下一个 Token <span class="arithmatex">\(x_{t+1}\)</span> 之前先生成一段隐藏思考 <span class="arithmatex">\(z_t\)</span>：</p>



<div class="arithmatex">\[P(x_{t+1} \mid x_{\leq t}) = \mathbb{E}_{z_t \sim \pi_\theta(\cdot | x_{\leq t})} \left[ \pi_\theta(x_{t+1} \mid x_{\leq t}, z_t) \right]
\tag{13.43}\]</div>

<p>在实践中，这通过混合带思考与不带思考的预测来近似：</p>



<div class="arithmatex">\[P(x_{t+1} \mid x_{\leq t}) = \alpha \cdot \pi_\theta(x_{t+1} \mid x_{\leq t}, z_t) + (1-\alpha) \cdot \pi_\theta(x_{t+1} \mid x_{\leq t})
\tag{13.44}\]</div>

#### 使用 REINFORCE 训练 {#training-with-reinforce}

<p>由于思考 <span class="arithmatex">\(z_t\)</span> 是离散潜变量，梯度使用 REINFORCE 估计：</p>



<div class="arithmatex">\[\nabla_\theta \mathcal{L}_{\text{QS}} = \mathbb{E}_{z_t} \left[ \nabla_\theta \log \pi_\theta(z_t \mid x_{\leq t}) \cdot \left( \log P(x_{t+1} \mid x_{\leq t}, z_t) - b_t \right) \right]
\tag{13.45}\]</div>

<p>其中 <span class="arithmatex">\(b_t\)</span> 为基线（例如无思考预测 <span class="arithmatex">\(\log \pi_\theta(x_{t+1} \mid x_{\leq t})\)</span>）。</p>

<div class="admonition warning"><p class="admonition-title">Quiet-STaR 的计算代价</p>
<p>Quiet-STaR 将推理成本提升为 <span class="arithmatex">\(L_z + 1\)</span> 倍（<span class="arithmatex">\(L_z\)</span> 为思考长度），且作用于<em>每一个</em> Token 位置。对于长度为 <span class="arithmatex">\(T\)</span> 的序列，若思考长度 <span class="arithmatex">\(L_z = 8\)</span>，计算量增加 <span class="arithmatex">\(9\times\)</span>。这使得 Quiet-STaR 在长序列上若无重大工程优化（如对思考的投机解码、缓存等）便不可行。</p>
</div>
