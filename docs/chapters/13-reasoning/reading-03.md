# 第 13 章 · 数学方法、扩展规律与开放问题 {#reading-reading-03}

## 13.6　关键方法及其数学基础 {#subsec:reasoning_methods}

### 13.6.1　用于推理的蒙特卡洛树搜索 {#subsubsec:mcts_reasoning}

<p>蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）为将推理视作树搜索提供了一个有原则的框架。在 AlphaProof <a class="book-citation" id="cite-alphaproof2024-1" href="../../references.md#ref-alphaproof2024" aria-label="参考文献 68">[68]</a> 等相关系统中，MCTS 作用于推理步骤而非棋类走子。</p>

##### 状态与动作空间 {#state-and-action-space}

<ul>
<li><p><strong>状态</strong> <span class="arithmatex">\(s_k\)</span>：部分推理链 <span class="arithmatex">\((q, r_1, r_2, \ldots, r_k)\)</span>，其中 <span class="arithmatex">\(r_i\)</span> 为推理步骤</p></li>
<li><p><strong>动作</strong> <span class="arithmatex">\(a\)</span>：下一个推理步骤（一句话或一段话）</p></li>
<li><p><strong>终止状态</strong>：包含最终答案的状态</p></li>
<li><p><strong>奖励</strong>：<span class="arithmatex">\(R(s_{\text{terminal}}) = r_{\text{acc}}\)</span>（见公式 [eq:r1_accuracy_reward]）</p></li>
</ul>

##### 部分解的价值函数 {#value-function-for-partial-solutions}

<p>价值函数 <span class="arithmatex">\(V(s_k)\)</span> 估计从部分状态 <span class="arithmatex">\(s_k\)</span> 出发达到正确答案的概率：</p>



<div class="arithmatex">\[V(s_k) = P(\text{correct answer} \mid s_k) \approx \frac{1}{M} \sum_{m=1}^{M} R(\text{rollout}_m(s_k))
\tag{13.30}\]</div>

<p>其中 <span class="arithmatex">\(\text{rollout}_m(s_k)\)</span> 是从 <span class="arithmatex">\(s_k\)</span> 出发、使用当前 Policy 走到终止状态的一次蒙特卡洛 rollout。</p>

##### UCB 探索 {#ucb-exploration}

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

##### MCTS 引导的训练 {#mcts-guided-training}

<p>MCTS 可被用于生成高质量训练数据：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{MCTS}}(\theta) = -\sum_{k} \sum_{a} \pi_{\text{MCTS}}(a \mid s_k) \log \pi_\theta(a \mid s_k)
\tag{13.32}\]</div>

<p>其中 <span class="arithmatex">\(\pi_{\text{MCTS}}(a \mid s_k) \propto N(s_k, a)^{1/\tau}\)</span> 是 MCTS Policy（带温度 <span class="arithmatex">\(\tau\)</span> 的访问计数分布）。</p>

### 13.6.2　过程奖励模型 {#subsubsec:prm_methods}

##### Math-Shepherd：自动化 PRM 训练 {#math-shepherd-automated-prm-training}

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

##### 用 PRM 进行 Best-of-N 选择 {#prm-for-best-of-n-selection}

<p>PRM 的一个主要应用是 <strong>Best-of-N 选择</strong>：生成 <span class="arithmatex">\(N\)</span> 个候选解，选择 PRM 分数最高的那一个：</p>



<div class="arithmatex">\[y^* = \arg\max_{y \in \{y_1, \ldots, y_N\}} R_{\text{PRM}}(q, y)
\tag{13.36}\]</div>

<p>这比（使用 ORM 的）多数投票更有效，因为 PRM 能够区分通过不同质量推理路径达到相同答案的解。</p>

### 13.6.3　结果奖励模型与多数投票 {#outcome-reward-models-and-majority-voting}

##### 多数投票（自洽性，Self-Consistency） {#majority-voting-self-consistency}

<p>测试时扩展（Test-Time Scaling）最简单的形式是多数投票 <a class="book-citation" id="cite-wang2023selfconsistency-1" href="../../references.md#ref-wang2023selfconsistency" aria-label="参考文献 367">[367]</a>：生成 <span class="arithmatex">\(N\)</span> 个解并返回最常见的答案：</p>



<div class="arithmatex">\[y^* = \arg\max_{a} \sum_{i=1}^{N} \mathbf{1}[y_i = a]
\tag{13.37}\]</div>

<p>在每个解独立正确概率为 <span class="arithmatex">\(p &gt; 0.5\)</span> 的假设下，多数投票正确的概率为：</p>



<div class="arithmatex">\[P(\text{majority correct}) = \sum_{k=\lceil N/2 \rceil}^{N} \binom{N}{k} p^k (1-p)^{N-k} \xrightarrow{N \to \infty} 1
\tag{13.38}\]</div>

##### 用 ORM 加权的多数投票 {#weighted-majority-voting-with-orm}

<p>ORM 可以通过按置信度加权来改进多数投票：</p>



<div class="arithmatex">\[y^* = \arg\max_{a} \sum_{i=1}^{N} R_{\text{ORM}}(q, y_i) \cdot \mathbf{1}[y_i = a]
\tag{13.39}\]</div>

### 13.6.4　用于推理的自博弈 {#self-play-for-reasoning}

<p>自博弈（self-play）方法让模型同时扮演<em>生成器</em>和<em>验证器</em>两种角色，从而生成训练数据。</p>

##### STaR：自学推理者（Self-Taught Reasoner） {#star-self-taught-reasoner}

<p>STaR <a class="book-citation" id="cite-zelikman2022star-1" href="../../references.md#ref-zelikman2022star" aria-label="参考文献 418">[418]</a> 通过迭代方式自举推理能力：</p>

<ol>
<li><p>为问题集生成推理链</p></li>
<li><p>保留导向正确答案的推理链（拒绝采样）</p></li>
<li><p>在保留的推理链上微调</p></li>
<li><p>用改进后的模型重复上述过程</p></li>
</ol>

<p>其核心洞见是模型可以为正确答案进行<em>合理化（rationalize）</em>：即便它无法从零开始解决一个问题，给定答案后它仍能生成一条合理的推理链，这条推理链可作为训练数据。</p>

##### 自博弈 RL {#self-play-rl}

<p>在用于推理的自博弈 RL 中，模型同时生成问题与解答：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{self-play}}(\theta) = \mathbb{E}_{q \sim \pi_\theta^{\text{gen}}} \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot|q)} \left[ r(y, y^*) \right]
\tag{13.40}\]</div>

<p>其中 <span class="arithmatex">\(\pi_\theta^{\text{gen}}\)</span> 生成问题，<span class="arithmatex">\(\pi_\theta^{\text{solve}}\)</span> 解答它们。生成器因产生具有挑战性但可解的问题而获得奖励。</p>

### 13.6.5　可验证奖励的强化学习（RLVR） {#subsubsec:rlvr}

<p>可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR） <a class="book-citation" id="cite-lambert2024tulu3-1" href="../../references.md#ref-lambert2024tulu3" aria-label="参考文献 198">[198]</a> 是一个使用<strong>真值验证（ground-truth verification）</strong>作为奖励信号的框架，适用于任何可以自动检查正确性的领域。</p>

##### 可验证领域 {#verifiable-domains}

<ul>
<li><p><strong>数学</strong>：通过 SymPy、Lean 或 Isabelle 进行符号验证</p></li>
<li><p><strong>代码</strong>：单元测试执行</p></li>
<li><p><strong>形式逻辑</strong>：证明检查</p></li>
<li><p><strong>事实问答</strong>：数据库查找</p></li>
<li><p><strong>游戏</strong>：胜负结果</p></li>
</ul>

##### RLVR 目标 {#rlvr-objective}



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

### 13.6.6　旅程学习（Journey Learning） {#journey-learning}

<p>旅程学习（Journey Learning） <a class="book-citation" id="cite-qin2024o1journey-1" href="../../references.md#ref-qin2024o1journey" aria-label="参考文献 298">[298]</a> 主张在<strong>完整推理轨迹</strong>上训练，包括失败的尝试和纠正，而不仅仅是成功的最终解。</p>

##### 动机 {#motivation-dup4}

<p>标准的拒绝采样会丢弃失败尝试。但失败尝试包含有价值的信息：</p>

<ul>
<li><p>哪些方法行不通（负样本）</p></li>
<li><p>如何识别错误并从中恢复（纠正模式）</p></li>
<li><p>问题空间的结构（探索数据）</p></li>
</ul>

##### Journey Learning 目标 {#journey-learning-objective}

<p>给定一条可能包含回溯的轨迹 <span class="arithmatex">\(\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)\)</span>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{journey}}(\theta) = -\sum_{t=0}^{T} w_t \log \pi_\theta(a_t \mid s_t)
\tag{13.42}\]</div>

<p>其中权重 <span class="arithmatex">\(w_t\)</span> 被设计为强调：</p>

<ul>
<li><p>最终导向成功的步骤（<span class="arithmatex">\(w_t &gt; 1\)</span>）</p></li>
<li><p>错误之后的纠正步骤（<span class="arithmatex">\(w_t &gt; 1\)</span>）</p></li>
<li><p>失败分支中的步骤（<span class="arithmatex">\(w_t &lt; 1\)</span>，但 <span class="arithmatex">\(&gt; 0\)</span>）</p></li>
</ul>

### 13.6.7　Quiet-STaR：在每个 Token 上推理 {#quiet-star-reasoning-at-every-token}

<p>Quiet-STaR <a class="book-citation" id="cite-zelikman2024quietstar-1" href="../../references.md#ref-zelikman2024quietstar" aria-label="参考文献 419">[419]</a> 将推理范式扩展到<strong>每一个 Token 位置</strong>：模型不仅在最终答案之前生成一条推理链，而是在每个 Token 位置都生成一段“思考（thought）”。</p>

##### 形式化 {#formulation}

<p>对于每个 Token 位置 <span class="arithmatex">\(t\)</span>，模型在预测下一个 Token <span class="arithmatex">\(x_{t+1}\)</span> 之前先生成一段隐藏思考 <span class="arithmatex">\(z_t\)</span>：</p>



<div class="arithmatex">\[P(x_{t+1} \mid x_{\leq t}) = \mathbb{E}_{z_t \sim \pi_\theta(\cdot | x_{\leq t})} \left[ \pi_\theta(x_{t+1} \mid x_{\leq t}, z_t) \right]
\tag{13.43}\]</div>

<p>在实践中，这通过混合带思考与不带思考的预测来近似：</p>



<div class="arithmatex">\[P(x_{t+1} \mid x_{\leq t}) = \alpha \cdot \pi_\theta(x_{t+1} \mid x_{\leq t}, z_t) + (1-\alpha) \cdot \pi_\theta(x_{t+1} \mid x_{\leq t})
\tag{13.44}\]</div>

##### 使用 REINFORCE 训练 {#training-with-reinforce}

<p>由于思考 <span class="arithmatex">\(z_t\)</span> 是离散潜变量，梯度使用 REINFORCE 估计：</p>



<div class="arithmatex">\[\nabla_\theta \mathcal{L}_{\text{QS}} = \mathbb{E}_{z_t} \left[ \nabla_\theta \log \pi_\theta(z_t \mid x_{\leq t}) \cdot \left( \log P(x_{t+1} \mid x_{\leq t}, z_t) - b_t \right) \right]
\tag{13.45}\]</div>

<p>其中 <span class="arithmatex">\(b_t\)</span> 为基线（例如无思考预测 <span class="arithmatex">\(\log \pi_\theta(x_{t+1} \mid x_{\leq t})\)</span>）。</p>

<div class="admonition warning"><p class="admonition-title">Quiet-STaR 的计算代价</p>
<p>Quiet-STaR 将推理成本提升为 <span class="arithmatex">\(L_z + 1\)</span> 倍（<span class="arithmatex">\(L_z\)</span> 为思考长度），且作用于<em>每一个</em> Token 位置。对于长度为 <span class="arithmatex">\(T\)</span> 的序列，若思考长度 <span class="arithmatex">\(L_z = 8\)</span>，计算量增加 <span class="arithmatex">\(9\times\)</span>。这使得 Quiet-STaR 在长序列上若无重大工程优化（如对思考的投机解码、缓存等）便不可行。</p>
</div>

## 13.7　推理的扩展律 {#subsec:reasoning_scaling}

<p>近期工作 <a class="book-citation" id="cite-snell2024scaling-1" href="../../references.md#ref-snell2024scaling" aria-label="参考文献 335">[335]</a><a class="book-citation" id="cite-wu2024empirical-1" href="../../references.md#ref-wu2024empirical" aria-label="参考文献 388">[388]</a> 已经证实测试时计算量与推理性能之间存在可预测的扩展关系，将经典扩展律 <a class="book-citation" id="cite-kaplan2020scaling-1" href="../../references.md#ref-kaplan2020scaling" aria-label="参考文献 177">[177]</a> 延伸到了推理阶段。</p>

### 13.7.1　训练计算量与测试时计算量的权衡 {#training-compute-vs.-test-time-compute-tradeoff}

<p>推理模型的根本性扩展问题是：<strong>给定固定的总计算预算 <span class="arithmatex">\(C_{\text{total}} = C_{\text{train}} + N \cdot C_{\text{test}}\)</span>（其中 <span class="arithmatex">\(N\)</span> 为查询数量），应如何分配计算？</strong></p>

<p>设 <span class="arithmatex">\(\mathcal{A}(C_{\text{train}}, C_{\text{test}})\)</span> 表示用 <span class="arithmatex">\(C_{\text{train}}\)</span> FLOPs 训练、每个查询给定 <span class="arithmatex">\(C_{\text{test}}\)</span> 推理 FLOPs 的模型准确率。经验上：</p>



<div class="arithmatex">\[\mathcal{A}(C_{\text{train}}, C_{\text{test}}) \approx 1 - \exp\!\left(-a \cdot C_{\text{train}}^{\alpha} \cdot C_{\text{test}}^{\beta}\right)
\tag{13.46}\]</div>

<p>其中 <span class="arithmatex">\(a, \alpha, \beta &gt; 0\)</span> 为常数。对于固定总预算 <span class="arithmatex">\(C_{\text{total}}\)</span>，最优分配满足训练与推理之间每 FLOP 的边际回报相等的条件：</p>



<div class="arithmatex">\[\frac{\partial \mathcal{A}}{\partial C_{\text{train}}} = \frac{1}{N} \cdot \frac{\partial \mathcal{A}}{\partial C_{\text{test}}}
\tag{13.47}\]</div>

<p>直觉上：一个 FLOP 的训练计算惠及全部 <span class="arithmatex">\(N\)</span> 个查询，而一个 FLOP 的测试时计算只惠及单个查询。在最优点，测试时计算的单查询边际价值是训练计算的 <span class="arithmatex">\(N\)</span> 倍（因为训练成本被摊销）。将此应用于公式 [eq:reasoning_scaling_law] 可得到最优训练计算占比：</p>



<div class="arithmatex">\[\frac{C_{\text{train}}^*}{C_{\text{total}}} = \frac{\alpha}{\alpha + \beta}
\tag{13.48}\]</div>

<p>对于特定预算结构 <span class="arithmatex">\(C_{\text{total}} = C_{\text{train}} + N \cdot C_{\text{test}}\)</span>，在乘性准确率模型下该占比与 <span class="arithmatex">\(N\)</span> 无关。然而在实际中 <span class="arithmatex">\(\alpha\)</span> 和 <span class="arithmatex">\(\beta\)</span> 是问题相关的：对于高流量部署（大 <span class="arithmatex">\(N\)</span>），即便基础模型的微小提升也会占主导，倾向于训练投入；对于低流量、高价值的查询（小 <span class="arithmatex">\(N\)</span>），测试时计算则更具成本效益。</p>

### 13.7.2　何时投资于更长推理链 vs. 更好的基础模型 {#when-to-invest-in-longer-chains-vs.-better-base-models}

<div class="admonition note"><p class="admonition-title">推理链长度与模型容量</p>
<p>对于容量为 <span class="arithmatex">\(C\)</span> 的模型在难度为 <span class="arithmatex">\(D\)</span> 的问题上，最优推理链长度 <span class="arithmatex">\(L^*\)</span> 满足：</p>



<div class="arithmatex">\[L^* \propto \frac{D}{C^{\gamma}}
\tag{13.49}\]</div>

<p>其中 <span class="arithmatex">\(\gamma &gt; 0\)</span>。这意味着：</p>

<ul>
<li><p><strong>困难问题</strong>无论模型规模如何都需要更长的推理链</p></li>
<li><p>在相同问题难度下，<strong>更大的模型</strong>需要更短的推理链</p></li>
<li><p><strong>收益递减</strong>：超过 <span class="arithmatex">\(L^*\)</span> 后，额外的 Token 不再带来收益，反而可能有害（过度思考）</p></li>
</ul>
</div>

<p>“过度思考（overthinking）”现象——拥有极长推理链的模型反而比拥有适中推理链的模型表现<em>更差</em>——已被经验观察到，归因于：</p>

<ul>
<li><p>长推理链中错误的累积（误差传播）</p></li>
<li><p>偏离主解路径的干扰</p></li>
<li><p>对错误中间结论的过度自信</p></li>
</ul>

### 13.7.3　最优 Token 预算分配 {#optimal-token-budget-allocation}

<p>对于给定 Token 预算 <span class="arithmatex">\(B\)</span> 的模型，“思考” Token <span class="arithmatex">\(T_{\text{think}}\)</span> 与“作答” Token <span class="arithmatex">\(T_{\text{answer}}\)</span> 之间的分配应满足：</p>



<div class="arithmatex">\[T_{\text{think}}^* = \arg\max_{T} \mathcal{A}(T, B - T)
\tag{13.50}\]</div>

<p>经验上，最优划分是问题相关的：</p>

<ul>
<li><p><strong>简单问题</strong>：<span class="arithmatex">\(T_{\text{think}}^* / B \approx 0.3\)</span>（30% 思考）</p></li>
<li><p><strong>困难问题</strong>：<span class="arithmatex">\(T_{\text{think}}^* / B \approx 0.8\)</span>（80% 思考）</p></li>
<li><p><strong>极困难问题</strong>：<span class="arithmatex">\(T_{\text{think}}^* / B \approx 0.95\)</span>（95% 思考，极简作答）</p></li>
</ul>

<p>这促成了<strong>自适应思考预算（adaptive thinking budgets）</strong>：为更难的问题分配更多 Token，难度可由模型在初次解题尝试中的不确定度来估计。</p>

## 13.8　推理模型比较 {#subsec:reasoning_comparison}

<p class="table-caption" id="table-13-3"><span class="table-number">表 13.3　</span>推理模型训练方法对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>PRM</strong></th><th scope="col"><strong>ORM</strong></th><th scope="col"><strong>MCTS</strong></th><th scope="col"><strong>蒸馏</strong></th><th scope="col"><strong>工具</strong></th><th scope="col"><strong>开源</strong></th></tr></thead>
<tbody><tr><td>OpenAI o1/o3</td><td>✓</td><td>✓</td><td>未知</td><td>–</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>DeepSeek-R1</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>QwQ / Qwen3</td><td>部分</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td>✓</td></tr>
<tr><td>AlphaProof</td><td>✓</td><td>✓</td><td>✓</td><td>–</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td></tr>
<tr><td>Math-Shepherd</td><td>✓</td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>–</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr>
<tr><td>STaR / Quiet-STaR</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td><td><span class="arithmatex">\(\times\)</span></td><td>–</td><td><span class="arithmatex">\(\times\)</span></td><td>✓</td></tr></tbody>
</table>
</div>

## 13.9　推理时算力扩展定律 {#sec:inference-time-scaling}

<p>训练时扩展定律（Chinchilla <a class="book-citation" id="cite-hoffmann2022chinchilla-1" href="../../references.md#ref-hoffmann2022chinchilla" aria-label="参考文献 140">[140]</a>）已被充分理解：模型性能随算力、数据与参数可预测地提升。<strong>推理时算力扩展</strong>则是新的前沿——这个实证发现表明，在<em>测试时</em>投入更多计算（更长的推理链、更多的搜索、迭代式精化）可以解决更大的 base 模型也解决不了的问题。</p>

<div class="admonition note"><p class="admonition-title">核心发现：把测试时算力当作一条扩展轴</p>
<p><strong>Snell 等人（2024）</strong> <a class="book-citation" id="cite-snell2024scaling-1--section-09" href="../../references.md#ref-snell2024scaling" aria-label="参考文献 335">[335]</a> 证明：对于困难的推理任务，测试时算力可以比扩展模型参数更有效：</p>

<ul>
<li><p>性能随推理 token 数<strong>对数线性</strong>提升：<span class="arithmatex">\(\text{acc} \approx a \cdot \log(\text{tokens}) + b\)</span>。</p></li>
<li><p>存在<strong>任务特定的最优 token 预算</strong>，超过之后收益急剧递减。</p></li>
<li><p>一个推理算力多 <span class="arithmatex">\(k\times\)</span> 的小模型可以匹敌训练参数多 <span class="arithmatex">\(k\times\)</span> 的大模型——而且对罕见的困难查询，服务成本更低。</p></li>
</ul>

<p>实证里程碑：<strong>o3</strong> 通过扩展推理时搜索，在 ARC-AGI（一个专门设计来抵抗模式匹配的基准）上取得 <strong>45.1%</strong>，远远超出仅靠模型规模所能达到的水平。</p>
</div>

### 13.9.1　过度思考问题 {#subsec:overthinking}

<p>推理时扩展引入了一个关键的失败模式：<strong>过度思考（overthinking）</strong>。被训练生成长思维链的推理模型即使在答案显而易见时也会继续生成 token，浪费算力，有时还会因引入虚假的推理步骤而降低准确率。</p>

<div class="admonition warning"><p class="admonition-title">过度思考：何时更多 token 反而有害</p>
<p>在简单问题上，推理模型的准确率随 token 预算呈现出一条典型的<strong>U 形曲线</strong>：</p>

<ol>
<li><p>短链：推理不足，准确率低。</p></li>
<li><p>最优链：推理正确，准确率峰值。</p></li>
<li><p>过长的链：模型对正确答案反复自我怀疑、引入错误，准确率<em>下降</em>。</p></li>
</ol>

<p>这意味着固定的“想得越久越好”策略是次优的。模型需要学会<strong>何时停止思考</strong>。</p>
</div>

### 13.9.2　高效推理技术 {#subsec:efficient-reasoning}

<p>有几种方法应对推理时扩展中的算力—质量权衡：</p>

<ul>
<li><p><strong>思维草图（Sketch-of-Thought）</strong>：在完整推理链之前先生成压缩的推理草图。通过在推理轨迹中把规划与执行分离，实现<strong>84% 的 token 削减</strong>，准确率损失极小。</p></li>
<li><p><strong>预算感知提示（budget-aware prompting）</strong>：显式地给模型下达 token 预算指令（“用 200 个 token 以内解决”）。在问题难度已知的基准上，推理成本减半。</p></li>
<li><p><strong>更短的链反而提升准确率</strong>：在某些任务上，强制更短的推理链能把准确率提升 <strong>34.5%</strong>——模型被迫更加直接，避开过度思考的螺旋。</p></li>
<li><p><strong>自适应深度</strong>：OpenAI 的 <code>reasoning_effort</code> 参数（low/medium/high）根据估计难度把查询路由到不同的 token 预算，为算力—质量权衡提供了实用 API。</p></li>
</ul>

### 13.9.3　与 RL 的联系：把推理算力视为 MDP 的时域 {#subsec:inference-mdp-connection}

<div class="admonition tip"><p class="admonition-title">MDP 框架中的推理时算力</p>
<p>推理时算力扩展在本书通篇使用的 RL 框架中有一个自然的解释：</p>

<ul>
<li><p>每个推理 token 都是<strong>MDP 中的一步</strong>。</p></li>
<li><p>分配更多推理算力 = 在相同奖励函数下<strong>更长的时域（horizon）</strong>。</p></li>
<li><p>过度思考问题 = 在代理指标（token 数）而非任务成功上的<strong>reward hacking</strong>。</p></li>
<li><p>高效推理 = 在 MDP 内学习<strong>最优停止策略</strong>。</p></li>
</ul>

<p>这一框架表明，推理时扩展定律并非仅仅是经验上的奇闻——它们反映了 RL 优化景观的根本性质。在推理任务上用 RL 训练的模型，隐式地同时在学习<em>想什么</em>与<em>想多久</em>。</p>
</div>

### 13.9.4　潜空间推理：思维链就是计算本身吗？ {#sec:latent-reasoning}

<p>越来越多的证据挑战“思维链<em>就是</em>推理本身”这一假设：</p>

<ul>
<li><p><strong>超越思维链的推理（Reasoning Beyond Chain-of-Thought）</strong> <a class="book-citation" id="cite-reasoning_latent2025-1" href="../../references.md#ref-reasoning_latent2025" aria-label="参考文献 168">[168]</a>：这项工作用稀疏自编码器（Sparse Autoencoder）探针检查内部表征，识别出在相应 token 生成<em>之前</em>就被激活的推理专用特征——暗示模型在写下思维链之前就已经“知道答案”。</p></li>
<li><p><strong>思考以回忆（Thinking to Recall）</strong> <a class="book-citation" id="cite-thinking_recall2026-1" href="../../references.md#ref-thinking_recall2026" aria-label="参考文献 115">[115]</a>（Google，COLM 2026）：证明生成推理轨迹能够解锁那些直接提示几乎无法触及的参数化知识——但轨迹本身可能只是<em>脚手架</em>，而非计算本身。</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">把思维链视为通信协议</p>
<p>新兴的观点是：思维链可能是模型内部推理回路与输出层之间的<em>通信协议</em>——一种“路由”经过包含答案的潜状态的方式——而非算法本身。这具有实际含义：(1) 验证推理质量不能只靠检查思维链，因为模型可能带着一条具有误导性或事后生成的轨迹而正确推理；(2) 未来的架构可能完全在潜空间中完成推理，只输出结论。</p>
</div>

## 13.10　面向推理的同策略自蒸馏 {#sec:on-policy-self-distillation}

<p>本章的方法——GRPO、过程奖励、MCTS——共享一个根本局限：<strong>稀疏反馈</strong>。无论序列多长，强化学习每回合只提供固定比特数的信息。一条 2000 token、最终得出错误答案的思维链只收到一个标量惩罚；模型学不到<em>哪些 token</em> 导致了失败。同策略自蒸馏（On-Policy Self-Distillation, OPSD） <a class="book-citation" id="cite-zhao2026opsd-1" href="../../references.md#ref-zhao2026opsd" aria-label="参考文献 430">[430]</a> 通过提供<strong>密集的 token 级监督</strong>、同时保留让 RL 有效的 on-policy 性质来解决这一问题。</p>

<div class="admonition note"><p class="admonition-title">同策略自蒸馏：核心思想</p>
<p>同一个 LLM 同时扮演两个角色：</p>

<ul>
<li><p><strong>教师（Teacher）</strong>：同一个模型，但以<em>特权信息</em>为条件（例如把已验证的解答或推理轨迹附加到提示中）。</p></li>
<li><p><strong>学生（Student）</strong>：模型只以原始问题为输入，生成自己的 rollout。</p></li>
</ul>

<p>训练最小化教师与学生分布之间的逐 token 散度，并在<em>学生自己的轨迹</em>上评估。不需要额外、更大的教师模型。</p>
</div>

### 13.10.1　训练目标 {#subsec:opsd-objective}

<p>令 <span class="arithmatex">\(\pi_\theta\)</span> 表示学生策略（只给定问题的模型），<span class="arithmatex">\(\pi_\theta^+\)</span> 表示教师策略（以特权上下文如已验证答案为条件的同一个模型）。对于从学生中采样的轨迹 <span class="arithmatex">\(x = (x_1, \ldots, x_T)\)</span>，OPSD 损失最小化逐 token 的反向 KL 散度：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_{x \sim \pi_\theta}\left[\sum_{t=1}^{T} \text{KL}\!\left(\pi_\theta(\cdot \mid x_{&lt;t}) \;\|\; \pi_\theta^+(\cdot \mid x_{&lt;t})\right)\right]\]</div>

<p>在实践中，这简化为在采样 token 上评估的对数概率之差：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{OPSD}}(\theta) \approx \mathbb{E}_{x \sim \pi_\theta}\left[\sum_{t=1}^{T} \log \pi_\theta(x_t \mid x_{&lt;t}) - \log \pi_\theta^+(x_t \mid x_{&lt;t})\right]\]</div>

<p>教师的对数概率只需一次前向传播（无梯度计算），使得该方法比运行独立的奖励模型或执行带延迟奖励的完整 RL rollout 便宜得多。</p>

### 13.10.2　为什么密集监督很重要 {#subsec:opsd-dense}

<p>信息论上的论证很有说服力：RL 每回合只教 <span class="arithmatex">\(O(1)\)</span> 比特（标量奖励），而 token 级蒸馏每回合教 <span class="arithmatex">\(O(T)\)</span> 比特。这直接转化为训练效率：</p>

<ul>
<li><p><strong>Qwen3 结果</strong>：Qwen3 技术报告称，同策略蒸馏以 RL 流水线<strong>十分之一的 GPU 时长</strong>（1,800 vs. 17,920 GPU 小时）在 AIME’24 上达到 74.4%，同时超过了 RL 的 67.6%。</p></li>
<li><p><strong>步数效率</strong>：实证对比显示，OPSD 用比 GRPO 少 7–10<span class="arithmatex">\(\times\)</span> 的梯度步数就能达到同等的推理性能；考虑到所需上下文长度更短、批大小更小，对应 50–100<span class="arithmatex">\(\times\)</span> 的算力节省。</p></li>
<li><p><strong>数据复用</strong>：与在重复提示上训练时会记住最终答案的 RL 不同，OPSD 通过反向 KL 学习逼近教师的完整分布，从而可以在小提示集上进行有效的多轮训练。</p></li>
</ul>

### 13.10.3　失败模式与缓解措施 {#subsec:opsd-failures}

<div class="admonition warning"><p class="admonition-title">特权导致的风格漂移</p>
<p>当教师以已验证解答为条件时，它倾向于产出更短、更直接的输出——因为它已经“知道”答案。由此产生的师生分布差距集中在<em>风格 token</em>（简洁、直接）上，而非<em>承载任务的 token</em>（正确的推理步骤）。这导致学生的回答变短，准确率却没有提升。</p>
</div>

<p>RLCSD <a class="book-citation" id="cite-pan2026rlcsd-1" href="../../references.md#ref-pan2026rlcsd" aria-label="参考文献 282">[282]</a> 通过<strong>对比式</strong>公式化解决这一病态：它在正确提示与错误提示两种条件下分别计算师生差距，然后相减，以消除任何提示都会诱导出的风格分量。残差信号集中在任务相关 token 上。</p>

<p>EGRSD <a class="book-citation" id="cite-ke2026egrsd-1" href="../../references.md#ref-ke2026egrsd" aria-label="参考文献 181">[181]</a> 采用互补的方法：它引入一个<strong>熵引导的置信门控</strong>，对教师预测分布高熵（不确定）的 token 位置降低权重，把梯度集中在教师自信而学生偏离的位置。</p>

### 13.10.4　与 RL 的联系及训练流水线 {#subsec:opsd-pipeline}

<p>OPSD 在后训练流水线中占据一个特定的生态位：</p>

<figure id="fig:opsd-pipeline"><a href="../../assets/diagrams/figure-13-6.svg" target="_blank" rel="noopener" aria-label="查看图 13.6 原图" title="点击查看原图"><img src="../../assets/diagrams/figure-13-6.svg" alt="OPSD 连接了 SFT 与 RL：它把同策略采样（避免分布失配）与密集的 token 级监督（避免奖励稀疏）结合起来。" loading="lazy"></a><figcaption><span class="figure-number">图 13.6　</span>OPSD 连接了 SFT 与 RL：它把同策略采样（避免分布失配）与密集的 token 级监督（避免奖励稀疏）结合起来。</figcaption></figure>

<p>前沿实验室中浮现的实用流水线是：(1) 在教师生成的推理轨迹上做 SFT，把策略初始化到正确的分布中；(2) 用 OPSD 以密集监督低成本地恢复或精化推理能力；(3) 在 OPSD 的教师信号变得不可靠（教师自己也解不出）的难题上，用 RL 做最后的冲刺。</p>

<div class="admonition tip"><p class="admonition-title">何时用 OPSD，何时用 RL</p>
<ul>
<li><p><strong>用 OPSD</strong>：当教师（或带特权上下文的模型自身）能可靠解决目标任务时；当算力预算有限时；当需要在特定领域微调后恢复推理行为时；以及当 RL 的稀疏性会导致灾难性遗忘的持续学习场景。</p></li>
<li><p><strong>用 RL</strong>：当需要突破教师的能力天花板时；当要探索任何教师分布中都不存在的新颖推理策略时；当可验证奖励很廉价时（代码执行、数学验证）。</p></li>
<li><p><strong>两者都用</strong>：先用 OPSD 低成本地达到教师水平，再用 RL 突破它。</p></li>
</ul>
</div>

## 13.11　总结与开放问题 {#subsec:reasoning_summary}

<p>RL 用于推理模型的领域进展极为迅速。已经浮现出若干关键经验：</p>

<ol>
<li><p><strong>可验证奖励已经足够</strong>：对于具备真值验证的领域（数学、代码），仅靠结果奖励就足以让 RL 发现复杂的推理策略，无需过程奖励模型。</p></li>
<li><p><strong>测试时计算是一个新维度</strong>：推理模型引入了新的扩展维度——推理计算——在困难推理任务上它与训练计算大致可互相替代。</p></li>
<li><p><strong>蒸馏极为有效</strong>：大型推理模型可以通过对生成推理链进行监督微调，将能力迁移到小得多的模型上，往往优于对小模型直接进行 RL 训练。同策略自蒸馏进一步以一小部分算力缩小了蒸馏与 RL 之间的差距。</p></li>
<li><p><strong>涌现的元认知</strong>：在推理任务上的 RL 训练会涌现出未被显式训练的自我纠正与验证行为。</p></li>
<li><p><strong>密集监督胜过稀疏奖励</strong>：同策略自蒸馏通过提供 token 级反馈而非回合级奖励，以低 10–100<span class="arithmatex">\(\times\)</span> 的算力达到 RL 级推理。</p></li>
</ol>

<div class="admonition question"><p class="admonition-title">RL 用于推理中的开放问题</p>
<p>若干根本性问题仍待回答：</p>

<ul>
<li><p><strong>泛化</strong>：在数学/代码上训练得到的推理能力，能否迁移到其他领域（科学推理、规划、社会推理）？</p></li>
<li><p><strong>忠实性（Faithfulness）</strong>：生成的推理链对最终答案是因果性负责的，还是事后合理化？潜空间推理研究表明，思维链可能是一种通信协议，而非计算本身。</p></li>
<li><p><strong>最优搜索</strong>：推理时的最优搜索策略是什么——束搜索（beam search）、MCTS，还是其他？</p></li>
<li><p><strong>奖励设计</strong>：对于没有真值验证器的领域，如何为推理设计可靠的奖励信号？</p></li>
<li><p><strong>过度思考</strong>：模型如何学会分配<em>恰当</em>的思考量——既不过少也不过多？推理时扩展定律显示，超过任务特定的最优预算后，收益递减并最终变为负。</p></li>
<li><p><strong>组合式推理</strong>：经过 RL 训练的推理模型能否解决需要组合多种不同推理技能的问题？</p></li>
<li><p><strong>潜空间 vs. 显式推理</strong>：未来的架构能否完全在连续潜空间中推理、只输出结论——如果可以，我们又该如何验证它们的推理？</p></li>
<li><p><strong>自蒸馏的天花板</strong>：OPSD 受限于教师的能力（以特权上下文为条件的模型）。我们如何在不必退回昂贵的 RL 探索的情况下突破这一天花板？</p></li>
</ul>
</div>

<p>推理模型的发展代表了一次范式转变：从<em>知道</em>事情的语言模型，转向能够<em>推演</em>事情的语言模型。本章所描述的方法——从可验证奖励的 RL，到推理时算力扩展，再到同策略自蒸馏——构成了一套快速扩张的工具箱，其持续发展很可能成为未来数年 AI 研究的中心焦点。</p>
