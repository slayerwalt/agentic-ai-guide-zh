# 13.2　测试时扩展方法 {#sec:test_time_methods}

<p>上述扩展律表明，在推理阶段投入更多算力可以显著提升推理性能。本节系统介绍将测试时扩展（Test-Time Scaling）落地的各类<strong>方法</strong>——从简单的思维链到复杂的树与图搜索算法。这些方法构成了一个用推理成本换取准确率的连续谱，理解其结构对于设计现代推理系统至关重要。</p>

<figure id="fig:test_time_spectrum"><a href="../../assets/figures/fig_045_test_time_spectrum.png" target="_blank" rel="noopener" aria-label="查看图 13.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_045_test_time_spectrum.png" alt="测试时扩展方法的连续谱。每种方法都以额外的推理算力换取更高的推理准确率。各方法在概念上层层递进：CoT 引入显式推理，Self-Consistency 加入采样，ToT 引入结构化搜索，GoT 加入合并操作，而 MCTS 加入学习到的价值引导。" loading="lazy"></a><figcaption><span class="figure-number">图 13.2　</span>测试时扩展方法的连续谱。每种方法都以额外的推理算力换取更高的推理准确率。各方法在概念上层层递进：CoT 引入显式推理，Self-Consistency 加入采样，ToT 引入结构化搜索，GoT 加入合并操作，而 MCTS 加入学习到的价值引导。</figcaption></figure>

## 13.2.1　思维链（CoT） {#subsec:cot_method}

<p>思维链 Prompt <a class="book-citation" id="cite-wei2022chain-1" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a> 是所有测试时扩展方法的基础。模型不再直接输出答案，而是生成中间推理步骤，将复杂问题分解为可处理的子问题。</p>

#### 零样本 CoT。 {#zero-shot-cot.}

<p>Kojima 等 <a class="book-citation" id="cite-kojima2022large-1" href="../../references.md#ref-kojima2022large" aria-label="参考文献 189">[189]</a> 证明，仅在 Prompt 末尾追加“Let’s think step by step”就能在没有任何示例的情况下激发推理行为。这一简单触发器可以激活足够大模型（<span class="arithmatex">\(\geq\)</span> 100B 参数）中的潜在推理能力。</p>

#### 少样本 CoT。 {#few-shot-cot.}

<p>Wei 等 <a class="book-citation" id="cite-wei2022chain-2" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a> 表明，提供少量带有显式推理轨迹的示例能够让较小的模型也有效地进行推理：</p>



<div class="arithmatex">\[\text{Prompt} = [(x_1, z_1, y_1), (x_2, z_2, y_2), \ldots, (x_k, z_k, y_k), (x_{\text{test}}, \texttt{?})]
\tag{13.2}\]</div>

<p>其中 <span class="arithmatex">\(z_i\)</span> 是为示例 <span class="arithmatex">\((x_i, y_i)\)</span> 手工编写的推理轨迹。</p>

#### 形式化刻画。 {#formal-characterization.}

<p>CoT 将单步预测 <span class="arithmatex">\(p(y|x)\)</span> 转化为多步序列生成：</p>



<div class="arithmatex">\[p(y|x) = \sum_{z} p(y|x, z) \cdot p(z|x) \approx p(y|x, z^*) \cdot p(z^*|x)
\tag{13.3}\]</div>

<p>其中 <span class="arithmatex">\(z^* = (z_1, z_2, \ldots, z_T)\)</span> 是贪心推理链。对所有可能链条求和是不可计算的；标准 CoT 仅采用单一样本（贪心或温度采样）。</p>

#### 局限性。 {#limitations.-dup2}

<p>单链 CoT 是脆弱的：一旦早期推理步骤出错，后续所有步骤都会建立在错误基础上，且没有任何恢复机制。</p>

## 13.2.2　自洽性（多数投票） {#self-consistency-majority-voting}

<p>自洽性（Self-Consistency） <a class="book-citation" id="cite-wang2023selfconsistency-1" href="../../references.md#ref-wang2023selfconsistency" aria-label="参考文献 367">[367]</a> 通过采样<strong>多条独立的推理链</strong>并对最终答案取多数投票，来缓解 CoT 单链脆弱的问题：</p>



<div class="arithmatex">\[\hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbf{1}[y_i = y], \quad \text{where } (z_i, y_i) \sim p(\cdot | x), \; T &gt; 0
\tag{13.4}\]</div>

<p><strong>关键性质</strong>：</p>

<ul>
<li><p>使用温度 <span class="arithmatex">\(T &gt; 0\)</span> 的采样以产生多样的链条（通常 <span class="arithmatex">\(T = 0.7\)</span>–<span class="arithmatex">\(1.0\)</span>）</p></li>
<li><p>链条之间无交互——完全可并行化</p></li>
<li><p>准确率随 <span class="arithmatex">\(N\)</span> 单调提升（<span class="arithmatex">\(N \approx 40\)</span> 之后收益递减）</p></li>
<li><p>在 GSM8K 上：CoT = 56.5%，Self-Consistency（<span class="arithmatex">\(N\)</span>=40）= 74.4%（使用 PaLM-540B <a class="book-citation" id="cite-chowdhery2022palm-1" href="../../references.md#ref-chowdhery2022palm" aria-label="参考文献 55">[55]</a>）</p></li>
<li><p>等价于<strong>使用结果奖励的 Best-of-N</strong>（多数投票充当隐式的 ORM）</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">多数投票为何有效</p>
<p>若模型生成正确推理链的概率为 <span class="arithmatex">\(p &gt; 0.5\)</span>，则由大数定律可知：当 <span class="arithmatex">\(N \to \infty\)</span> 时，对 <span class="arithmatex">\(N\)</span> 个独立样本做多数投票的准确率趋近 100%。即便 <span class="arithmatex">\(p = 0.3\)</span>（模型通常是错的），只要正确答案集中在某一个值上、而错误答案彼此分散，多数投票仍能恢复出正确答案。这是测试时扩展的统计基础。</p>
</div>

## 13.2.3　思维树（ToT） {#subsec:tot_method}

<p>思维树（Tree-of-Thoughts，ToT） <a class="book-citation" id="cite-yao2024tree-1" href="../../references.md#ref-yao2024tree" aria-label="参考文献 409">[409]</a> 将 CoT 从<strong>线性链</strong>推广为<strong>树形结构</strong>，使模型能够探索多条推理路径、评估中间状态，并从前景不佳的分支回溯。这为推理过程引入了刻意的规划。</p>

#### 核心抽象。 {#core-abstraction.}

<p>推理问题被分解为对一棵树的搜索，其中：</p>

<ul>
<li><p><strong>根节点</strong>：初始问题陈述 <span class="arithmatex">\(x\)</span></p></li>
<li><p><strong>节点</strong>：部分推理状态 <span class="arithmatex">\(s = (x, z_1, \ldots, z_k)\)</span></p></li>
<li><p><strong>边</strong>：单个推理步骤（“想法”）<span class="arithmatex">\(z_{k+1}\)</span></p></li>
<li><p><strong>叶子</strong>：包含最终答案的完整解答</p></li>
<li><p><strong>价值函数</strong>：<span class="arithmatex">\(V(s)\)</span> 估计某个部分解的前景如何</p></li>
</ul>

#### 形式化定义。 {#formal-definition.}



<div class="arithmatex">\[\text{ToT} = (\mathcal{G}, \mathcal{E}, V, \pi_\theta, \text{Search})
\tag{13.5}\]</div>

<p>其中：</p>

<ul>
<li><p><span class="arithmatex">\(\mathcal{G}\)</span>：<strong>想法生成器</strong>——产生 <span class="arithmatex">\(b\)</span> 个候选的下一想法：<span class="arithmatex">\(\{z^{(1)}, \ldots, z^{(b)}\} \sim \pi_\theta(\cdot | s)\)</span></p></li>
<li><p><span class="arithmatex">\(\mathcal{E}\)</span>：<strong>状态评估器</strong>——为部分解打分：<span class="arithmatex">\(V(s) \in \{\)</span><em>sure</em>, <em>maybe</em>, <em>impossible</em><span class="arithmatex">\(\}\)</span> 或 <span class="arithmatex">\(V(s) \in [0, 1]\)</span></p></li>
<li><p><span class="arithmatex">\(\pi_\theta\)</span>：生成想法的语言模型</p></li>
<li><p><span class="arithmatex">\(\text{Search}\)</span>：搜索算法（BFS 或 DFS）</p></li>
</ul>

<figure id="figure-13-3"><a href="../../assets/figures/fig_046_tot_example.png" target="_blank" rel="noopener" aria-label="查看图 13.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_046_tot_example.png" alt="思维树在“24 点”任务上的运行：对 4, 9, 10, 13 进行四则运算得到 24。在每一层，模型生成 \(b=3\) 个候选想法，分别评估（sure/maybe/impossible），剪掉前景不佳的分支，并扩展最有希望的分支。绿色路径通向解答；红色路径在早期被剪枝。" loading="lazy"></a><figcaption><span class="figure-number">图 13.3　</span>思维树在“24 点”任务上的运行：对 4, 9, 10, 13 进行四则运算得到 24。在每一层，模型生成 <span class="arithmatex">\(b=3\)</span> 个候选想法，分别评估（sure/maybe/impossible），剪掉前景不佳的分支，并扩展最有希望的分支。绿色路径通向解答；红色路径在早期被剪枝。</figcaption></figure>

#### 搜索算法。 {#search-algorithms.}

<p><strong>BFS（广度优先搜索）：</strong></p>

<ol>
<li><p>为当前深度的每个节点生成 <span class="arithmatex">\(b\)</span> 个候选想法</p></li>
<li><p>用 <span class="arithmatex">\(V(\cdot)\)</span> 评估所有候选</p></li>
<li><p>保留前 <span class="arithmatex">\(k\)</span> 个最有希望的状态（束搜索）</p></li>
<li><p>将这 <span class="arithmatex">\(k\)</span> 个状态全部推进到下一层</p></li>
<li><p>重复，直到找到解答或达到深度上限</p></li>
</ol>

<p><strong>DFS（深度优先搜索）：</strong></p>

<ol>
<li><p>为当前状态生成 <span class="arithmatex">\(b\)</span> 个候选想法</p></li>
<li><p>评估：若 <span class="arithmatex">\(V(s) =\)</span> <em>impossible</em>，立即回溯</p></li>
<li><p>若 <span class="arithmatex">\(V(s) =\)</span> <em>sure/maybe</em>，向更深处递归（选择最有希望的一个）</p></li>
<li><p>若达到深度上限仍未得到解答，则回溯</p></li>
<li><p>继续直到找到解答或所有分支都被探索完</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">ToT：价值评估 Prompt</p>
<div class="highlight"><pre><span></span><code><span class="c1"># LLM 用于评估部分推理状态：</span>
<span class="n">EVAL_PROMPT</span> <span class="o">=</span> <span class="s2">"""Evaluate if this partial solution can reach 24.</span>

<span class="s2">Numbers remaining: [4, 4, 10]</span>
<span class="s2">Steps so far: 13 - 9 = 4</span>

<span class="s2">Can these remaining numbers (4, 4, 10) be combined using +,-,*,/</span>
<span class="s2">to make 24?</span>

<span class="s2">Analysis: 4 * (10 - 4) = 4 * 6 = 24. Yes!</span>

<span class="s2">Judge: sure/maybe/impossible</span>
<span class="s2">Answer: sure"""</span>

<span class="c1"># 想法生成 Prompt：</span>
<span class="n">GEN_PROMPT</span> <span class="o">=</span> <span class="s2">"""Input: 4 9 10 13</span>
<span class="s2">Possible next steps:</span>
<span class="s2">1. 13 - 9 = 4 (left: 4 4 10)</span>
<span class="s2">2. 10 + 13 = 23 (left: 4 9 23)</span>
<span class="s2">3. 9 - 4 = 5 (left: 5 10 13)</span>
<span class="s2">..."""</span>
</code></pre></div>
</div>

#### 计算成本。 {#computational-cost.}

<p>对于分支因子为 <span class="arithmatex">\(b\)</span>、深度为 <span class="arithmatex">\(d\)</span>、束宽为 <span class="arithmatex">\(k\)</span> 的 ToT：</p>



<div class="arithmatex">\[\text{LLM calls (BFS)} = \underbrace{k \cdot b}_{\text{generation}} + \underbrace{k \cdot b}_{\text{evaluation}} = 2kb \text{ per level} \implies \text{Total} = 2kbd
\tag{13.6}\]</div>

<p>对于 24 点游戏：<span class="arithmatex">\(b=3, k=2, d=3 \implies 36\)</span> 次 LLM 调用，而标准 CoT 仅需 1 次。</p>

#### 结果。 {#results.-dup2}

<p>在 24 点游戏（一个具挑战性的算术推理任务）上，ToT 达到 74% 的成功率，而 CoT 仅 4%——在相同基础模型（GPT-4）之上，结构化搜索带来了巨大的提升。</p>

## 13.2.4　思维图（GoT） {#subsec:got_method}

<p>思维图（Graph-of-Thoughts，GoT） <a class="book-citation" id="cite-besta2024graph-1" href="../../references.md#ref-besta2024graph" aria-label="参考文献 25">[25]</a> 将 ToT 从树推广到<strong>有向无环图（DAG）</strong>，引入了一项关键能力：<strong>合并</strong>来自不同分支的部分解。这让模型可以把多条推理路径中的洞见综合到一个精炼的解答中。</p>

#### 关键操作。 {#key-operations.}

<p>GoT 在 ToT 之上引入了三种操作：</p>

<ul>
<li><p><strong>Generate</strong>：从一个状态产生新的想法（与 ToT 相同）</p></li>
<li><p><strong>Aggregate/Merge</strong>：把多个想法合并为一个更精炼的想法——这是树结构中不可能做到的</p></li>
<li><p><strong>Refine</strong>：基于反馈对一个想法进行迭代改进</p></li>
<li><p><strong>Score</strong>：评估想法的质量（与 ToT 的价值函数相同）</p></li>
</ul>

<figure id="fig:got_comparison"><a href="../../assets/figures/fig_047_got_comparison.png" target="_blank" rel="noopener" aria-label="查看图 13.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_047_got_comparison.png" alt="CoT（线性链）、ToT（树——只能分支不能合并）与 GoT（DAG——分支可以合并）的对比。对于一个排序任务，GoT 可以把数组拆为子问题、独立（并行）求解，然后合并结果——在纯树结构中不可能做到。这使得分治式推理成为可能。" loading="lazy"></a><figcaption><span class="figure-number">图 13.4　</span>CoT（线性链）、ToT（树——只能分支不能合并）与 GoT（DAG——分支可以合并）的对比。对于一个排序任务，GoT 可以把数组拆为子问题、独立（并行）求解，然后<strong>合并</strong>结果——在纯树结构中不可能做到。这使得分治式推理成为可能。</figcaption></figure>

#### 图操作（形式化）。 {#graph-operations-formal.}

<p>设 <span class="arithmatex">\(\mathcal{V} = \{v_1, \ldots, v_n\}\)</span> 为想法节点，<span class="arithmatex">\(\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}\)</span> 为有向边。GoT 支持：</p>



<div class="arithmatex">\[\begin{align*}
\textbf{Generate}(v) &amp;: v \to \{v_{c_1}, \ldots, v_{c_b}\} &amp;&amp; \text{(创建子节点)} \tag{13.7}\\

\textbf{Aggregate}(v_1, \ldots, v_k) &amp;\to v_{\text{merged}} &amp;&amp; \text{(将 $k$ 个想法合并为一个)} \tag{13.8}\\

\textbf{Refine}(v, n) &amp;\to v&#x27; &amp;&amp; \text{(通过 $n$ 次迭代改进 $v$)} \tag{13.9}\\

\textbf{Score}(v) &amp;\to s \in [0, 1] &amp;&amp; \text{(评估想法质量)}
\tag{13.10}
\end{align*}\]</div>

<p><strong>Aggregate</strong> 操作是关键差异：它从多个父节点向单个子节点连边，从而形成 DAG 而非树。这使得以下能力成为可能：</p>

<ul>
<li><p><strong>分治</strong>：拆分问题 <span class="arithmatex">\(\to\)</span> 并行求解子问题 <span class="arithmatex">\(\to\)</span> 合并解答</p></li>
<li><p><strong>集成推理</strong>：生成多个视角，然后综合出最佳想法</p></li>
<li><p><strong>迭代精炼</strong>：把评估结果反馈回去以改进早期想法</p></li>
</ul>

#### 结果。 {#results.-1}

<p>在排序任务（一项需要合并的任务）上，GoT 在质量相当时相比 ToT 节省 62% 成本。在集合求交与关键词计数任务上，由于合并操作支持更高效的分解，GoT 在相同质量下减少了 30–40% 的 LLM 调用。</p>

## 13.2.5　结合奖励模型的 Best-of-N {#subsec:bon_method}

<p>Best-of-N 采样（Rejection Sampling）（BoN） <a class="book-citation" id="cite-nakano2021webgpt-1" href="../../references.md#ref-nakano2021webgpt" aria-label="参考文献 262">[262]</a><a class="book-citation" id="cite-stiennon2020learning-1" href="../../references.md#ref-stiennon2020learning" aria-label="参考文献 339">[339]</a> 是最简单的扩展方法，它利用一个<strong>学习得到的奖励模型</strong>在候选中进行选择：</p>



<div class="arithmatex">\[y^* = \arg\max_{y \in \{y_1, \ldots, y_N\}} R_\phi(x, y), \quad y_i \sim \pi_\theta(\cdot | x)
\tag{13.11}\]</div>

<p><strong>按奖励模型类型的变体</strong>：</p>

<ul>
<li><p><strong>结合 ORM 的 BoN</strong>：对完整解答打分，选择得分最高者。当 ORM <span class="arithmatex">\(\approx\)</span> 正确性检查时等价于 Self-Consistency。</p></li>
<li><p><strong>结合 PRM 的 BoN</strong>：对每一推理步打分，选择最小步得分最高的解答（即在任何步骤上出错的可能性最小）。</p></li>
<li><p><strong>加权 BoN</strong>：按奖励对候选加权：<span class="arithmatex">\(y^* \sim \text{softmax}(R(y_1)/\tau, \ldots, R(y_N)/\tau)\)</span>。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">BoN 扩展律</p>
<p>对于单样本准确率为 <span class="arithmatex">\(p\)</span> 的模型，<span class="arithmatex">\(N\)</span> 次采样中至少有一个正确样本的概率为：</p>



<div class="arithmatex">\[P(\text{success with BoN}) = 1 - (1-p)^N
\tag{13.12}\]</div>

<p>在完美奖励模型（总能正确选择的预言机）下：</p>

<ul>
<li><p><span class="arithmatex">\(p = 0.3, N = 10\)</span>：成功率 = <span class="arithmatex">\(97\%\)</span></p></li>
<li><p><span class="arithmatex">\(p = 0.1, N = 50\)</span>：成功率 = <span class="arithmatex">\(99.5\%\)</span></p></li>
</ul>

<p>在实践中，不完美的奖励模型会限制有效的 <span class="arithmatex">\(N\)</span>——超过 <span class="arithmatex">\(N \approx 64\)</span>–<span class="arithmatex">\(256\)</span> 后，奖励模型的误差开始主导，准确率趋于停滞甚至下降（<strong>奖励黑客</strong>）。</p>
</div>

## 13.2.6　面向推理的蒙特卡洛树搜索（MCTS） {#monte-carlo-tree-search-mcts-for-reasoning}

<p>蒙特卡洛树搜索（MCTS） <a class="book-citation" id="cite-kocsis2006bandit-1" href="../../references.md#ref-kocsis2006bandit" aria-label="参考文献 187">[187]</a><a class="book-citation" id="cite-silver2016mastering-1" href="../../references.md#ref-silver2016mastering" aria-label="参考文献 331">[331]</a> 将 ToT 的结构化探索与<strong>学习到的价值估计</strong>及<strong>访问计数统计</strong>结合起来，从而最优地分配推理算力。MCTS 最初为博弈而设计（AlphaGo <a class="book-citation" id="cite-silver2016mastering-2" href="../../references.md#ref-silver2016mastering" aria-label="参考文献 331">[331]</a>），后被 AlphaProof <a class="book-citation" id="cite-alphaproof2024-1" href="../../references.md#ref-alphaproof2024" aria-label="参考文献 68">[68]</a>、rStar <a class="book-citation" id="cite-qi2024mutual-1" href="../../references.md#ref-qi2024mutual" aria-label="参考文献 297">[297]</a> 等系统改造用于 LLM 推理。</p>

#### 算法（针对 LLM 推理改造）。 {#algorithm-adapted-for-llm-reasoning.}

<p>每次 MCTS 迭代包含四个阶段：</p>

<figure id="fig:mcts_phases"><a href="../../assets/figures/fig_048_mcts_phases.png" target="_blank" rel="noopener" aria-label="查看图 13.5 原图" title="点击查看原图"><img src="../../assets/figures/fig_048_mcts_phases.png" alt="面向推理的 MCTS 的四个阶段：(1) 选择：用 UCB 遍历树以找到一个有前途的叶子节点；(2) 扩展：从该叶子节点生成新的推理步骤；(3) 模拟：将推理补全到终止状态并进行评估；(4) 反向传播：沿路径更新价值估计。" loading="lazy"></a><figcaption><span class="figure-number">图 13.5　</span>面向推理的 MCTS 的四个阶段：(1) <strong>选择</strong>：用 UCB 遍历树以找到一个有前途的叶子节点；(2) <strong>扩展</strong>：从该叶子节点生成新的推理步骤；(3) <strong>模拟</strong>：将推理补全到终止状态并进行评估；(4) <strong>反向传播</strong>：沿路径更新价值估计。</figcaption></figure>

#### 推理中的 UCB。 {#ucb-for-reasoning.}

<p>节点选择使用 PUCT（Predictor + UCB applied to Trees）：</p>



<div class="arithmatex">\[a^* = \arg\max_a \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s, a)} \right]
\tag{13.13}\]</div>

<p>其中 <span class="arithmatex">\(P(s,a) = \pi_\theta(a|s)\)</span> 是 LLM 从状态 <span class="arithmatex">\(s\)</span> 生成步骤 <span class="arithmatex">\(a\)</span> 的先验概率。这使探索偏向 LLM 本就认为可能的步骤，而 UCB 项鼓励尝试探索不足的备选方案。</p>

<div class="admonition example"><p class="admonition-title">面向数学推理的 MCTS：运行示例</p>
<p><strong>问题</strong>：证明 <span class="arithmatex">\(\sqrt{2}\)</span> 是无理数。</p>

<p><strong>第 1 次迭代</strong>（Selection <span class="arithmatex">\(\to\)</span> 根节点，Expansion）：</p>

<ul>
<li><p>生成 3 个候选的首步：</p>

<ol>
<li><p>“Assume for contradiction that <span class="arithmatex">\(\sqrt{2} = p/q\)</span> in lowest terms.”（<span class="arithmatex">\(P = 0.7\)</span>）</p></li>
<li><p>“Consider the decimal expansion of <span class="arithmatex">\(\sqrt{2}\)</span> = 1.414...”（<span class="arithmatex">\(P = 0.15\)</span>）</p></li>
<li><p>“Use the fundamental theorem of arithmetic.”（<span class="arithmatex">\(P = 0.10\)</span>）</p></li>
</ol></li>
<li><p>从 <span class="arithmatex">\(z_1\)</span> 做 Rollout：4 步内得到正确证明 <span class="arithmatex">\(\to\)</span> <span class="arithmatex">\(r = 1.0\)</span></p></li>
<li><p>从 <span class="arithmatex">\(z_2\)</span> 做 Rollout：失败（小数展开不能证明无理性）<span class="arithmatex">\(\to\)</span> <span class="arithmatex">\(r = 0.0\)</span></p></li>
<li><p>反向传播：<span class="arithmatex">\(Q(s_0, z_1) = 1.0\)</span>，<span class="arithmatex">\(N(s_0, z_1) = 1\)</span></p></li>
</ul>

<p><strong>第 2 次迭代</strong>（Selection：按 UCB 选中 <span class="arithmatex">\(z_1\)</span>）：</p>

<ul>
<li><p>从状态“Assume <span class="arithmatex">\(\sqrt{2} = p/q\)</span>...”进行扩展：</p>

<ol>
<li><p>“Then <span class="arithmatex">\(2 = p^2/q^2\)</span>, so <span class="arithmatex">\(p^2 = 2q^2\)</span>.”（<span class="arithmatex">\(P = 0.8\)</span>）</p></li>
<li><p>“Then <span class="arithmatex">\(p\)</span> and <span class="arithmatex">\(q\)</span> share no common factors.”（<span class="arithmatex">\(P = 0.15\)</span>）</p></li>
</ol></li>
<li><p>从 <span class="arithmatex">\(z_4\)</span> 做 Rollout：正确续推 <span class="arithmatex">\(\to r = 1.0\)</span></p></li>
<li><p>反向传播：<span class="arithmatex">\(Q(s_0, z_1) = 1.0\)</span>，<span class="arithmatex">\(Q(s_1, z_4) = 1.0\)</span></p></li>
</ul>

<p><strong>20 次迭代之后</strong>：树已探索 8 条不同的推理路径。访问次数最多的路径被选为最终证明：<span class="arithmatex">\(z_1 \to z_4 \to z_6 \to z_8\)</span>（基于奇偶性论证的经典反证法）。</p>
</div>

#### 对比：ToT vs. MCTS。 {#comparison-tot-vs.-mcts.}

<p class="table-caption" id="table-13-1"><span class="table-number">表 13.1　</span>面向推理的 Tree-of-Thoughts 与 Monte Carlo Tree Search 对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>ToT</strong></th><th scope="col"><strong>MCTS</strong></th></tr></thead>
<tbody><tr><td>价值估计</td><td>LLM Prompt（“sure/maybe/impossible”）</td><td>学习得到的价值网络 + Rollout 统计</td></tr>
<tr><td>探索方式</td><td>固定束宽；不回访</td><td>UCB 自适应地把预算分配给有前途的节点</td></tr>
<tr><td>算力分配</td><td>各深度层均匀</td><td>聚焦：在更难的子问题上做更多模拟</td></tr>
<tr><td>训练集成</td><td>无训练；纯 Prompt</td><td>可将 MCTS 策略蒸馏回基础模型 <a class="book-citation" id="cite-silver2016mastering-3" href="../../references.md#ref-silver2016mastering" aria-label="参考文献 331">[331]</a></td></tr>
<tr><td>最适用于</td><td>分支简单的问题（24 点）</td><td>需要深度探索的复杂问题（证明、代码）</td></tr></tbody>
</table>
</div>

## 13.2.7　推理步级的束搜索 {#subsec:beam_reasoning}

<p>束搜索——在 NMT 与文本生成中长期作为标准方法——也可以在<em>推理步</em>层面而非 Token 层面应用。我们不再跟踪前 <span class="arithmatex">\(k\)</span> 个 Token 序列，而是跟踪前 <span class="arithmatex">\(k\)</span> 个<strong>推理前缀</strong>：</p>



<div class="arithmatex">\[\mathcal{B}_d = \text{top-}k\left\{ (s_1, \ldots, s_d) : \sum_{i=1}^d \log \pi_\theta(s_i | s_{&lt;i}) + \lambda \cdot V_\phi(s_1, \ldots, s_d) \right\}
\tag{13.14}\]</div>

<p>其中评分将 LLM 的对数概率（流畅性）与价值模型估计（正确性）相结合。这本质上是用学习到的价值函数取代提示式价值函数的 ToT-BFS。</p>

## 13.2.8　迭代精炼与自我纠正 {#iterative-refinement-and-self-correction}

<p>与探索<em>广度</em>（多条并行链）不同，迭代精炼把算力投入到<em>深度</em>——反复改进同一个解答：</p>



<div class="arithmatex">\[y^{(t+1)} = \text{LLM}\!\left(\text{``Improve this solution:&#x27;&#x27;}, y^{(t)}, \text{``Errors found:&#x27;&#x27;}, e^{(t)}\right)
\tag{13.15}\]</div>

<p>其中 <span class="arithmatex">\(e^{(t)}\)</span> 可来自：</p>

<ul>
<li><p><strong>自我验证</strong>：让模型检查自己的答案</p></li>
<li><p><strong>外部验证</strong>：运行代码、符号化地校验数学</p></li>
<li><p><strong>评判模型</strong>：由一个独立模型识别错误</p></li>
</ul>

<p>代表性方法：<strong>Self-Refine</strong> <a class="book-citation" id="cite-madaan2023selfrefine-1" href="../../references.md#ref-madaan2023selfrefine" aria-label="参考文献 246">[246]</a>（迭代自我反馈）、<strong>Reflexion</strong> <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>（通过存入记忆的反思实现的语言化 RL）以及 <strong>LATS</strong> <a class="book-citation" id="cite-zhou2024lats-1" href="../../references.md#ref-zhou2024lats" aria-label="参考文献 438">[438]</a>（树搜索 + 基于反思的剪枝）。</p>

## 13.2.9　方法对比与选型指南 {#method-comparison-and-selection-guide}

<p class="table-caption" id="table-13-2"><span class="table-number">表 13.2　</span>测试时扩展方法的综合比较。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>结构</strong></th><th scope="col"><strong>LLM 调用</strong></th><th scope="col"><strong>可并行</strong></th><th scope="col"><strong>需要 RM？</strong></th><th scope="col"><strong>最适用于</strong></th></tr></thead>
<tbody><tr><td>CoT <a class="book-citation" id="cite-wei2022chain-3" href="../../references.md#ref-wei2022chain" aria-label="参考文献 374">[374]</a></td><td>链</td><td>1</td><td>N/A</td><td>否</td><td>简单—中等难度问题</td></tr>
<tr><td>Self-Consistency <a class="book-citation" id="cite-wang2023selfconsistency-2" href="../../references.md#ref-wang2023selfconsistency" aria-label="参考文献 367">[367]</a></td><td>并行链</td><td><span class="arithmatex">\(N\)</span></td><td>✓ 完全</td><td>否（多数投票）</td><td>答案离散的数学题</td></tr>
<tr><td>Best-of-N + ORM</td><td>并行链</td><td><span class="arithmatex">\(N\)</span> + 1</td><td>✓ 完全</td><td>是（ORM）</td><td>有良好 RM 的通用任务</td></tr>
<tr><td>Best-of-N + PRM</td><td>并行链</td><td><span class="arithmatex">\(N\)</span> + <span class="arithmatex">\(N{\cdot}K\)</span></td><td>✓ 完全</td><td>是（PRM）</td><td>复杂多步推理</td></tr>
<tr><td>ToT <a class="book-citation" id="cite-yao2024tree-2" href="../../references.md#ref-yao2024tree" aria-label="参考文献 409">[409]</a></td><td>树（BFS/DFS）</td><td><span class="arithmatex">\(O(kbd)\)</span></td><td>部分</td><td>LLM 作判官</td><td>结构化搜索问题</td></tr>
<tr><td>GoT <a class="book-citation" id="cite-besta2024graph-2" href="../../references.md#ref-besta2024graph" aria-label="参考文献 25">[25]</a></td><td>DAG</td><td><span class="arithmatex">\(O(kbd)\)</span></td><td>部分</td><td>LLM 作判官</td><td>可分解的问题</td></tr>
<tr><td>MCTS <a class="book-citation" id="cite-kocsis2006bandit-2" href="../../references.md#ref-kocsis2006bandit" aria-label="参考文献 187">[187]</a></td><td>树 + 价值</td><td><span class="arithmatex">\(O(N_{\text{sim}} \cdot d)\)</span></td><td>部分</td><td>是（价值网络）</td><td>困难证明、编程</td></tr>
<tr><td>Self-Refine <a class="book-citation" id="cite-madaan2023selfrefine-2" href="../../references.md#ref-madaan2023selfrefine" aria-label="参考文献 246">[246]</a></td><td>线性（迭代）</td><td><span class="arithmatex">\(2T\)</span></td><td>否</td><td>自我评判</td><td>开放式生成</td></tr>
<tr><td>LATS <a class="book-citation" id="cite-zhou2024lats-2" href="../../references.md#ref-zhou2024lats" aria-label="参考文献 438">[438]</a></td><td>树 + 反思</td><td><span class="arithmatex">\(O(N \cdot d)\)</span></td><td>部分</td><td>LLM 作判官</td><td>Agent 任务</td></tr></tbody>
</table>
</div>

<div class="admonition note"><p class="admonition-title">何时选用哪种方法</p>
<ul>
<li><p><strong>预算 <span class="arithmatex">\(&lt;\)</span> 5<span class="arithmatex">\(\times\)</span> 基础成本</strong>：使用 CoT 或 Self-Consistency。性价比最高。</p></li>
<li><p><strong>预算 5–50<span class="arithmatex">\(\times\)</span></strong>：使用结合 PRM 的 Best-of-N（若你有好的奖励模型）或 <span class="arithmatex">\(b=3, k=2\)</span> 的 ToT-BFS。</p></li>
<li><p><strong>预算 50–500<span class="arithmatex">\(\times\)</span></strong>：使用配备已训练价值函数的 MCTS。这正是 DeepSeek-R1 与 OpenAI o1 所处的区间——以长推理链实现的隐式树搜索。</p></li>
<li><p><strong>需要并行</strong>：Self-Consistency 与 Best-of-N 完全可并行；ToT/MCTS 需要顺序的深度扩展。</p></li>
<li><p><strong>无可用奖励模型</strong>：使用 Self-Consistency（多数投票）或采用 LLM 作判官评估的 ToT。</p></li>
<li><p><strong>可分解的问题</strong>：当问题具有天然子问题时（排序、多文档综合、带模块的代码），GoT 表现出色。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">推理模型中的隐式测试时扩展</p>
<p>现代推理模型（DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a>、OpenAI o1/o3 <a class="book-citation" id="cite-openai2024o1-1" href="../../references.md#ref-openai2024o1" aria-label="参考文献 275">[275]</a><a class="book-citation" id="cite-openai2025o3-1" href="../../references.md#ref-openai2025o3" aria-label="参考文献 278">[278]</a>）通过生成长思维链来执行<strong>隐式测试时扩展</strong>。它们的“思考”Token 起到了类似 MCTS Rollout 的作用：模型探索多种思路、回溯（“Wait, let me reconsider...”）、验证中间步骤，并在更难的子问题上分配更多 Token。R1/o1 训练的关键洞见是：GRPO/RL 让模型在<em>一次生成内</em>就执行这种隐式搜索，从而无需外部编排（ToT Prompt、MCTS 基础设施）。模型自身即成为搜索算法。</p>
</div>
