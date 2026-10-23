# 14.4　排序任务的指标 {#subsec:ranking-metrics}

<p>当目标是按质量对模型进行排序时，成对比较数据比绝对分数更为可靠。本节推导 LLM 评估中常用的主要排序系统。</p>

## 14.4.1　ELO 评分系统 {#elo-rating-system}

<p>ELO 系统 <a class="book-citation" id="cite-elo1978rating-1" href="../../references.md#ref-elo1978rating" aria-label="参考文献 84">[84]</a> 最初为国际象棋而设计，它为每位选手（模型）赋予一个标量评分 <span class="arithmatex">\(R\)</span>，使得选手 <span class="arithmatex">\(A\)</span> 对阵选手 <span class="arithmatex">\(B\)</span> 的期望得分为：</p>



<div class="arithmatex">\[E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
\tag{14.8}\]</div>

#### 推导。 {#derivation.}

<p>ELO 模型假设：每位选手在某一局比赛中的表现是一个以其评分为中心的 Logistic 分布所抽取的随机变量。<span class="arithmatex">\(A\)</span> 击败 <span class="arithmatex">\(B\)</span> 的概率为：</p>



<div class="arithmatex">\[P(A \succ B) = \sigma\!\left(\frac{R_A - R_B}{s}\right) = \frac{1}{1 + e^{-(R_A - R_B)/s}}
\tag{14.9}\]</div>

<p>其中 <span class="arithmatex">\(s = 400/\ln(10) \approx 173.7\)</span> 是一个尺度参数，其选择使得 400 分的差距对应于 <span class="arithmatex">\(10:1\)</span> 的胜率比。每场比赛后，给定结果 <span class="arithmatex">\(S_A \in \{0, 0.5, 1\}\)</span>（负、平、胜），评分更新如下：</p>



<div class="arithmatex">\[R_A \leftarrow R_A + K(S_A - E_A), \qquad R_B \leftarrow R_B + K(S_B - E_B)
\tag{14.10}\]</div>

<p>其中 <span class="arithmatex">\(K\)</span> 是控制学习率的 <span class="arithmatex">\(K\)</span>-因子。在 Chatbot Arena 中使用 <span class="arithmatex">\(K = 4\)</span>。</p>

<div class="admonition tip"><p class="admonition-title">ELO 的直觉</p>
<p>ELO 本质上是在 Logistic 模型下，对观察结果的对数似然进行随机梯度下降式更新。每场比赛提供一个带噪的 Gradient 信号；<span class="arithmatex">\(K\)</span>-因子控制步长。<span class="arithmatex">\(K\)</span> 越大，适应越快但噪声越大；<span class="arithmatex">\(K\)</span> 越小则越稳定，但反映真实实力变化的速度越慢。</p>
</div>

#### ELO 的 Bootstrap 置信区间。 {#bootstrap-confidence-intervals-for-elo.}

<p>由于 ELO 评分依赖于比赛被处理的顺序，置信区间通过 Bootstrap 重采样计算：对对战日志有放回地重采样 <span class="arithmatex">\(B = 1000\)</span> 次，对每次重采样都从头重新计算 ELO 评分，并以第 2.5 与第 97.5 百分位作为 95% 置信区间。</p>

## 14.4.2　Bradley–Terry 模型 {#bradleyterry-model}

<p>Bradley–Terry（BT）模型 <a class="book-citation" id="cite-bradley1952rank-1" href="../../references.md#ref-bradley1952rank" aria-label="参考文献 29">[29]</a> 是 ELO 的一种最大似然替代方案。给定 <span class="arithmatex">\(n\)</span> 个模型，其实力参数为 <span class="arithmatex">\(\beta_1, \ldots, \beta_n &gt; 0\)</span>，模型 <span class="arithmatex">\(i\)</span> 击败模型 <span class="arithmatex">\(j\)</span> 的概率为：</p>



<div class="arithmatex">\[P(i \succ j) = \frac{\beta_i}{\beta_i + \beta_j}
\tag{14.11}\]</div>

<p>给定一组成对结果 <span class="arithmatex">\(\{(i_k, j_k, y_k)\}_{k=1}^{M}\)</span>，其中若 <span class="arithmatex">\(i_k\)</span> 击败 <span class="arithmatex">\(j_k\)</span> 则 <span class="arithmatex">\(y_k = 1\)</span>，否则 <span class="arithmatex">\(y_k = 0\)</span>，对数似然为：</p>



<div class="arithmatex">\[\ell(\boldsymbol{\beta}) = \sum_{k=1}^{M} \left[ y_k \log \frac{\beta_{i_k}}{\beta_{i_k} + \beta_{j_k}} + (1-y_k) \log \frac{\beta_{j_k}}{\beta_{i_k} + \beta_{j_k}} \right]
\tag{14.12}\]</div>

<p>极大似然估计 <span class="arithmatex">\(\hat{\boldsymbol{\beta}}\)</span> 可通过迭代缩放或梯度上升求得。BT 模型仅在相差一个乘法常数的意义下可辨识；常用的归一化是 <span class="arithmatex">\(\sum_i \log \beta_i = 0\)</span>。在对数空间中令 <span class="arithmatex">\(\theta_i = \log \beta_i\)</span> 得：</p>



<div class="arithmatex">\[P(i \succ j) = \sigma(\theta_i - \theta_j)
\tag{14.13}\]</div>

<p>这等价于一个带条目特定截距的 Logistic 回归。在能拿到完整对战历史时，BT 模型优于 ELO，因为它同时利用了所有数据，而不是顺序处理比赛。</p>

## 14.4.3　TrueSkill {#trueskill}

<p>TrueSkill <a class="book-citation" id="cite-herbrich2006trueskill-1" href="../../references.md#ref-herbrich2006trueskill" aria-label="参考文献 138">[138]</a> 是一种贝叶斯式的实力评分系统，将每位选手的实力建模为高斯随机变量 <span class="arithmatex">\(s_i \sim \mathcal{N}(\mu_i, \sigma_i^2)\)</span>。选手 <span class="arithmatex">\(i\)</span> 在某一局中的表现为 <span class="arithmatex">\(p_i = s_i + \epsilon_i\)</span>，其中 <span class="arithmatex">\(\epsilon_i \sim \mathcal{N}(0, \beta^2)\)</span> 为比赛特定的噪声。当 <span class="arithmatex">\(p_i &gt; p_j\)</span> 时，选手 <span class="arithmatex">\(i\)</span> 击败选手 <span class="arithmatex">\(j\)</span>。</p>

<p>观察到 <span class="arithmatex">\(i \succ j\)</span> 后的后验更新通过期望传播（Expectation Propagation, EP）计算。胜方的关键更新方程为：</p>



<div class="arithmatex">\[\mu_i \leftarrow \mu_i + \frac{\sigma_i^2}{c} \cdot v\!\left(\frac{\mu_i - \mu_j}{c}\right)
\tag{14.14}\]</div>



<div class="arithmatex">\[\sigma_i^2 \leftarrow \sigma_i^2 \left[1 - \frac{\sigma_i^2}{c^2} \cdot w\!\left(\frac{\mu_i - \mu_j}{c}\right)\right]
\tag{14.15}\]</div>

<p>其中 <span class="arithmatex">\(c = \sqrt{2\beta^2 + \sigma_i^2 + \sigma_j^2}\)</span>，<span class="arithmatex">\(v(t) = \phi(t)/\Phi(t)\)</span>，<span class="arithmatex">\(w(t) = v(t)(v(t) + t)\)</span> 为截断高斯修正因子（<span class="arithmatex">\(\phi\)</span> 与 <span class="arithmatex">\(\Phi\)</span> 分别是标准正态的 PDF 与 CDF）。TrueSkill 的不确定性估计 <span class="arithmatex">\(\sigma_i\)</span> 在识别需要更多评估数据的模型时尤其有用。</p>

## 14.4.4　带置信区间的胜率 {#win-rate-with-confidence-intervals}

<p>最简单的排序指标是胜率：模型 <span class="arithmatex">\(A\)</span> 在成对比较中胜出的比例。给定 <span class="arithmatex">\(n\)</span> 次比较，其中 <span class="arithmatex">\(w\)</span> 次胜出，胜率为 <span class="arithmatex">\(\hat{p} = w/n\)</span>。在 <span class="arithmatex">\(p = 0\)</span> 与 <span class="arithmatex">\(p = 1\)</span> 附近，Wilson 分数置信区间 <a class="book-citation" id="cite-wilson1927probable-1" href="../../references.md#ref-wilson1927probable" aria-label="参考文献 380">[380]</a> 的覆盖性能优于朴素的 Wald 区间，因此更被推荐：</p>



<div class="arithmatex">\[\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
\tag{14.16}\]</div>

<p>其中 95% 区间对应 <span class="arithmatex">\(z = 1.96\)</span>。对于多方比较，胜率应针对一个固定的基线模型来计算，以确保可比性。</p>

## 14.4.5　Chatbot Arena 方法论 {#chatbot-arena-methodology}

<p>Chatbot Arena <a class="book-citation" id="cite-zheng2023judging-1" href="../../references.md#ref-zheng2023judging" aria-label="参考文献 433">[433]</a> 将上述要素组合为一个生产规模的评估系统：</p>

<ol>
<li><p>用户提交 Prompt，并获得两个匿名化模型的回答。</p></li>
<li><p>用户对偏好的回答进行投票（或声明平局）。</p></li>
<li><p>投票通过 BT 模型聚合以生成排行榜。</p></li>
<li><p>对每个模型的分数报告 Bootstrap 置信区间。</p></li>
<li><p>置信区间重叠的模型在统计上被视为无法区分。</p></li>
</ol>

<p>截至 2024 年，Chatbot Arena 已累计收集超过一百万条人类偏好投票，是目前公开可得的最大规模 LLM 偏好数据集。</p>
