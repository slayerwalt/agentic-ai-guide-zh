# 28.19　列表式 Reward 与高级 RM 问题 {#listwise-rewards-and-advanced-rm-questions}

<section class="quiz-question" id="q-source-77"><h2 class="question-title">Q：解释 Plackett-Luce 模型。它如何推广 Bradley-Terry？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：Bradley-Terry 建模<em>成对</em>偏好：<span class="arithmatex">\(P(y_1 \succ y_2) = \sigma(r(y_1) - r(y_2))\)</span>。</p>

<p>Plackett-Luce 把 <span class="arithmatex">\(K\)</span> 项的<em>完整排序</em>建模为顺序选择：</p>



<div class="arithmatex">\[P(\pi) = \prod_{i=1}^K \frac{e^{r(y_{\pi(i)})}}{\sum_{j=i}^K e^{r(y_{\pi(j)})}}\]</div>

<p>解释：依次挑选剩余项中最好的。位置 1 = 对全部 <span class="arithmatex">\(K\)</span> 做 Softmax；位置 2 = 对剩余 <span class="arithmatex">\(K-1\)</span> 做 Softmax；以此类推。</p>

<p><strong>推广关系</strong>：当 <span class="arithmatex">\(K=2\)</span> 时，PL 恰好退化为 BT：<span class="arithmatex">\(P(y_1 \succ y_2) = \frac{e^{r(y_1)}}{e^{r(y_1)} + e^{r(y_2)}} = \sigma(r(y_1) - r(y_2))\)</span>。</p>

<p><strong>优势</strong>：<span class="arithmatex">\(K=8\)</span> 项的排序提供 <span class="arithmatex">\(\binom{8}{2} = 28\)</span> 个隐式成对比较，外加相对差距信息——比单一成对样本丰富得多。</p>

<p><em><strong>复习：</strong>第 <a href="../09-reward-models/index.md">9</a> 章（Reward 模型训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-78"><h2 class="question-title">Q：什么是过程奖励模型（PRM），它在什么情况下优于结果奖励模型（Outcome Reward Model, ORM）？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>ORM</strong>：仅对最终输出打分。<span class="arithmatex">\(r(x, y_{\text{final}})\)</span> = 整个响应一个标量。</p>

<p><strong>PRM</strong>：对每一<em>步</em>推理打分。<span class="arithmatex">\(r(x, y_{\text{step } t})\)</span> = 每一中间步一个标量。</p>

<p><strong>PRM 更好的场景</strong>：</p>

<ol>
<li><p><strong>长推理链</strong>：10+ 步的数学题。ORM 无法判断哪一步错了；PRM 提供逐步信用分配。</p></li>
<li><p><strong>搜索/验证</strong>：PRM 支持树搜索（对推理步做 beam search，剪掉步 Reward 低的分支）。</p></li>
<li><p><strong>训练信号密度</strong>：PRM 每条轨迹给 <span class="arithmatex">\(T\)</span> 个 Reward（每步一个），而 ORM 只有一个 <span class="arithmatex">\(\to\)</span> 更低方差的优势估计。</p></li>
</ol>

<p><strong>ORM 更好的场景</strong>：任务短（单轮）；步骤边界不清晰；每步标注成本过高。</p>

<p><strong>PRM 标注</strong>：可通过&quot;Math-Shepherd&quot;方法自动化：对每一步，从该点多次补全整个解。如果从步骤 <span class="arithmatex">\(t\)</span> 出发的补全成功而从步骤 <span class="arithmatex">\(t+1\)</span> 出发的补全失败，则步骤 <span class="arithmatex">\(t+1\)</span> 很可能是错的。</p>

<p><em><strong>复习：</strong>第 <a href="../09-reward-models/index.md">9</a> 章与第 <a href="../13-reasoning/index.md">13</a> 章（Reward 模型训练；大型推理模型的 RL）。</em></p>
</details></section>
