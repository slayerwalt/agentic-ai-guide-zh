# 28.21　LLM 评估问题 {#llm-evaluation-questions}

<section class="quiz-question" id="q-source-83"><h2 class="question-title">Q：推导 ELO 评分更新规则，并解释 Chatbot Arena 为何使用它</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>ELO 推导</strong>：</p>

<p>玩家 A 对 B 的期望得分：<span class="arithmatex">\(E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}\)</span>（逻辑斯蒂模型）。</p>

<p>对局后，实际得分 <span class="arithmatex">\(S_A \in \{0, 0.5, 1\}\)</span>：<span class="arithmatex">\(R_A&#x27; = R_A + K(S_A - E_A)\)</span></p>

<p><span class="arithmatex">\(K\)</span> 因子控制更新幅度（<span class="arithmatex">\(K\)</span> 越大对近期结果反应越敏感）。</p>

<p><strong>为什么 Chatbot Arena 使用 ELO</strong>：</p>

<ol>
<li><p><strong>传递性</strong>：若 A 胜 B 且 B 胜 C，ELO 预测 A 胜 C。这从成对比较给出全序。</p></li>
<li><p><strong>在线更新</strong>：新增模型无需重新评估所有对。每次新比较都增量更新评分。</p></li>
<li><p><strong>置信度</strong>：在 <span class="arithmatex">\(N\)</span> 次比较后，评分不确定性以 <span class="arithmatex">\(O(1/\sqrt{N})\)</span> 收缩。标准误：<span class="arithmatex">\(\text{SE} \approx \frac{400}{\sqrt{N}}\)</span>。</p></li>
<li><p><strong>捕捉人类偏好</strong>：真实用户无需阐明标准即可提供诚实偏好。聚合揭示了真实模型质量。</p></li>
</ol>

<p><strong>Chatbot Arena 细节</strong>：使用 Bradley-Terry MLE（收敛时等价于 ELO），并配合 Bootstrap 置信区间。风格控制 ELO 去除长度/格式偏倚。</p>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章（LLM 评估）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-84"><h2 class="question-title">Q：代码生成的 pass@k 指标是什么，为什么无偏估计很重要？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>pass@k</strong> = 生成的 <span class="arithmatex">\(k\)</span> 个样本中至少有一个通过全部测试用例的概率。</p>

<p><strong>朴素（有偏）估计器</strong>：生成 <span class="arithmatex">\(k\)</span> 个样本，看是否有任意一个通过。问题：高方差、昂贵（每题需要很多次试验）。</p>

<p><strong>无偏估计器</strong>（Chen 等，2021）：生成 <span class="arithmatex">\(n \geq k\)</span> 个样本，统计其中通过的数量 <span class="arithmatex">\(c\)</span>：</p>



<div class="arithmatex">\[\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\]</div>

<p><strong>无偏为何重要</strong>：</p>

<ol>
<li><p>一次性生成 <span class="arithmatex">\(n\)</span> 个样本（例如 <span class="arithmatex">\(n=200\)</span>），从同一批样本算出 pass@1、pass@10、pass@100</p></li>
<li><p>无需将整个评估重复 <span class="arithmatex">\(k\)</span> 次</p></li>
<li><p>统计上精确（组合论证：不含任何正确样本的 <span class="arithmatex">\(k\)</span>-子集占比）</p></li>
<li><p>通过对数空间数值稳定计算：<span class="arithmatex">\(\text{pass@}k = 1 - \exp\left(\sum_{i=0}^{k-1} \log(n-c-i) - \log(n-i)\right)\)</span></p></li>
</ol>

<p><strong>直觉</strong>：若 50/200 个样本通过（<span class="arithmatex">\(c=50\)</span>，<span class="arithmatex">\(n=200\)</span>），pass@1 <span class="arithmatex">\(\approx 0.25\)</span>，pass@10 <span class="arithmatex">\(\approx 0.94\)</span>。该估计器统计大小为 <span class="arithmatex">\(k\)</span> 的抽取中至少包含一个成功的比例。</p>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章与第 <a href="../21-environments/index.md">21</a> 章（LLM 评估；Agent 环境）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-85"><h2 class="question-title">Q：如何检测和缓解基准污染（Benchmark Contamination）？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>污染</strong>：训练数据中包含基准测试样例（或其近似改写），使分数虚高。</p>

<p><strong>检测方法</strong>：</p>

<ol>
<li><p><strong>N-gram 重叠</strong>：检查训练数据是否与测试条目精确或近似匹配。8-gram 重叠且覆盖率 <span class="arithmatex">\(&gt;\)</span>80% 即可疑。</p></li>
<li><p><strong>Canary 字符串</strong>：在测试集中插入唯一标识符；检查模型是否能复现它们。</p></li>
<li><p><strong>改写基准</strong>：创建语义等价但文本不同的基准版本。准确率大幅下降表明存在记忆。</p></li>
<li><p><strong>时序分析</strong>：比较模型在训练截止前 vs 截止后测试条目上的表现。在旧条目上异常高表现暗示污染。</p></li>
<li><p><strong>成员推断</strong>：统计检验判断特定样例是否出现在训练数据中。</p></li>
</ol>

<p><strong>缓解</strong>：</p>

<ul>
<li><p><strong>动态基准</strong>：定期生成新测试条目（LiveCodeBench、Chatbot Arena）</p></li>
<li><p><strong>私有测试集</strong>：保持测试条目保密（LMSYS）</p></li>
<li><p><strong>训练期去污染</strong>：从训练数据中删除检测到的重叠</p></li>
<li><p><strong>报告污染分析</strong>：在基准分数旁披露重叠度指标</p></li>
</ul>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章（LLM 评估）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-86"><h2 class="question-title">Q：解释 LLM-as-Judge 中的位置偏倚（Position Bias），以及如何缓解</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>位置偏倚</strong>：当用 LLM 评判两个响应（A vs B）时，模型系统性地偏好特定位置的响应（通常是第一个或最后一个），与质量无关。</p>

<p><strong>经验幅度</strong>：GPT-4 表现出 10%–15% 的位置偏倚；Claude 为 5%–10%。较小的模型偏倚更大。</p>

<p><strong>缓解策略</strong>：</p>

<ol>
<li><p><strong>位置交换</strong>：每对评判两次（A-B 与 B-A）。最终决定 = 多数。若不一致，则标为“平局”。这能消除系统性位置偏倚，但成本翻倍。</p></li>
<li><p><strong>多评委合议</strong>：使用 3 个及以上不同模型作为评委。多数投票减少单一模型的偏倚。</p></li>
<li><p><strong>参考引导</strong>：提供评分量表或参考答案。评委按量表独立为每个响应打分，再比较分数（完全消除成对比较）。</p></li>
<li><p><strong>校准 Prompt</strong>：加入明确指令：“呈现顺序是随机的，不应影响你的判断。”</p></li>
</ol>

<p><strong>其他偏倚</strong>：冗长偏倚（偏好更长响应）、自我增强偏倚（模型偏好自身输出）、权威偏倚（顺从于引用来源的响应）。</p>

<p><em><strong>复习：</strong>第 <a href="../14-evaluation/index.md">14</a> 章（LLM 评估）。</em></p>
</details></section>
