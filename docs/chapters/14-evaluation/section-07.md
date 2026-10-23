# 14.7　LLM-as-Judge {#subsec:llm-as-judge}

<p>LLM-as-Judge <a class="book-citation" id="cite-zheng2023judging-1" href="../../references.md#ref-zheng2023judging" aria-label="参考文献 433">[433]</a> 使用一个能力强的 LLM 来评估其他（或同一个）LLM 的输出。该方法无需人工标注就能扩展到大规模评估集合，并能为其判断提供详尽的理由。</p>

## 14.7.1　设置与 Prompt 模板 {#setup-and-prompt-templates}

<p>向评判者提供一个 Prompt、一个或多个模型回答，以及一份评估细则。常见的三种格式：</p>

#### 点评式评分（Pointwise scoring）。 {#pointwise-scoring.}

<p>评判者对单个回答给出绝对分数：</p>

<div class="admonition example"><p class="admonition-title">Pointwise Judge Prompt</p>
<div class="highlight"><pre><span></span><code><span class="n">POINTWISE_PROMPT</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">You are an expert evaluator. Rate the following response on a scale</span>
<span class="s2">of 1-10 for helpfulness, accuracy, and clarity.</span>

<span class="s2">[Question]</span>
<span class="si">{question}</span>

<span class="s2">[Response]</span>
<span class="si">{response}</span>

<span class="s2">Provide your evaluation in the following format:</span>
<span class="s2">Reasoning: &lt;step-by-step analysis&gt;</span>
<span class="s2">Score: &lt;integer from 1 to 10&gt;</span>
<span class="s2">"""</span>
</code></pre></div>
</div>

#### 成对比较（Pairwise comparison）。 {#pairwise-comparison.}

<p>评判者比较两个回答并选出更优者：</p>

<div class="admonition example"><p class="admonition-title">Pairwise Judge Prompt</p>
<div class="highlight"><pre><span></span><code><span class="n">PAIRWISE_PROMPT</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">You are an expert evaluator. Compare the two responses below and</span>
<span class="s2">determine which is better. Consider helpfulness, accuracy, and</span>
<span class="s2">depth of explanation.</span>

<span class="s2">[Question]</span>
<span class="si">{question}</span>

<span class="s2">[Response A]</span>
<span class="si">{response_a}</span>

<span class="s2">[Response B]</span>
<span class="si">{response_b}</span>

<span class="s2">Output exactly one of: [[A]], [[B]], or [[C]] (tie).</span>
<span class="s2">Reasoning: &lt;your analysis&gt;</span>
<span class="s2">Verdict: &lt;[[A]], [[B]], or [[C]]&gt;</span>
<span class="s2">"""</span>
</code></pre></div>
</div>

#### 参考引导式评分（Reference-guided scoring）。 {#reference-guided-scoring.}

<p>向评判者提供参考答案，并据此对回答进行打分。这对评判者本身可能缺乏可靠知识的事实型任务尤其有用。</p>

## 14.7.2　位置偏置的缓解 {#position-bias-mitigation}

<p>LLM 评判者会表现出<em>位置偏置</em>（position bias）：系统性地偏好出现在特定位置（首位或末位）的回答。此偏置可能高达 10–15 个百分点。缓解策略：</p>

<ol>
<li><p><strong>交换增强：</strong>对每一对回答都以两种顺序（A vs. B 与 B vs. A）进行评判。判断一致则采纳；判断不一致则记为平局。</p></li>
<li><p><strong>校准式 Prompt：</strong>显式指令评判者：“你的评估不应受到回答呈现顺序的影响。”</p></li>
<li><p><strong>评判者集成：</strong>使用多个评判者并配以不同的位置顺序，再聚合其裁决。</p></li>
<li><p><strong>强制思维链：</strong>要求评判者在给出裁决前先产出详尽理由，以降低对表层位置线索的依赖。</p></li>
</ol>

<div class="admonition warning"><p class="admonition-title">冗长偏置</p>
<p>LLM 评判者同样表现出冗长偏置：更长的回答会被系统性地偏好，即便额外内容是无关或重复的。缓解方式：指示评判者对不必要的长度进行惩罚，并关注信息的质量而非数量。或者，在评判前将回答截断到固定长度。</p>
</div>

## 14.7.3　多评判者面板 {#multi-judge-panels}

<p>单一评判者可能带有系统性偏置。来自不同模型家族的评判者面板能提供更稳健的评估。给定 <span class="arithmatex">\(J\)</span> 位评判者，其裁决为 <span class="arithmatex">\(v_1, \ldots, v_J \in \{A, B, \text{tie}\}\)</span>，面板裁决由多数投票决定。面板一致率为：</p>



<div class="arithmatex">\[\text{Agreement} = \frac{1}{\binom{J}{2}} \sum_{i &lt; j} \mathbf{1}[v_i = v_j]
\tag{14.36}\]</div>

<p>对于三人评判者面板，一致裁决（三人全部同意）视为高置信度；2–1 分裂视为中置信度；三方平局视为低置信度。</p>

## 14.7.4　LLM 评判者的一致性指标 {#agreement-metrics-for-llm-judges}

<p>为验证 LLM 评判者，将其裁决与留出集上的人工标注进行比较。关键指标：</p>

<ul>
<li><p><strong>一致率：</strong>评判者与人类标注一致的样本比例。</p></li>
<li><p><strong>Cohen’s <span class="arithmatex">\(\kappa\)</span>：</strong>经偶然性修正的一致率（公式 <a href="section-02.md#eq:cohens-kappa">14.1</a>）。</p></li>
<li><p><strong>Spearman’s <span class="arithmatex">\(\rho\)</span>：</strong>评判者分数与人类分数之间的秩相关，适用于序数评分。</p></li>
<li><p><strong>Kendall’s <span class="arithmatex">\(\tau\)</span>：</strong>另一种秩相关，对平局更具鲁棒性。</p></li>
</ul>

<p>若评判者在一份具有代表性的样本上与人类标注者达到 <span class="arithmatex">\(\kappa &gt; 0.6\)</span> 且一致率 <span class="arithmatex">\(&gt; 80\%\)</span>，则视为可靠。</p>

## 14.7.5　G-Eval 框架 {#g-eval-framework}

<p>G-Eval <a class="book-citation" id="cite-liu2023geval-1" href="../../references.md#ref-liu2023geval" aria-label="参考文献 231">[231]</a> 是一个面向 LLM 评估的结构化框架，使用思维链（Chain-of-Thought，CoT）提示与 Token 概率加权来产出更可靠的分数。框架如下：</p>

<ol>
<li><p><strong>生成评估步骤：</strong>提示 LLM 为评估任务生成详尽的评分细则（例如“列出你评估一篇摘要连贯性时所采取的步骤”）。</p></li>
<li><p><strong>以概率加权方式打分：</strong>对每个分值 <span class="arithmatex">\(s \in \{1, 2, 3, 4, 5\}\)</span>，从评判模型获取对数概率 <span class="arithmatex">\(\log P_\theta(s \mid \text{prompt, steps, response})\)</span>。最终分数为以概率加权的平均：</p>



<div class="arithmatex">\[\text{G-Eval score} = \sum_{s=1}^{5} s \cdot \frac{e^{\log P_\theta(s)}}{\sum_{s&#x27;=1}^{5} e^{\log P_\theta(s&#x27;)}}
\tag{14.37}\]</div></li>
<li><p><strong>归一化：</strong>将分数除以最大分值，映射到 <span class="arithmatex">\([0, 1]\)</span>。</p></li>
</ol>

<p>G-Eval 与人类判断的相关性高于直接提示，尤其在连贯性与一致性这类细腻维度上，因为概率加权捕捉了评判者的不确定性，而非强制其作离散选择。</p>

<div class="admonition tip"><p class="admonition-title">为何 G-Eval 有效</p>
<p>标准提示要求评判者输出单个 Token（例如&quot;4&quot;），这丢弃了模型的不确定性。G-Eval 读取所有分值 Token 上的概率分布，实质上是在评判者的信念下计算期望分数。这类似于使用后验分布的均值而非众数。</p>
</div>
