# 14.8　评估陷阱 {#subsec:eval-pitfalls}

<p>即便是精心设计的评估流水线也可能给出误导性的结果。本节梳理最常见的失效模式。</p>

## 14.8.1　基准污染 {#benchmark-contamination}

<p>当评估数据出现在模型的训练集中时（直接以原文出现，或间接以复述/语义相似形式出现），就发生了基准污染（benchmark contamination）。被污染的模型会取得虚高的分数，无法反映其真实的泛化能力。</p>

#### 检测方法： {#detection-methods}

<ul>
<li><p><strong><span class="arithmatex">\(n\)</span>-gram 重叠：</strong>计算与训练语料具有高 <span class="arithmatex">\(n\)</span>-gram 重叠（例如 ROUGE-L <span class="arithmatex">\(&gt; 0.8\)</span>）的评估样本比例。</p></li>
<li><p><strong>成员推断（Membership Inference）：</strong>使用成员推断攻击来估计每个评估样本曾出现在训练集中的概率。</p></li>
<li><p><strong>金丝雀串（Canary strings）：</strong>在评估样本中嵌入独特的随机生成字符串，并检查模型是否能将其补全。</p></li>
<li><p><strong>时间留出：</strong>使用模型训练截止日期之后产生的评估数据。</p></li>
</ul>

#### 缓解： {#mitigation}

<ul>
<li><p>维护一个永不公开发布的私有测试集。</p></li>
<li><p>定期用新样本刷新基准。</p></li>
<li><p>报告训练数据截止日期以及去污染流程。</p></li>
</ul>

## 14.8.2　对基准的过拟合 {#overfitting-to-benchmarks}

<p>即使没有直接污染，模型也可能通过反复评估与超参调优而被隐式地为特定基准而优化。这是一种<em>自适应过拟合</em>（adaptive overfitting）：基准向模型开发决策泄漏了信息。</p>

<div class="admonition warning"><p class="admonition-title">基准的生命周期</p>
<p>基准的效用会随着研究社区对其的优化而随时间退化。MMLU <a class="book-citation" id="cite-hendrycks2021measuring-1" href="../../references.md#ref-hendrycks2021measuring" aria-label="参考文献 137">[137]</a> 曾是对世界知识的挑战性测试，如今模型在其上已接近人类水平，但这些模型在新的知识任务上仍会失败。新基准应被视为临时的信号源，而非永久的金标准。</p>
</div>

## 14.8.3　评估中的 Goodhart 定律 {#goodharts-law-in-evaluation}

<p>Goodhart 定律指出：<em>“当一个度量成为目标时，它就不再是好的度量。”</em> <a class="book-citation" id="cite-goodhart1984problems-1" href="../../references.md#ref-goodhart1984problems" aria-label="参考文献 112">[112]</a> 在 LLM 评估中，这有多种体现：</p>

<ul>
<li><p><strong>Reward 欺骗（Reward hacking）：</strong>用基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）训练的模型可能学会利用 Reward 模型，而非真正提升质量。模型可能学会输出冗长、听起来自信但实际上事实错误的回答，这类回答在 Reward 模型上得分却很高。</p></li>
<li><p><strong>指标博弈：</strong>为最大化 BLEU 或 ROUGE 而微调的模型，可能产出在这些指标上得分良好但对人类用处不大的输出。</p></li>
<li><p><strong>评判者博弈：</strong>用 LLM-as-Judge 反馈训练的模型可能学到评判者的偏置（例如冗长偏置），而非真正提升质量。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">抵御 Goodhart 定律的策略</p>
<ol>
<li><p><strong>指标多样化：</strong>使用来自不同家族的多个指标；一个能博弈某个指标的模型，未必能同时博弈所有指标。</p></li>
<li><p><strong>留出评估：</strong>保留一些不参与训练或模型选择的评估指标。</p></li>
<li><p><strong>人工抽查：</strong>定期采样模型输出供人工审阅，独立于自动化指标之外。</p></li>
<li><p><strong>对抗性评估：</strong>主动探测自动化指标会漏掉的失效模式。</p></li>
<li><p><strong>外在验证：</strong>定期用外在结果来验证内在指标。</p></li>
</ol>
</div>

## 14.8.4　其他陷阱 {#additional-pitfalls}

#### Prompt 敏感性。 {#prompt-sensitivity.}

<p>LLM 的性能可能因评估 Prompt 的微小改动而剧烈变化（例如加入&quot;Think step by step&quot;或更改回答格式）。务必报告所使用的确切 Prompt，并考虑在多种 Prompt 变体上进行评估。</p>

#### 聚合伪影。 {#aggregation-artefacts.}

<p>在难度水平与分数分布各异的任务上对分数取平均，可能产生误导性的聚合指标。一个在简单任务上表现优异但在困难任务上失败的模型，可能与一个表现均衡的模型获得相同的平均分。</p>

#### 人工评估中的选择偏差。 {#selection-bias-in-human-evaluation.}

<p>人类评估者并不是终端用户的随机样本。众包平台上的标注者与目标用户群体可能在偏好、文化背景与领域知识上存在差异。</p>

#### 评估—部署失配。 {#evaluationdeployment-mismatch.}

<p>评估 Prompt 往往比真实用户查询更短、更干净、形式也更规整。在基准 Prompt 上表现良好的模型，在生产环境中遇到嘈杂、含糊的多轮对话（Multi-Turn）时可能显著退化。</p>

<div class="admonition question"><p class="admonition-title">评估设计的关键问题</p>
<p>在部署评估流水线之前，请自问：</p>

<ol>
<li><p>评估指标是否与部署目标对齐？</p></li>
<li><p>评估数据是否代表目标分布？</p></li>
<li><p>是否已评估污染与过拟合风险？</p></li>
<li><p>是否对所有指标都报告了置信区间？</p></li>
<li><p>评估是否可复现（固定随机种子、Prompt 版本化、公开测试集）？</p></li>
<li><p>评估是否已与人类判断或外在结果进行过验证？</p></li>
</ol>
</div>
