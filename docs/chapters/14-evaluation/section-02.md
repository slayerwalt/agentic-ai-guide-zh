# 14.2　评估数据采集 {#subsec:data-collection}

<p>高质量的评估数据是可信基准的基础。本节涵盖人工标注流水线的设计、标注质量的统计度量，以及在众包与专家标注之间如何取舍。</p>

## 14.2.1　人工标注流水线 {#human-annotation-pipelines}

<p>一条健壮的标注流水线包含五个阶段：</p>

<ol>
<li><p><strong>任务定义。</strong>精确说明标注任务：评分对象是什么、采用何种量表、依据哪些标准。此阶段的歧义会传导为带噪标签。</p></li>
<li><p><strong>规范编写。</strong>撰写标注规范，并配以覆盖边界情形的详细示例。在全面部署前先用小规模试点组反复打磨。</p></li>
<li><p><strong>标注者招募与培训。</strong>选择具备相应背景知识的标注者。组织一次校准会议，让标注者标注相同样本并讨论分歧。</p></li>
<li><p><strong>质量控制。</strong>在标注队列中嵌入带已知标签的金标样本。对在金标样本上准确率低于阈值的标注者进行标记。</p></li>
<li><p><strong>聚合。</strong>使用多数投票、平均，或概率模型（如 Dawid–Skene）来合并每条样本的多次标注。</p></li>
</ol>

## 14.2.2　标注者间一致性 {#inter-annotator-agreement}

<p>原始一致率（所有标注者达成一致的样本比例）是一个不充分的度量，因为它没有考虑偶然一致。两个标准的偶然性修正度量是 Cohen’s <span class="arithmatex">\(\kappa\)</span> <a class="book-citation" id="cite-cohen1960coefficient-1" href="../../references.md#ref-cohen1960coefficient" aria-label="参考文献 60">[60]</a>（两位标注者）和 Fleiss’ <span class="arithmatex">\(\kappa\)</span> <a class="book-citation" id="cite-fleiss1971measuring-1" href="../../references.md#ref-fleiss1971measuring" aria-label="参考文献 92">[92]</a>（多位标注者）。</p>

#### Cohen’s Kappa。 {#cohens-kappa.}

<p>给定两位标注者将 <span class="arithmatex">\(N\)</span> 个样本标注为 <span class="arithmatex">\(k\)</span> 个类别，令 <span class="arithmatex">\(p_o\)</span> 为观察到的一致率，<span class="arithmatex">\(p_e\)</span> 为独立性假设下的期望一致率：</p>

<span id="eq:cohens-kappa"></span>

<div class="arithmatex">\[\kappa = \frac{p_o - p_e}{1 - p_e}

\tag{14.1}\]</div>

<p>其中</p>



<div class="arithmatex">\[p_o = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{annotator 1 agrees with annotator 2 on item } i]
\tag{14.2}\]</div>

<p>以及</p>



<div class="arithmatex">\[p_e = \sum_{c=1}^{k} p_{1c} \cdot p_{2c}
\tag{14.3}\]</div>

<p>其中 <span class="arithmatex">\(p_{jc}\)</span> 是标注者 <span class="arithmatex">\(j\)</span> 将样本判为类别 <span class="arithmatex">\(c\)</span> 的比例。Cohen’s <span class="arithmatex">\(\kappa\)</span> 的取值从 <span class="arithmatex">\(-1\)</span>（完全不一致）经 <span class="arithmatex">\(0\)</span>（偶然一致）到 <span class="arithmatex">\(1\)</span>（完全一致）。一般认为 <span class="arithmatex">\(0.6\)</span> 以上可以接受；<span class="arithmatex">\(0.8\)</span> 以上为强一致。</p>

#### Fleiss’ Kappa。 {#fleiss-kappa.}

<p>对于 <span class="arithmatex">\(n\)</span> 位标注者将 <span class="arithmatex">\(N\)</span> 个样本标注为 <span class="arithmatex">\(k\)</span> 个类别，令 <span class="arithmatex">\(n_{ij}\)</span> 为将样本 <span class="arithmatex">\(i\)</span> 标注为类别 <span class="arithmatex">\(j\)</span> 的标注者人数。定义：</p>



<div class="arithmatex">\[\bar{P}_i = \frac{1}{n(n-1)} \sum_{j=1}^{k} n_{ij}(n_{ij} - 1), \qquad \bar{P} = \frac{1}{N}\sum_{i=1}^{N}\bar{P}_i
\tag{14.4}\]</div>



<div class="arithmatex">\[\bar{P}_j^e = \frac{1}{Nn}\sum_{i=1}^{N} n_{ij}, \qquad P_e = \sum_{j=1}^{k} \left(\bar{P}_j^e\right)^2
\tag{14.5}\]</div>



<div class="arithmatex">\[\kappa_F = \frac{\bar{P} - P_e}{1 - P_e}
\tag{14.6}\]</div>

<div class="admonition warning"><p class="admonition-title">Kappa 的局限</p>
<p>Kappa 对类别分布敏感：当某一类别占据主导时，即使原始一致率很高，kappa 仍可能很低（即<em>kappa 悖论</em>）。对于序数量表，加权 kappa（按分歧的距离比例进行惩罚）更为合适。在 LLM 评估中，评分通常采用 1–5 的 Likert 量表，应始终报告加权 kappa。</p>
</div>

## 14.2.3　标注规范设计 {#annotation-guideline-design}

<p>有效的标注规范通常具备以下若干特征：</p>

<ul>
<li><p><strong>可操作化的标准。</strong>用具体、可观察的行为替换“有用”这类模糊词汇：例如“回答直接处理了用户的问题，并提供了完成所述任务所需的全部信息”。</p></li>
<li><p><strong>详细示例。</strong>每个评分等级至少提供两个示例，包括边界情形。</p></li>
<li><p><strong>决策树。</strong>对于复杂任务，用流程图引导标注者完成一系列二元决策，可降低认知负担并提升一致性。</p></li>
<li><p><strong>明确的范围。</strong>说明标注者<em>不</em>应考虑哪些因素（例如“不要因风格偏好而扣分；只关注事实准确性”）。</p></li>
</ul>

## 14.2.4　众包标注与专家标注 {#crowdsourcing-vs.-expert-annotation}

<p class="table-caption" id="table-14-2"><span class="table-number">表 14.2　</span>LLM 评估中众包标注与专家标注的对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>众包</strong></th><th scope="col"><strong>专家标注</strong></th></tr></thead>
<tbody><tr><td>单条样本成本</td><td>低（$0.01–$0.10）</td><td>高（$1–$50）</td></tr>
<tr><td>吞吐量</td><td>极高</td><td>低</td></tr>
<tr><td>领域知识</td><td>低</td><td>高</td></tr>
<tr><td>一致性</td><td>不稳定</td><td>高</td></tr>
<tr><td>适用任务</td><td>简单偏好、流畅度</td><td>技术准确性、安全性</td></tr>
<tr><td>平台</td><td>MTurk、Prolific、Scale AI</td><td>领域专家、内部团队</td></tr>
<tr><td>质量控制</td><td>金标样本、注意力检查</td><td>校准会议、同行复核</td></tr></tbody>
</table>
</div>

<p>对于安全关键型评估（例如检测有害输出、评估医疗建议），专家标注是不可妥协的。对于大规模偏好采集（例如构建 Reward 模型训练集），配以严格质量控制的众包往往是唯一可行的选择。</p>
