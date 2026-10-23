# 14.1　评估方案设计 {#subsec:eval-scheme}

<p>在收集任何一个数据点之前，从业者必须先决定要测量<em>什么</em>以及<em>如何</em>测量。一套有原则的分类法能避免一个常见错误：仅凭便利性而非与部署目标的对齐性来选择指标。</p>

## 14.1.1　评估类型分类法 {#taxonomy-of-evaluation-types}

#### 内在评估与外在评估（Intrinsic vs. Extrinsic）。 {#intrinsic-vs.-extrinsic-evaluation.}

<p><em>内在</em>（Intrinsic）评估孤立地度量模型输出的性质，不参照下游应用。留出语料上的困惑度、相对参考译文的 BLEU 分数、编码基准上的 pass@<span class="arithmatex">\(k\)</span>，都属于内在评估。<em>外在</em>（Extrinsic）评估则度量模型对真实任务或系统的影响：将 LLM 接入客服流水线后，工单升级率是否下降？编码助手是否提升了开发者的开发速度？</p>

<div class="admonition tip"><p class="admonition-title">内在—外在差距</p>
<p>内在指标廉价且可复现，但往往与真实世界的效用相关性很弱。困惑度更低的模型未必更有用。外在指标昂贵且缓慢，但能直接度量我们真正关心的东西。一套成熟的评估策略会用内在指标做快速迭代，用外在指标做最终验证。</p>
</div>

#### 自动评估与人工评估（Automatic vs. Human）。 {#automatic-vs.-human-evaluation.}

<p><em>自动</em>评估使用确定性函数（BLEU、精确匹配）或学习得到的模型（BERTScore、LLM-as-Judge），无需人工参与即可对输出打分。<em>人工</em>评估则由标注者对模型输出进行打分或排序。表 <a href="#tab:eval-taxonomy">14.1</a>总结了二者的权衡。</p>

<p class="table-caption" id="tab:eval-taxonomy"><span class="table-number">表 14.1　</span>评估方法分类法及其关键权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>类型</strong></th><th scope="col"><strong>成本</strong></th><th scope="col"><strong>速度</strong></th><th scope="col"><strong>可复现性</strong></th><th scope="col"><strong>效度</strong></th></tr></thead>
<tbody><tr><td>自动（基于规则）</td><td>极低</td><td>极快</td><td>完美</td><td>低–中</td></tr>
<tr><td>自动（基于模型）</td><td>低</td><td>快</td><td>高</td><td>中–高</td></tr>
<tr><td>众包人工</td><td>中</td><td>数天</td><td>中</td><td>中</td></tr>
<tr><td>专家人工</td><td>高</td><td>数周</td><td>低–中</td><td>高</td></tr>
<tr><td>外在 / A/B 测试</td><td>极高</td><td>数月</td><td>低</td><td>极高</td></tr></tbody>
</table>
</div>

#### 基于参考与无参考评估（Reference-Based vs. Reference-Free）。 {#reference-based-vs.-reference-free-evaluation.}

<p>基于参考的指标（BLEU、ROUGE、BERTScore）将模型输出与一个或多个金标准参考进行比较。无参考指标（困惑度、LLM-as-Judge、人类偏好）则无需参考即可评估质量。当输出空间过大、无法穷尽收集参考时（例如开放式对话），无参考方法是必不可少的。</p>

## 14.1.2　何时使用何种方法 {#when-to-use-what}

<div class="admonition example"><p class="admonition-title">对话助手的评估策略</p>
<p><strong>开发阶段：</strong>使用自动指标（困惑度、摘要子任务上的 ROUGE、工具调用上的 pass@<span class="arithmatex">\(k\)</span>）进行快速迭代。每晚在标准评测套件（MMLU、HellaSwag、HumanEval）上运行基准测试。</p>

<p><strong>发布前阶段：</strong>开展人类偏好研究，将新模型与上一个检查点对比。在多样化的 Prompt 集合上使用 LLM-as-Judge 进行可扩展的成对比较。</p>

<p><strong>发布后阶段：</strong>监控外在指标（用户满意度分数、任务完成率），并留意生产 Prompt 中的分布漂移。</p>
</div>

<p>一个实用的决策框架：</p>

<ul>
<li><p>若任务有明确正确答案（数学、代码、事实型问答）：使用精确匹配或基于执行的指标。</p></li>
<li><p>若任务是开放式的但存在参考输出：使用基于参考的指标作为下界，并辅以 LLM-as-Judge。</p></li>
<li><p>若任务是主观的（有用性、语气、创造性）：使用人工评估或经过良好校准的 LLM 评判者。</p></li>
<li><p>若任务涉及多步 Agent 行为：使用任务成功率与轨迹效率（见第 <a href="section-06.md#subsec:agentic-metrics">14.6</a>节）。</p></li>
</ul>
