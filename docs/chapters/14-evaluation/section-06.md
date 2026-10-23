# 14.6　Agent 任务的指标 {#subsec:agentic-metrics}

<p>Agent 化的 LLM 在环境中运作、执行动作序列，并须完成多步任务。标准生成指标在此并不充分；Agent 评估需要能够刻画任务完成度、效率以及中间步骤质量的指标。</p>

## 14.6.1　任务成功率 {#task-success-rate}

<p>Agent 任务的主要指标是任务成功率（Task Success Rate, TSR）：Agent 达成指定目标状态的任务比例：</p>



<div class="arithmatex">\[\text{TSR} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathbf{1}[\text{goal}(\tau) \text{ achieved}]
\tag{14.31}\]</div>

<p>目标的达成通常由一个确定性的 Oracle 来验证（例如检查数据库状态、文件系统状态或测试用例执行结果）。对于允许部分得分的任务，可以定义分级成功度量：</p>



<div class="arithmatex">\[\text{TSR}_{\text{graded}} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{score}(\tau) \in [0, 1]
\tag{14.32}\]</div>

## 14.6.2　轨迹效率 {#trajectory-efficiency}

<p>一个成功的 Agent 应以尽可能少的非必要动作完成任务。轨迹效率衡量最优轨迹长度与 Agent 实际轨迹长度之比：</p>



<div class="arithmatex">\[\eta = \frac{L^*}{L_{\text{agent}}}
\tag{14.33}\]</div>

<p>其中 <span class="arithmatex">\(L^*\)</span> 是最短成功轨迹的长度（由 Oracle 或人类专家计算得出），<span class="arithmatex">\(L_{\text{agent}}\)</span> 是 Agent 所执行的动作数。<span class="arithmatex">\(\eta \in (0, 1]\)</span>，<span class="arithmatex">\(\eta = 1\)</span> 表示最优效率。对于失败的轨迹，<span class="arithmatex">\(\eta = 0\)</span>。</p>

<p>一个互补指标是<em>冗余率</em>（redundancy rate）：Agent 动作中不出现在任何最优轨迹里的比例。</p>

## 14.6.3　工具调用准确性 {#tool-use-accuracy}

<p>对于调用外部工具（API、代码解释器、搜索引擎）的 Agent，工具调用（Tool Calling）准确性衡量调用的正确性：</p>



<div class="arithmatex">\[\text{TUA} = \frac{\text{\# correct tool calls}}{\text{\# total tool calls}}
\tag{14.34}\]</div>

<p>一次工具调用被视为正确当且仅当：（a）选择了正确的工具，（b）参数有效，（c）该调用发生在轨迹中合适的时机。当工具选择正确但参数错误时，可酌情给予部分得分。</p>

## 14.6.4　多步推理准确性 {#multi-step-reasoning-accuracy}

<p>对于需要推理链的任务（例如多跳问答、数学解题），步骤级准确性衡量正确推理步骤所占的比例：</p>



<div class="arithmatex">\[\text{SRA} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{1}{|S_\tau|} \sum_{s \in S_\tau} \mathbf{1}[s \text{ is correct}]
\tag{14.35}\]</div>

<p>其中 <span class="arithmatex">\(S_\tau\)</span> 是轨迹 <span class="arithmatex">\(\tau\)</span> 中的推理步骤集合。步骤的正确性可由过程奖励模型（Process Reward Model, PRM）或人工标注来验证。</p>

## 14.6.5　SWE-bench 方法论 {#swe-bench-methodology}

<p>SWE-bench <a class="book-citation" id="cite-jimenez2024swebench-1" href="../../references.md#ref-jimenez2024swebench" aria-label="参考文献 169">[169]</a> 在真实软件工程任务上评估 LLM：给定一个 GitHub issue 描述与仓库代码库，模型必须生成解决该 issue 的 patch。评估流程如下：</p>

<ol>
<li><p>将 issue 描述与相关代码上下文提供给模型。</p></li>
<li><p>模型生成一个 patch（统一 diff 格式）。</p></li>
<li><p>将 patch 应用到仓库。</p></li>
<li><p>执行仓库的测试套件；若全部测试通过，则任务成功。</p></li>
</ol>

<p>主要指标为 <strong>% Resolved</strong>：生成的 patch 能通过全部测试的 issue 所占比例。SWE-bench Verified 是一个经过人工标注者验证为可解且无歧义的 500 题精选子集。SWE-bench Lite 则是一个 300 题的子集，专为更快的评估而设计。</p>

<div class="admonition note"><p class="admonition-title">SWE-bench 关键数据（截至 2024 年）</p>
<ul>
<li><p><strong>完整基准：</strong>来自 12 个热门 Python 仓库的 2,294 个任务。</p></li>
<li><p><strong>最佳开源 Agent：</strong>约 43% 解决率（SWE-bench Verified）。</p></li>
<li><p><strong>人类表现：</strong>约 87% 解决率（每个任务 15 分钟）。</p></li>
<li><p><strong>评估成本：</strong>基于 API 的模型每个任务约 $0.25。</p></li>
</ul>
</div>

## 14.6.6　WebArena 方法论 {#webarena-methodology}

<p>WebArena <a class="book-citation" id="cite-zhou2024webarena-1" href="../../references.md#ref-zhou2024webarena" aria-label="参考文献 440">[440]</a> 在沙箱浏览器环境中以贴近真实的网页导航任务评估 Agent。该基准包含跨五个 Web 应用（电商、社交论坛、协作开发、内容管理与地图）的 812 个任务。评估方式如下：</p>

<ul>
<li><p><strong>功能性评估：</strong>通过检查应用状态来验证任务结果（例如“商品是否已加入购物车？”“帖子是否已创建？”）。</p></li>
<li><p><strong>基于 URL 的评估：</strong>对于导航任务，将最终 URL 与期望 URL 进行比较。</p></li>
<li><p><strong>基于程序的评估：</strong>由自定义评估脚本检查复杂条件（例如“价格是否小于 $50？”）。</p></li>
</ul>

<p>主要指标是任务成功率。人类表现约为 78%；当前最先进的 Agent 约为 35–45%。</p>

<p class="table-caption" id="table-14-4"><span class="table-number">表 14.4　</span>Agent 评估基准对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>基准</strong></th><th scope="col"><strong>领域</strong></th><th scope="col"><strong>任务数</strong></th><th scope="col"><strong>评估方式</strong></th><th scope="col"><strong>SOTA（%）</strong></th></tr></thead>
<tbody><tr><td>SWE-bench</td><td>软件工程</td><td>2,294</td><td>测试执行</td><td><span class="arithmatex">\(\sim\)</span>43</td></tr>
<tr><td>SWE-bench Lite</td><td>软件工程</td><td>300</td><td>测试执行</td><td><span class="arithmatex">\(\sim\)</span>50</td></tr>
<tr><td>WebArena</td><td>网页导航</td><td>812</td><td>状态/URL/程序</td><td><span class="arithmatex">\(\sim\)</span>40</td></tr>
<tr><td>ALFWorld <a class="book-citation" id="cite-shridhar2021alfworld-1" href="../../references.md#ref-shridhar2021alfworld" aria-label="参考文献 329">[329]</a></td><td>家居任务</td><td>3,553</td><td>仿真器状态</td><td><span class="arithmatex">\(\sim\)</span>90</td></tr>
<tr><td>AgentBench <a class="book-citation" id="cite-liu2023agentbench-1" href="../../references.md#ref-liu2023agentbench" aria-label="参考文献 229">[229]</a></td><td>多领域</td><td>1,091</td><td>任务特定</td><td><span class="arithmatex">\(\sim\)</span>45</td></tr></tbody>
</table>
</div>
