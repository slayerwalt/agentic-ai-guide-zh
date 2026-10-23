# 21.2　环境设计原则 {#sec:env-design}

<p>一个设计良好的 Agent 环境暴露出四个正交的设计轴：<em>观察空间</em>、<em>动作空间</em>、<em>奖励信号</em>和<em>Episode 结构</em>。让每一项都正确是必要的；让四者同时正确则是环境工程的手艺所在。</p>

## 21.2.1　观察空间设计 {#observation-space-design}

<p>观察是 Agent 在每一步<em>看到</em>的内容。对于基于 LLM 的 Agent，观察几乎总以文本形式呈现，但其来源材料差异极大：</p>

<ul>
<li><p><strong>纯文本</strong>：终端输出、文件内容、API 响应、错误信息。与任意 LLM 兼容度最高，但丢失空间和视觉结构。</p></li>
<li><p><strong>结构化（JSON/XML）</strong>：机器可读的状态表示。可实现精确的指代落地，但要求 Agent 解析结构而非阅读散文。</p></li>
<li><p><strong>多模态</strong>：截图、可访问性树（accessibility tree）、渲染后的 HTML。GUI 和 Web 任务必备；需要具视觉能力的模型或独立的感知模块。</p></li>
<li><p><strong>混合</strong>：截图配合可访问性树（OSWorld 和 VisualWebArena 中使用），同时提供视觉上下文和结构化元素标识符，融合两种模态的优势。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">观察泄露（Observation Leakage）</p>
<p>一个常见的设计错误是在观察中包含 Agent 本不应访问的信息——例如真值答案、奖励值或未来的任务步骤。观察泄露会虚高基准分数，并产出在真实环境（缺乏这些信息）中部署时灾难性失败的 Agent。</p>
</div>

## 21.2.2　动作空间设计 {#action-space-design-dup2}

<p>动作空间定义了 Agent 可以<em>做</em>什么。对 LLM Agent 而言，动作通常是一段文本字符串，由环境解析并执行。常见动作类型包括：</p>

<ul>
<li><p><strong>工具调用</strong>：对外部函数（搜索、计算器、日历）的结构化调用。常以 JSON 或 XML 函数调用语法格式化。</p></li>
<li><p><strong>代码执行</strong>：Agent 写出的代码在沙箱中运行；stdout/stderr 作为下一次观察返回。这是最具表达力的动作类型。</p></li>
<li><p><strong>API 交互</strong>：对 Web 服务的 HTTP 请求、数据库查询、Shell 命令。</p></li>
<li><p><strong>GUI 动作</strong>：<code>click(x,y)</code>、<code>type(&quot;text&quot;)</code>、<code>scroll(direction)</code>、<code>key(&quot;Enter&quot;)</code>。用于计算机使用类环境。</p></li>
<li><p><strong>自然语言</strong>：发送给另一个 Agent、人类或子任务规划器的自由文本。</p></li>
</ul>

## 21.2.3　奖励信号设计 {#reward-signal-design}

<p>奖励设计是环境工程中最难的部分。奖励必须：</p>

<ol>
<li><p><strong>对齐</strong>：高奖励应对应真正的任务完成，而非表面代理指标。</p></li>
<li><p><strong>可学习</strong>：信号必须足够稠密，以使 Agent 能够取得进展；长时程任务上的纯稀疏奖励若无额外塑形（shaping）通常无法学习。</p></li>
<li><p><strong>抗作弊</strong>：Agent 不应能够在未真正完成任务的情况下获得高奖励（reward hacking，奖励欺骗）。</p></li>
</ol>

<p class="table-caption" id="table-21-1"><span class="table-number">表 21.1　</span>Agent 环境中各类奖励信号及其权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>奖励类型</strong></th><th scope="col"><strong>优点</strong></th><th scope="col"><strong>缺点</strong></th></tr></thead>
<tbody><tr><td>稀疏（结束时 0/1）</td><td>对齐性好，难以作弊</td><td>难以学习</td></tr>
<tr><td>稠密（步级）</td><td>易于学习</td><td>易产生塑形伪影</td></tr>
<tr><td>内在（好奇心）</td><td>驱动探索</td><td>可能偏离任务</td></tr>
<tr><td>LLM-as-judge</td><td>灵活、细腻</td><td>昂贵、不一致</td></tr>
<tr><td>基于执行</td><td>真值依据</td><td>仅适用于可验证任务</td></tr></tbody>
</table>
</div>

## 21.2.4　Episode 结构 {#episode-structure}

<p>Episode 可以采用若干结构：</p>

<ul>
<li><p><strong>定长</strong>：Agent 恰好走 <span class="arithmatex">\(T\)</span> 步。实现简单；在已解决的任务上浪费算力。</p></li>
<li><p><strong>提前终止</strong>：当 Agent 标示完成或达到终止状态时 Episode 结束。更高效，但需要可靠的终止检测器。</p></li>
<li><p><strong>开放式</strong>：无固定时程；Agent 持续运行直到资源预算（Token、API 调用次数、墙钟时间）耗尽。最接近真实部署，但最难评估。</p></li>
</ul>

#### 自适应 Episode 长度与提前终止。 {#adaptive-episode-length-and-early-termination.}

<p>近期工作挑战了“Episode 长度必须在训练前固定”的假设：</p>

<ul>
<li><p><strong>时程课程。</strong> AELA <a class="book-citation" id="cite-yoo2025aela-1" href="../../references.md#ref-yoo2025aela" aria-label="参考文献 411">[411]</a> 从短 Episode 开始，并随 Agent 能力的提升（以 Policy 熵的收敛来度量）逐渐扩展时程。前期的短 Episode 每个训练样本能暴露更多样化的初始状态。</p></li>
<li><p><strong>截断作为 RL 惩罚。</strong> DLER <a class="book-citation" id="cite-liu2025dler-1" href="../../references.md#ref-liu2025dler" aria-label="参考文献 235">[235]</a> 表明，对于推理模型，最简单的长度控制——硬截断——只要配合批级别的奖励归一化和动态采样，就能很好地工作，从而避免被截断的 rollout 丢失奖励信号。</p></li>
<li><p><strong>学习停止。</strong> 模型本身可以学习何时停止推理，而非依赖固定预算。<a class="book-citation" id="cite-liu2025answerstop-1" href="../../references.md#ref-liu2025answerstop" aria-label="参考文献 227">[227]</a> 提出三种策略：当连续推理步骤收敛到同一答案时停止；提升“思考结束” Token 的概率；或在隐状态激活上训练一个轻量分类器以预测最优停止点。</p></li>
<li><p><strong>部分 rollout 回收。</strong> APRIL <a class="book-citation" id="cite-april2025-1" href="../../references.md#ref-april2025" aria-label="参考文献 251">[251]</a> 超额发起 rollout 请求，一旦达到目标 Batch 数即终止；未完成的回复被回收作为后续步骤的热启动前缀，从而消除了少数慢样本阻塞整个 Batch 的长尾停顿（吞吐提升 20–35%）。TLT <a class="book-citation" id="cite-hu2025tlt-1" href="../../references.md#ref-hu2025tlt" aria-label="参考文献 149">[149]</a> 针对同一瓶颈，通过即时训练一个自适应草稿模型来对掉队样本进行投机解码（端到端加速 1.7<span class="arithmatex">\(\times\)</span>，无损）。</p></li>
</ul>

## 21.2.5　难度课程与自适应环境 {#difficulty-curriculum-and-adaptive-environments}

<p>静态基准测量 Agent 能力的一个固定快照。自适应环境更进一步：在线监控 Agent 表现并调整任务难度，使 Agent 保持在“最近发展区”——难到足以从中学习，易到偶尔能成功。相关技术包括：</p>

<ul>
<li><p><strong>程序化生成</strong>：从参数化分布中采样任务；根据近期成功率调整难度参数。Prioritized Level Replay <a class="book-citation" id="cite-jiang2021plr-1" href="../../references.md#ref-jiang2021plr" aria-label="参考文献 167">[167]</a> 通过估计的学习潜力（如 GAE 幅值）对每个生成关卡打分，并更频繁地回放高价值关卡。</p></li>
<li><p><strong>自博弈 / 对抗式环境设计</strong>：PAIRED <a class="book-citation" id="cite-dennis2020paired-1" href="../../references.md#ref-dennis2020paired" aria-label="参考文献 74">[74]</a> 训练一个对手提出能最大化“主角”与“反派” Agent 之间<em>悔值（regret）</em>的环境，从而无需手工设计难度时间表即可产生复杂度逐步上升的自然课程。</p></li>
<li><p><strong>后见之明重标注</strong>：用 Agent <em>实际</em>达到的目标重新标注失败轨迹，使失败也能提供学习信号（Hindsight Experience Replay, HER） <a class="book-citation" id="cite-andrychowicz2017hindsight-1" href="../../references.md#ref-andrychowicz2017hindsight" aria-label="参考文献 8">[8]</a>。</p></li>
<li><p><strong>面向 LLM 的难度定向数据筛选</strong>：在 RLVR 训练中，并非所有问题都提供等量信号。近期工作优先选择中等难度的问题——即模型成功率大致在 30–70% 之间的题目——因为它们能提供最高的梯度信息量 <a class="book-citation" id="cite-wang2025dataefficiency-1" href="../../references.md#ref-wang2025dataefficiency" aria-label="参考文献 370">[370]</a>。ADCL <a class="book-citation" id="cite-liu2025adcl-1" href="../../references.md#ref-liu2025adcl" aria-label="参考文献 230">[230]</a> 随着模型提升而周期性地重新估计难度，避免课程过时。</p></li>
</ul>
