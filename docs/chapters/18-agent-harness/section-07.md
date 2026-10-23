# 18.7　错误处理与恢复 {#subsec:error-handling}

<p>Agent 运行在对抗性的、不可预测的环境中。健壮的错误处理是不可妥协的。</p>

## 18.7.1　重试策略 {#retry-strategies}

<ul>
<li><p><strong>指数退避：</strong> 对瞬时失败（速率限制、网络错误），在 <span class="arithmatex">\(\min(2^k \cdot t_0 + \epsilon, t_{\max})\)</span> 秒后重试，其中 <span class="arithmatex">\(k\)</span> 是重试次数，<span class="arithmatex">\(\epsilon\)</span> 是随机抖动。</p></li>
<li><p><strong>Fallback 模型：</strong> 如果主模型不可用或返回错误，则回退到备用模型（能力可能稍弱但可用）。</p></li>
<li><p><strong>优雅降级：</strong> 如果某个工具不可用，告知模型并让其在没有该工具的情况下尝试完成任务。</p></li>
</ul>

<p>第 <span class="arithmatex">\(k\)</span> 次重试的退避延迟为：</p>

<span id="eq:backoff"></span>

<div class="arithmatex">\[t_k = \min\!\left(2^k \cdot t_0 + \mathcal{U}(0, t_0),\; t_{\max}\right), \quad k = 0, 1, 2, \ldots

\tag{18.14}\]</div>

## 18.7.2　循环检测 {#loop-detection}

<p>Agent 可能陷入无限循环——反复以相同参数调用同一工具，或在两个状态之间振荡。检测与自我修正策略 <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>：</p>

<ul>
<li><p><strong>最大迭代守卫：</strong> 对每个任务的步骤数设置硬上限（例如 50 步）。</p></li>
<li><p><strong>动作去重：</strong> 对每个 (tool, args) 对进行哈希；如果同一调用出现 <span class="arithmatex">\(k\)</span> 次，则跳出循环。</p></li>
<li><p><strong>进度检测：</strong> 如果 Agent 状态在 <span class="arithmatex">\(k\)</span> 步内未变化，则触发“卡住”处理器。</p></li>
</ul>

<p>形式化地，当相同的动作哈希在大小为 <span class="arithmatex">\(W\)</span> 的滑动窗口内出现时，即判定为循环：</p>



<div class="arithmatex">\[\text{loop\_detected} \iff \exists\, i &lt; j \leq t: \text{hash}(\text{action}_i) = \text{hash}(\text{action}_j) \;\land\; j - i \leq W
\tag{18.15}\]</div>

## 18.7.3　优雅失败 {#graceful-failure}

<p>当 Agent 无法完成任务时：</p>

<ol>
<li><p>说明已完成的部分（部分结果）。</p></li>
<li><p>说明任务为何无法完成。</p></li>
<li><p>建议恢复动作（例如“请提供您的 API key 以启用网页搜索”）。</p></li>
<li><p>保留状态，以便任务能够被恢复。</p></li>
</ol>

## 18.7.4　可观测性 {#observability}

<div class="admonition note"><p class="admonition-title">Agent 的可观测性三件套</p>
<ul>
<li><p><strong>Trace（追踪）：</strong> 对每次 Agent 运行的端到端追踪，包含每次 LLM 调用、工具调用与状态转移的 span。工具：LangSmith、Arize Phoenix、OpenTelemetry。</p></li>
<li><p><strong>Log（日志）：</strong> 对每个事件（发送 Prompt、接收响应、调用工具、抛出错误）的结构化日志。包含 Token 数、延迟与成本。</p></li>
<li><p><strong>Metric（指标）：</strong> 聚合统计——任务成功率、每任务平均步骤数、工具错误率、每任务成本、p95 延迟。</p></li>
</ul>
</div>

<div class="admonition warning"><p class="admonition-title">调试鸿沟</p>
<p>LLM Agent 出了名地难调试，因为失败通常是<em>语义性</em>的（模型做出了错误决策）而非<em>语法性</em>的（代码异常）。在 replay（回放）工具上进行投入：能够用修改后的 Prompt 或模型重新运行任何过往 Agent 追踪，并并排比较输出。</p>
</div>
