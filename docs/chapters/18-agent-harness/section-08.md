# 18.8　扩展与生产环境关注点 {#scaling-and-production-concerns}

## 18.8.1　延迟优化 {#latency-optimization-dup2}

<ul>
<li><p><strong>并行工具调用：</strong> 使用 <code>asyncio</code> 或线程池并发执行独立的工具调用。对 <span class="arithmatex">\(N\)</span> 个并行调用，可将多工具延迟降低 <span class="arithmatex">\(N\times\)</span>。</p></li>
<li><p><strong>流式（Streaming）：</strong> 使用流式 API，在模型响应完成之前就开始处理。降低用户的首 Token 时延。</p></li>
<li><p><strong>Prompt 缓存：</strong> 许多提供商（Anthropic、OpenAI）为重复前缀（例如 system prompt + 工具定义）提供 Prompt 缓存。可将被缓存部分的延迟与成本降低 50–90%。</p></li>
<li><p><strong>推测执行（Speculative execution）：</strong> 在模型完成生成之前就开始执行最可能的下一次工具调用，如果预测错误则取消。</p></li>
</ul>

## 18.8.2　成本管理 {#cost-management}

<ul>
<li><p><strong>Token 预算：</strong> 强制执行每任务和每用户的 Token 预算。接近上限时告警。</p></li>
<li><p><strong>模型路由：</strong> 对简单步骤（工具选择、格式化）使用便宜且快速的模型（例如 GPT-4o-mini、Claude Haiku），仅在复杂推理时使用昂贵的模型（GPT-4o、Claude Opus） <a class="book-citation" id="cite-chen2023frugalgpt-1" href="../../references.md#ref-chen2023frugalgpt" aria-label="参考文献 46">[46]</a>。</p></li>
<li><p><strong>缓存：</strong> 缓存确定性的工具输出（例如数据库查询、静态网页）以避免冗余 API 调用。</p></li>
</ul>

<p>具有 <span class="arithmatex">\(T\)</span> 个 LLM 步骤和 <span class="arithmatex">\(K\)</span> 次工具调用的 Agent 任务的总成本为：</p>



<div class="arithmatex">\[\text{Cost}_{\text{task}} = \sum_{i=1}^{T} \underbrace{p_{\text{in}} \cdot n_{\text{in},i} + p_{\text{out}} \cdot n_{\text{out},i}}_{\text{LLM cost}} + \sum_{j=1}^{K} \underbrace{c_j}_{\text{tool cost}}
\tag{18.16}\]</div>

<p>其中 <span class="arithmatex">\(p_{\text{in}}, p_{\text{out}}\)</span> 是每 Token 价格，<span class="arithmatex">\(n_{\text{in},i}, n_{\text{out},i}\)</span> 是第 <span class="arithmatex">\(i\)</span> 步的输入/输出 Token 数，<span class="arithmatex">\(c_j\)</span> 是第 <span class="arithmatex">\(j\)</span> 次工具调用的成本。</p>

## 18.8.3　速率限制与排队 {#rate-limiting-and-queuing}

<p>当并发运行大量 Agent 时：</p>

<ul>
<li><p><strong>Token 桶速率限制器：</strong> 在共享同一 API key 的所有 Agent 间强制执行每分钟 Token 上限。</p></li>
<li><p><strong>优先级队列：</strong> 高优先级任务（交互式用户请求）抢占低优先级任务（批处理）。</p></li>
<li><p><strong>反压（Backpressure）：</strong> 当队列满时，以 <code>503 Service Unavailable</code> 拒绝新任务，而不是无限制地静默排队。</p></li>
</ul>

## 18.8.4　生产环境评测 {#evaluation-in-production}

<ul>
<li><p><strong>A/B 测试：</strong> 将一部分流量路由到新版本 Agent，比较成功率、成本与延迟。</p></li>
<li><p><strong>灰度（Canary）部署：</strong> 在监控回归的同时，逐步将流量切到新版本。</p></li>
<li><p><strong>Shadow 模式：</strong> 让新 Agent 与生产 Agent 并行运行，比较输出，但只将生产 Agent 的输出提供给用户。</p></li>
<li><p><strong>LLM-as-judge：</strong> 使用另一个 LLM 在有用性、准确性与安全性等维度上评测 Agent 输出 <a class="book-citation" id="cite-zheng2023judging-1" href="../../references.md#ref-zheng2023judging" aria-label="参考文献 433">[433]</a>。</p></li>
</ul>
