# 24.2　Google A2A 协议 {#sec:a2a:google}

<p>2025 年 4 月，Google（在 50 余家技术合作伙伴的共同贡献下）发布了 <strong>智能体到智能体（A2A）协议</strong> <a class="book-citation" id="cite-google-a2a-2025-1" href="../../references.md#ref-google-a2a-2025" aria-label="参考文献 113">[113]</a>，这是一份用于 AI Agent 之间互操作通信的开放规范。该协议随后被捐赠给 <strong>Linux Foundation</strong>，截至 2026 年支持机构已超过 150 家。A2A 围绕一组核心原则进行设计，使其区别于早期的临时方案。</p>

## 24.2.1　设计理念 {#design-philosophy-dup2}

<p>A2A 规范明确提出了五条指导原则（根据官方规范 <a class="book-citation" id="cite-google-a2a-2025-2" href="../../references.md#ref-google-a2a-2025" aria-label="参考文献 113">[113]</a> §1.2 改编）：</p>

<div class="admonition note"><p class="admonition-title">A2A 设计原则</p>
<p>不透明执行（Opaque execution）</p>

<p>调用方 Agent 永远不会探查远程 Agent 的内部——它们只通过声明的接口进行交互。目标究竟是 GPT-4、Gemini 还是基于规则的系统，对协议而言无关紧要，从而真正实现异构 Agent 生态。</p>

<p>企业级就绪（Enterprise readiness）</p>

<p>认证（OAuth 2.0、API 密钥、JWT）、审计日志和合规要求并非事后补丁——它们从一开始就在协议层面被集成。</p>

<p>模态无关性（Modality agnosticism）</p>

<p>一条消息可以同时包含文本、二进制文件和结构化 JSON 负载，从而无需扩展协议即可支持处理图像、音频、代码或文档的 Agent。</p>

<p>基于现有标准的简洁性（Simplicity via existing standards）</p>

<p>A2A 不发明新的传输层，而是复用 HTTP/HTTPS 配合 JSON-RPC 2.0 消息、用于流式传输的服务器发送事件（Server-Sent Events, SSE），以及作为备选绑定的 gRPC——这些都是每个基础设施团队已经在运行的技术。</p>

<p>异步优先的任务模型（Async-first task model）</p>

<p>长时间运行的操作是常态而非例外。推送通知与轮询都是一等机制，因此调用方无需保持连接长达数小时。</p>
</div>

## 24.2.2　Agent Card {#agent-cards}

<p>A2A 可发现性的基础是 <strong>Agent Card</strong>——一份托管在固定端点（<code>/.well-known/agent.json</code>）上的机器可读 JSON 清单。它声明该 Agent 能做什么、如何认证、把任务发送到何处——类似于 OpenAPI 规范，但面向的是自主 Agent 而非 REST 端点。</p>

<div class="admonition example"><p class="admonition-title">Agent Card 结构</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 托管于 https://agent.example.com/.well-known/agent.json 的 Agent Card</span>
<span class="n">agent_card</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"DataAnalysisAgent"</span><span class="p">,</span>
    <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Analyzes structured datasets, produces statistical summaries, "</span>
                   <span class="s2">"generates visualizations, and answers data questions."</span><span class="p">,</span>
    <span class="s2">"url"</span><span class="p">:</span> <span class="s2">"https://agent.example.com/a2a"</span><span class="p">,</span>
    <span class="s2">"version"</span><span class="p">:</span> <span class="s2">"1.2.0"</span><span class="p">,</span>
    <span class="s2">"capabilities"</span><span class="p">:</span> <span class="p">{</span>
        <span class="s2">"streaming"</span><span class="p">:</span> <span class="kc">True</span><span class="p">,</span>
        <span class="s2">"pushNotifications"</span><span class="p">:</span> <span class="kc">True</span><span class="p">,</span>
        <span class="s2">"stateTransitionHistory"</span><span class="p">:</span> <span class="kc">True</span>
    <span class="p">},</span>
    <span class="s2">"authentication"</span><span class="p">:</span> <span class="p">{</span>
        <span class="s2">"schemes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"Bearer"</span><span class="p">,</span> <span class="s2">"ApiKey"</span><span class="p">]</span>
    <span class="p">},</span>
    <span class="s2">"skills"</span><span class="p">:</span> <span class="p">[</span>
        <span class="p">{</span>
            <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"statistical-analysis"</span><span class="p">,</span>
            <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"Statistical Analysis"</span><span class="p">,</span>
            <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Compute descriptive statistics, run hypothesis tests, "</span>
                           <span class="s2">"fit regression models on tabular data."</span><span class="p">,</span>
            <span class="s2">"tags"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"statistics"</span><span class="p">,</span> <span class="s2">"data"</span><span class="p">,</span> <span class="s2">"analysis"</span><span class="p">,</span> <span class="s2">"regression"</span><span class="p">],</span>
            <span class="s2">"examples"</span><span class="p">:</span> <span class="p">[</span>
                <span class="s2">"What is the correlation between columns A and B?"</span><span class="p">,</span>
                <span class="s2">"Run a t-test comparing these two groups."</span><span class="p">,</span>
                <span class="s2">"Fit a linear regression predicting sales from ad spend."</span>
            <span class="p">],</span>
            <span class="s2">"inputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">,</span> <span class="s2">"data"</span><span class="p">],</span>
            <span class="s2">"outputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">,</span> <span class="s2">"data"</span><span class="p">,</span> <span class="s2">"file"</span><span class="p">]</span>
        <span class="p">},</span>
        <span class="p">{</span>
            <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"visualization"</span><span class="p">,</span>
            <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"Data Visualization"</span><span class="p">,</span>
            <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Generate charts, plots, and dashboards from data."</span><span class="p">,</span>
            <span class="s2">"tags"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"charts"</span><span class="p">,</span> <span class="s2">"plots"</span><span class="p">,</span> <span class="s2">"visualization"</span><span class="p">,</span> <span class="s2">"dashboard"</span><span class="p">],</span>
            <span class="s2">"examples"</span><span class="p">:</span> <span class="p">[</span>
                <span class="s2">"Create a bar chart of monthly revenue."</span><span class="p">,</span>
                <span class="s2">"Plot the distribution of customer ages."</span>
            <span class="p">],</span>
            <span class="s2">"inputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">,</span> <span class="s2">"data"</span><span class="p">],</span>
            <span class="s2">"outputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"file"</span><span class="p">,</span> <span class="s2">"text"</span><span class="p">]</span>
        <span class="p">}</span>
    <span class="p">],</span>
    <span class="s2">"defaultInputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">],</span>
    <span class="s2">"defaultOutputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">]</span>
<span class="p">}</span>
</code></pre></div>
</div>

<p>Agent Card 支持<em>基于能力的路由</em>：编排型 Agent 可以从注册中心获取卡片，将子任务在语义上匹配到最合适的 Agent，并据此分派——整个过程无需任何硬编码的路由逻辑。</p>

## 24.2.3　任务生命周期 {#task-lifecycle}

<p>A2A 把所有工作都建模为 <strong>Task</strong>。一个任务会沿着一个明确定义的状态机推进：</p>

<p><code>submitted</code></p>

<p>客户端已发出任务；服务器已确认接收。</p>

<p><code>working</code></p>

<p>Agent 正在主动处理。客户端可以进行轮询，或等待 SSE 事件。</p>

<p><code>input-required</code></p>

<p>Agent 在继续之前需要来自用户或调用方 Agent 的额外信息（例如澄清问题、缺失的凭据）。</p>

<p><code>completed</code></p>

<p>任务成功完成；结果可在响应中获取。</p>

<p><code>failed</code></p>

<p>发生了不可恢复的错误；错误消息会说明原因。</p>

<p><code>rejected</code></p>

<p>Agent 拒绝了该任务（例如超出其能力范围或未获授权）。该状态于 A2A v1.0 加入。</p>

<p><code>canceled</code></p>

<p>任务被中止，可能由客户端或服务器发起。</p>

## 24.2.4　基于 Server-Sent Events 的流式传输 {#streaming-via-server-sent-events}

<p>对于会产生增量输出的任务（例如正在撰写的长报告、正在生成的代码文件），A2A 使用 <strong>服务器发送事件（Server-Sent Events, SSE）</strong>。客户端打开一个持久 HTTP 连接，接收 JSON 事件流：</p>

<div class="admonition example"><p class="admonition-title">SSE 事件流示例</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 每个 SSE 事件携带一个 TaskStatusUpdateEvent 或 TaskArtifactUpdateEvent</span>
<span class="c1"># 下面是“撰写一份研究报告”任务的事件流示例：</span>

<span class="c1"># 事件 1：状态更新</span>
<span class="n">data</span><span class="p">:</span> <span class="p">{</span>
  <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"task-abc123"</span><span class="p">,</span>
  <span class="s2">"status"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"state"</span><span class="p">:</span> <span class="s2">"working"</span><span class="p">},</span>
  <span class="s2">"final"</span><span class="p">:</span> <span class="n">false</span>
<span class="p">}</span>

<span class="c1"># 事件 2：部分产物（流式文本）</span>
<span class="n">data</span><span class="p">:</span> <span class="p">{</span>
  <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"task-abc123"</span><span class="p">,</span>
  <span class="s2">"artifact"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span> <span class="s2">"text"</span><span class="p">:</span> <span class="s2">"## Introduction</span><span class="se">\n\n</span><span class="s2">Recent advances in..."</span><span class="p">}],</span>
    <span class="s2">"index"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span>
    <span class="s2">"append"</span><span class="p">:</span> <span class="n">false</span><span class="p">,</span>
    <span class="s2">"lastChunk"</span><span class="p">:</span> <span class="n">false</span>
  <span class="p">},</span>
  <span class="s2">"final"</span><span class="p">:</span> <span class="n">false</span>
<span class="p">}</span>

<span class="c1"># 事件 3：追加更多文本</span>
<span class="n">data</span><span class="p">:</span> <span class="p">{</span>
  <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"task-abc123"</span><span class="p">,</span>
  <span class="s2">"artifact"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span> <span class="s2">"text"</span><span class="p">:</span> <span class="s2">" reinforcement learning have shown..."</span><span class="p">}],</span>
    <span class="s2">"index"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span>
    <span class="s2">"append"</span><span class="p">:</span> <span class="n">true</span><span class="p">,</span>   <span class="c1"># 追加到已有产物</span>
    <span class="s2">"lastChunk"</span><span class="p">:</span> <span class="n">false</span>
  <span class="p">},</span>
  <span class="s2">"final"</span><span class="p">:</span> <span class="n">false</span>
<span class="p">}</span>

<span class="c1"># 终止事件：任务完成</span>
<span class="n">data</span><span class="p">:</span> <span class="p">{</span>
  <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"task-abc123"</span><span class="p">,</span>
  <span class="s2">"status"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"state"</span><span class="p">:</span> <span class="s2">"completed"</span><span class="p">},</span>
  <span class="s2">"final"</span><span class="p">:</span> <span class="n">true</span>
<span class="p">}</span>
</code></pre></div>
</div>

## 24.2.5　长时间运行任务的推送通知 {#push-notifications-for-long-running-tasks}

<p>当任务可能持续数分钟乃至数小时，保持一个常开的 SSE 连接并不现实。A2A 支持<strong>推送通知</strong>：客户端注册一个 webhook URL，服务器会随着任务的推进通过 POST 发送状态更新。</p>

<div class="highlight"><pre><span></span><code><span class="c1"># 客户端在提交任务时注册推送通知端点</span>
<span class="n">task_request</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"task-xyz789"</span><span class="p">,</span>
    <span class="s2">"message"</span><span class="p">:</span> <span class="p">{</span>
        <span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
        <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span> <span class="s2">"text"</span><span class="p">:</span> <span class="s2">"Analyze Q3 sales data and produce a report."</span><span class="p">}]</span>
    <span class="p">},</span>
    <span class="s2">"pushNotification"</span><span class="p">:</span> <span class="p">{</span>
        <span class="s2">"url"</span><span class="p">:</span> <span class="s2">"https://my-orchestrator.example.com/webhooks/a2a"</span><span class="p">,</span>
        <span class="s2">"token"</span><span class="p">:</span> <span class="s2">"secret-hmac-token-for-verification"</span><span class="p">,</span>
        <span class="s2">"authentication"</span><span class="p">:</span> <span class="p">{</span>
            <span class="s2">"schemes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"Bearer"</span><span class="p">],</span>
            <span class="s2">"credentials"</span><span class="p">:</span> <span class="s2">"eyJhbGciOiJIUzI1NiJ9..."</span>
        <span class="p">}</span>
    <span class="p">}</span>
<span class="p">}</span>
<span class="c1"># 当任务在不同状态之间转换时，</span>
<span class="c1"># 服务器会将 TaskStatusUpdateEvent 对象 POST 到该 webhook URL。</span>
</code></pre></div>

## 24.2.6　消息格式 {#message-format}

<p>A2A 消息由一个 <strong>role</strong>（<code>user</code> 或 <code>agent</code>）加上一组带类型的 <strong>parts</strong>（文本、文件或结构化数据）组成。完整的消息模式、多模态示例以及上下文传递指南详见第 <a href="section-05.md#sec:a2a:messages">24.5</a> 节。</p>

## 24.2.7　认证与授权 {#authentication-and-authorization}

<p>A2A 支持多种认证方案，这些方案在 Agent Card 中声明，并按每次请求执行：</p>

<ul>
<li><p><strong>Bearer Token（JWT/OAuth 2.0）</strong>：企业部署的标准方案；令牌携带的 scope 限定了调用方 Agent 可以请求的内容。</p></li>
<li><p><strong>API 密钥</strong>：适用于内部或可信环境的更简单方案。</p></li>
<li><p><strong>双向 TLS（Mutual TLS, mTLS）</strong>：用于高安全部署的基于证书的认证。</p></li>
<li><p><strong>OpenID Connect</strong>：联合身份机制，使跨组织的 Agent 通信成为可能。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">强制执行授权 scope</p>
<p>接收任务的 Agent 不仅要验证<em>谁</em>在调用（认证），还要验证<em>它被允许请求什么</em>（授权）。一个 <code>ReportingAgent</code> 可能接受任何已认证 Agent 的只读数据查询，但将写操作限定给持有特定 OAuth scope 的 Agent。未能强制执行这一点会在多 Agent 系统中制造权限提升漏洞。</p>
</div>
