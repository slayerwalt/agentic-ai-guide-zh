# 第 24 章 · 智能体到智能体通信（Agent-to-Agent, A2A） {#sec:a2a}

<p>随着大语言模型从孤立的助手演化为由专业化 Agent 组成的协作网络，<em>Agent 之间如何对话</em>这一问题已变得与单个 Agent 内部如何推理同等重要。本章将介绍那些使多 Agent 系统能够协调、委派并共同解决任何单一 Agent 都无法独立处理的问题的协议、模式与工程实践。</p>

## 24.1　动机：为何 Agent 必须相互通信 {#sec:a2a:motivation}

<div class="admonition tip"><p class="admonition-title">专业化的必然性</p>
<p>单一的通才型 Agent 面临一个根本性矛盾：知识的广度与能力的深度。真实世界的任务——例如法律文档审阅、多步科学研究、企业级软件开发——同时需要两者。Agent 间通信通过让一个<em>由专业化 Agent 组成的网络</em>进行协作来化解这一矛盾：每个 Agent 贡献自己的长处，同时把短板委派出去。</p>
</div>

<p>推动结构化 Agent 间通信需求的力量主要有以下几股：</p>

##### 认知负载与上下文限制。 {#cognitive-load-and-context-limits.}

<p>每个 LLM 都在有限的上下文窗口内运行。复杂的工作流——涉及数百份文档、Tool 调用和推理步骤——很快会超出单一 Agent 的内存承载力。通过将任务分解到多个 Agent，每个 Agent 都在可管理的上下文中工作，而编排型 Agent 只需维护高层状态。</p>

##### 专业化与专家能力。 {#specialization-and-expertise.}

<p>不同的 Agent 可针对特定领域进行微调、Prompt 设计或 Tool 配置：例如可访问编译器和测试运行器的 <code>CodeAgent</code>、可访问判例数据库的 <code>LegalAgent</code>、配备统计库的 <code>DataAgent</code>。将子任务路由给合适的专家可同时提升质量与效率。</p>

##### 并行性与吞吐量。 {#parallelism-and-throughput.}

<p>彼此独立的子任务可以同时分派给多个 Agent。一个研究编排器可能并行地把文献检索扇出到五个专业化 Agent，然后再综合它们的结果——大幅缩短墙钟时间。</p>

##### 故障隔离与韧性。 {#fault-isolation-and-resilience.}

<p>当某个 Agent 失败时，设计良好的多 Agent 系统可以换用另一个 Agent 重试、回退到更简单的方案，或者升级到人工介入——而不会让整个工作流崩溃。</p>

##### 委派与交接。 {#delegation-and-handoff.}

<p>随着上下文转移，长时间运行的任务可能需要在 Agent 之间交接。最初的 <code>PlannerAgent</code> 分解目标，将子任务交给 <code>ExecutorAgents</code>，最后由 <code>ReviewerAgent</code> 校验输出——每个 Agent 只接收它所需要的那部分上下文。</p>

<div class="admonition note"><p class="admonition-title">A2A 通信的核心需求</p>
<ol>
<li><p><strong>可发现性</strong>：Agent 必须能够找到其他 Agent 并理解它们的能力。</p></li>
<li><p><strong>互操作性</strong>：由不同团队或厂商构建的 Agent 必须使用共同的协议进行通信。</p></li>
<li><p><strong>异步性</strong>：长时间运行的任务不得阻塞调用方；结果应通过回调或轮询返回。</p></li>
<li><p><strong>安全性</strong>：Agent 之间必须相互认证，并强制执行授权边界。</p></li>
<li><p><strong>可观测性</strong>：每一次消息交换都必须可追踪，以便调试和审计。</p></li>
</ol>
</div>

## 24.2　Google A2A 协议 {#sec:a2a:google}

<p>2025 年 4 月，Google（在 50 余家技术合作伙伴的共同贡献下）发布了 <strong>智能体到智能体（A2A）协议</strong> <a class="book-citation" id="cite-google-a2a-2025-1" href="../../references.md#ref-google-a2a-2025" aria-label="参考文献 113">[113]</a>，这是一份用于 AI Agent 之间互操作通信的开放规范。该协议随后被捐赠给 <strong>Linux Foundation</strong>，截至 2026 年支持机构已超过 150 家。A2A 围绕一组核心原则进行设计，使其区别于早期的临时方案。</p>

### 24.2.1　设计理念 {#design-philosophy-dup2}

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

### 24.2.2　Agent Card {#agent-cards}

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

### 24.2.3　任务生命周期 {#task-lifecycle}

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

### 24.2.4　基于 Server-Sent Events 的流式传输 {#streaming-via-server-sent-events}

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

### 24.2.5　长时间运行任务的推送通知 {#push-notifications-for-long-running-tasks}

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

### 24.2.6　消息格式 {#message-format}

<p>A2A 消息由一个 <strong>role</strong>（<code>user</code> 或 <code>agent</code>）加上一组带类型的 <strong>parts</strong>（文本、文件或结构化数据）组成。完整的消息模式、多模态示例以及上下文传递指南详见第 <a href="#sec:a2a:messages">24.5</a> 节。</p>

### 24.2.7　认证与授权 {#authentication-and-authorization}

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

## 24.3　通信模式 {#sec:a2a:patterns}

<p>多 Agent 系统会根据任务的性质、延迟要求以及参与 Agent 的数量，采用多种不同的通信模式。</p>

### 24.3.1　请求-响应（Request-Response） {#request-response}

<p>最简单的模式：Agent A 把任务发给 Agent B，并等待完整的响应。适用于结果需要在继续之前获得的、简短且定义明确的子任务。</p>

### 24.3.2　流式传输（Streaming） {#streaming}

<p>Agent A 打开一个 SSE 连接；Agent B 在产生部分结果时以流的形式发送出来。非常适合长文本生成（报告、代码）、实时协作或渐进式 UI 更新。</p>

<div class="admonition example"><p class="admonition-title">流式传输模式的应用场景</p>
<p>编排器请求 <code>WritingAgent</code> 起草一份 10 页的技术文档。编排器无需等待 2 分钟以获得完整文档，而是在每一节写完时即流式接收，从而让 <code>ReviewAgent</code> 在后续章节仍在生成时就开始审阅前面的章节——这种流水线将总延迟降低了 40–60%。</p>
</div>

### 24.3.3　多轮交互（Multi-Turn） {#multi-turn-interaction}

<p>某些任务需要迭代式精化。Agent 进入 <code>input-required</code> 状态，编排器提供澄清信息，然后任务恢复执行。这与人类协作流程相似：起草 <span class="arithmatex">\(\to\)</span> 反馈 <span class="arithmatex">\(\to\)</span> 修改。</p>

<div class="highlight"><pre><span></span><code><span class="c1"># 多轮：编排器处理 input-required 状态</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">run_multiturn_task</span><span class="p">(</span><span class="n">client</span><span class="p">,</span> <span class="n">initial_message</span><span class="p">):</span>
    <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span><span class="n">message</span><span class="o">=</span><span class="n">initial_message</span><span class="p">)</span>

    <span class="k">while</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="ow">not</span> <span class="ow">in</span> <span class="p">(</span><span class="s2">"completed"</span><span class="p">,</span> <span class="s2">"failed"</span><span class="p">,</span> <span class="s2">"canceled"</span><span class="p">):</span>
        <span class="k">if</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="o">==</span> <span class="s2">"input-required"</span><span class="p">:</span>
            <span class="c1"># Agent 需要澄清</span>
            <span class="n">clarification_needed</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">message</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Agent asks: </span><span class="si">{</span><span class="n">clarification_needed</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

            <span class="c1"># 编排器生成或转发澄清回复</span>
            <span class="n">user_reply</span> <span class="o">=</span> <span class="k">await</span> <span class="n">get_clarification</span><span class="p">(</span><span class="n">clarification_needed</span><span class="p">)</span>

            <span class="c1"># 发送回复以继续任务</span>
            <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span>
                <span class="n">task_id</span><span class="o">=</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span>
                <span class="n">message</span><span class="o">=</span><span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
                         <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span> <span class="s2">"text"</span><span class="p">:</span> <span class="n">user_reply</span><span class="p">}]}</span>
            <span class="p">)</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="c1"># 仍在执行 —— 延迟后再轮询</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
            <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">get_task</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">)</span>

    <span class="k">return</span> <span class="n">task</span>
</code></pre></div>

### 24.3.4　广播（Broadcast） {#broadcast}

<p>编排器将同一条消息同时发送给多个 Agent——适用于发布通告、分发共享上下文，或触发并行独立的工作流。</p>

### 24.3.5　发布-订阅（Publish-Subscribe, Pub-Sub） {#publish-subscribe-pub-sub}

<p>Agent 订阅事件通道（例如 <code>new-document-uploaded</code>、<code>model-retrained</code>）。当事件触发时，所有订阅的 Agent 都会收到通知。这种方式解耦了生产者与消费者，并支持响应式、事件驱动的架构。</p>

### 24.3.6　协商（Negotiation） {#negotiation}

<p>两个 Agent 互相交换提案与反提案，以便在计划、资源分配或方案上达成一致。常见于 Agent 具有不同目标或约束的多 Agent 规划系统中。</p>

<div class="admonition example"><p class="admonition-title">协商模式</p>
<p><code>PlannerAgent</code> 提出一个 5 步研究计划。<code>ResourceAgent</code> 回复说第 3 步（运行一次大规模仿真）将超出计算预算。<code>PlannerAgent</code> 反提议一次规模缩小的仿真。<code>ResourceAgent</code> 批准。随后达成一致的计划被分派给执行型 Agent。</p>
</div>

### 24.3.7　基于拍卖的任务分配（Auction-Based Task Allocation） {#auction-based-task-allocation}

<p>编排器宣布带需求的任务；候选 Agent 提交竞标（预计完成时间、置信度、成本）；编排器把任务授予中标方。这使得在一组 Agent 上进行动态的、市场化的负载均衡成为可能。</p>

<p class="table-caption" id="table-24-1"><span class="table-number">表 24.1　</span>A2A 通信模式总览。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>延迟</strong></th><th scope="col"><strong>最佳适用场景</strong></th></tr></thead>
<tbody><tr><td>请求-响应</td><td>低</td><td>简短、定义明确的子任务</td></tr>
<tr><td>流式传输</td><td>低（首 Token）</td><td>长文本生成、实时 UI</td></tr>
<tr><td>多轮</td><td>中</td><td>需要澄清的模糊任务</td></tr>
<tr><td>广播</td><td>低</td><td>共享上下文分发</td></tr>
<tr><td>发布-订阅</td><td>可变</td><td>事件驱动的响应式工作流</td></tr>
<tr><td>协商</td><td>中–高</td><td>资源受限的规划</td></tr>
<tr><td>拍卖</td><td>中</td><td>动态负载均衡</td></tr></tbody>
</table>
</div>

## 24.4　Agent 发现与路由 {#sec:a2a:discovery}

<p>在一个 Agent 能够与另一个 Agent 通信之前，它必须先<em>找到</em>对方。Agent 发现就是定位可以处理特定任务的 Agent 的过程。</p>

### 24.4.1　Agent 注册中心 {#agent-registries}

<p><strong>Agent 注册中心</strong>是一种目录服务，用于索引 Agent Card 并提供搜索与查找 API。常见的部署模型有两种：</p>

<p>中心化注册中心（Centralized Registry）</p>

<p>单一权威注册中心（例如企业服务目录）索引所有 Agent。运维简单，但会形成单点故障，且可能无法扩展到跨组织部署。</p>

<p>联合注册中心（Federated Registry）</p>

<p>由多个注册中心组成，每个对特定领域或组织具有权威性，并通过跨注册中心搜索协议互通。更具韧性和隐私保护，但需要标准化的联合协议。</p>

### 24.4.2　基于能力的路由 {#capability-based-routing}

<p>编排器不会硬编码 Agent 的 URL，而是进行<strong>基于能力的路由</strong>：先向注册中心查询匹配所需技能的 Agent，再从中选择最合适的。</p>

<div class="highlight"><pre><span></span><code><span class="k">class</span><span class="w"> </span><span class="nc">AgentRouter</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""根据能力匹配将任务路由到 Agent。"""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">registry_url</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">registry_url</span> <span class="o">=</span> <span class="n">registry_url</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_cache</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">list</span><span class="p">[</span><span class="n">AgentCard</span><span class="p">]]</span> <span class="o">=</span> <span class="p">{}</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">find_agents</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">required_skill</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                          <span class="n">tags</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="n">AgentCard</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""向注册中心查询具备所需技能的 Agent。"""</span>
        <span class="n">params</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"skill"</span><span class="p">:</span> <span class="n">required_skill</span><span class="p">}</span>
        <span class="k">if</span> <span class="n">tags</span><span class="p">:</span>
            <span class="n">params</span><span class="p">[</span><span class="s2">"tags"</span><span class="p">]</span> <span class="o">=</span> <span class="s2">","</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">tags</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="n">resp</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">registry_url</span><span class="si">}</span><span class="s2">/agents"</span><span class="p">,</span> <span class="n">params</span><span class="o">=</span><span class="n">params</span><span class="p">)</span>
            <span class="k">return</span> <span class="p">[</span><span class="n">AgentCard</span><span class="p">(</span><span class="o">**</span><span class="n">card</span><span class="p">)</span> <span class="k">for</span> <span class="n">card</span> <span class="ow">in</span> <span class="n">resp</span><span class="o">.</span><span class="n">json</span><span class="p">()[</span><span class="s2">"agents"</span><span class="p">]]</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">route</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task_description</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AgentCard</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""在语义上将任务描述匹配到最合适的可用 Agent。"""</span>
        <span class="c1"># 对任务描述进行嵌入</span>
        <span class="n">task_embedding</span> <span class="o">=</span> <span class="k">await</span> <span class="n">embed</span><span class="p">(</span><span class="n">task_description</span><span class="p">)</span>

        <span class="c1"># 获取所有已注册的 Agent</span>
        <span class="n">all_agents</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">find_agents</span><span class="p">(</span><span class="n">required_skill</span><span class="o">=</span><span class="s2">"*"</span><span class="p">)</span>

        <span class="c1"># 按任务与 Agent 描述的余弦相似度对每个 Agent 评分</span>
        <span class="n">scored</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="n">all_agents</span><span class="p">:</span>
            <span class="n">agent_embedding</span> <span class="o">=</span> <span class="k">await</span> <span class="n">embed</span><span class="p">(</span><span class="n">agent</span><span class="o">.</span><span class="n">description</span><span class="p">)</span>
            <span class="n">score</span> <span class="o">=</span> <span class="n">cosine_similarity</span><span class="p">(</span><span class="n">task_embedding</span><span class="p">,</span> <span class="n">agent_embedding</span><span class="p">)</span>
            <span class="n">scored</span><span class="o">.</span><span class="n">append</span><span class="p">((</span><span class="n">score</span><span class="p">,</span> <span class="n">agent</span><span class="p">))</span>

        <span class="c1"># 返回得分最高的 Agent</span>
        <span class="n">scored</span><span class="o">.</span><span class="n">sort</span><span class="p">(</span><span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="n">x</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span> <span class="n">reverse</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">scored</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="mi">1</span><span class="p">]</span>
</code></pre></div>

### 24.4.3　等价 Agent 间的负载均衡 {#load-balancing-across-equivalent-agents}

<p>当多个 Agent 提供相同能力时，路由器必须分配负载。常用策略包括：</p>

<ul>
<li><p><strong>轮询（Round-robin）</strong>：在所有可用 Agent 之间平均分发任务。</p></li>
<li><p><strong>最少负载（Least-loaded）</strong>：路由到活跃任务最少的 Agent（需要健康/指标端点）。</p></li>
<li><p><strong>延迟感知（Latency-aware）</strong>：路由到近期响应时间最低的 Agent。</p></li>
<li><p><strong>亲和性路由（Affinity-based）</strong>：将相关任务路由到同一个 Agent，以利用已缓存的上下文。</p></li>
</ul>

### 24.4.4　版本管理与兼容性 {#version-management-and-compatibility}

<p>Agent Card 包含 <code>version</code> 字段。编排器应指定最低版本要求，并在只有旧版本可用时进行优雅降级。推荐使用语义化版本（Semantic Versioning） <a class="book-citation" id="cite-preston2024semver-1" href="../../references.md#ref-preston2024semver" aria-label="参考文献 293">[293]</a>（<code>MAJOR.MINOR.PATCH</code>）：破坏性接口变更递增 <code>MAJOR</code>，新增能力递增 <code>MINOR</code>。</p>

<div class="admonition warning"><p class="admonition-title">长生命周期系统中的版本偏斜</p>
<p>在生产环境的多 Agent 系统中，不同 Agent 可能在不同时间被升级，从而产生版本偏斜（Version Skew）。一个针对 Agent Card v2.1 编译的编排器，可能遇到仍运行 v1.3 的 Agent。务必实现向后兼容的消息处理，并明确测试跨版本场景。</p>
</div>

## 24.5　消息格式与 Schema {#sec:a2a:messages}

### 24.5.1　结构化与非结构化消息 {#structured-vs.-unstructured-messages}

<p>A2A 支持从完全非结构化（纯文本）到完全结构化（带类型的 JSON Schema）的整个谱系。选择何种格式取决于参与的 Agent：</p>

<p class="table-caption" id="table-24-2"><span class="table-number">表 24.2　</span>结构化与非结构化 A2A 消息的权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>消息类型</strong></th><th scope="col"><strong>优点</strong></th><th scope="col"><strong>缺点</strong></th></tr></thead>
<tbody><tr><td>纯文本</td><td>灵活、可读性强、易于生成</td><td>难以可靠解析，无 Schema 校验</td></tr>
<tr><td>结构化 JSON</td><td>机器可解析、可校验、带类型</td><td>需要事先约定 Schema，灵活性较差</td></tr>
<tr><td>混合（文本 + 数据）</td><td>人类可读的意图 + 机器可解析的负载</td><td>构造和解析更复杂</td></tr></tbody>
</table>
</div>

### 24.5.2　多模态消息 {#multi-modal-messages}

<p>A2A 消息的结构是一个 <strong>role</strong>（<code>user</code> 或 <code>agent</code>）外加一组带类型的 <strong>parts</strong>：</p>

<p class="table-caption" id="table-24-3"><span class="table-number">表 24.3　</span>A2A 消息 part 类型（线上格式使用 <code>&quot;type&quot;: &quot;text&quot;|&quot;file&quot;|&quot;data&quot;</code>）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Part 类型</strong></th><th scope="col"><strong>字段</strong></th><th scope="col"><strong>应用场景</strong></th></tr></thead>
<tbody><tr><td><code>TextPart</code></td><td><code>text: string</code></td><td>自然语言指令、响应</td></tr>
<tr><td><code>FilePart</code></td><td><code>mimeType</code>、<code>uri</code> 或 <code>bytes</code></td><td>文档、图像、音频、代码文件</td></tr>
<tr><td><code>DataPart</code></td><td><code>data: object</code></td><td>结构化 JSON（Tool 结果、Schema）</td></tr></tbody>
</table>
</div>

<p>现代 Agent 越来越多地处理非文本模态。A2A 的 <code>FilePart</code> 支持任意 MIME 类型，从而可以构建丰富的多模态工作流：</p>

<div class="admonition example"><p class="admonition-title">多模态 A2A 消息：数据分析</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 一条同时包含文本指令、数据负载和文件的消息</span>
<span class="n">message</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
    <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[</span>
        <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span>
            <span class="s2">"text"</span><span class="p">:</span> <span class="s2">"Analyze the attached CSV and the schema below. "</span>
                    <span class="s2">"Identify anomalies and produce a summary report."</span>
        <span class="p">},</span>
        <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"file"</span><span class="p">,</span>
            <span class="s2">"mimeType"</span><span class="p">:</span> <span class="s2">"text/csv"</span><span class="p">,</span>
            <span class="s2">"uri"</span><span class="p">:</span> <span class="s2">"https://storage.example.com/data/sales_q3.csv"</span>
        <span class="p">},</span>
        <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"data"</span><span class="p">,</span>
            <span class="s2">"data"</span><span class="p">:</span> <span class="p">{</span>
                <span class="s2">"schema"</span><span class="p">:</span> <span class="p">{</span>
                    <span class="s2">"columns"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"date"</span><span class="p">,</span> <span class="s2">"region"</span><span class="p">,</span> <span class="s2">"product"</span><span class="p">,</span> <span class="s2">"revenue"</span><span class="p">,</span> <span class="s2">"units"</span><span class="p">],</span>
                    <span class="s2">"types"</span><span class="p">:</span>   <span class="p">[</span><span class="s2">"date"</span><span class="p">,</span> <span class="s2">"string"</span><span class="p">,</span> <span class="s2">"string"</span><span class="p">,</span> <span class="s2">"float"</span><span class="p">,</span> <span class="s2">"int"</span><span class="p">]</span>
                <span class="p">},</span>
                <span class="s2">"expectedRowCount"</span><span class="p">:</span> <span class="mi">15000</span><span class="p">,</span>
                <span class="s2">"anomalyThreshold"</span><span class="p">:</span> <span class="mf">3.0</span>  <span class="c1"># z 分数阈值</span>
            <span class="p">}</span>
        <span class="p">}</span>
    <span class="p">]</span>
<span class="p">}</span>
</code></pre></div>
</div>

<div class="admonition example"><p class="admonition-title">多模态 A2A 消息：图像分析</p>
<div class="highlight"><pre><span></span><code><span class="c1"># 多模态消息：文本 + 图像 + 结构化数据</span>
<span class="n">multimodal_message</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
    <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[</span>
        <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"text"</span><span class="p">,</span>
         <span class="s2">"text"</span><span class="p">:</span> <span class="s2">"Describe what is wrong with this chart and suggest fixes."</span><span class="p">},</span>
        <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"file"</span><span class="p">,</span>
         <span class="s2">"mimeType"</span><span class="p">:</span> <span class="s2">"image/png"</span><span class="p">,</span>
         <span class="s2">"bytes"</span><span class="p">:</span> <span class="n">base64</span><span class="o">.</span><span class="n">b64encode</span><span class="p">(</span><span class="n">chart_image_bytes</span><span class="p">)</span><span class="o">.</span><span class="n">decode</span><span class="p">()},</span>
        <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"data"</span><span class="p">,</span>
         <span class="s2">"data"</span><span class="p">:</span> <span class="p">{</span>
             <span class="s2">"chartType"</span><span class="p">:</span> <span class="s2">"bar"</span><span class="p">,</span>
             <span class="s2">"dataSource"</span><span class="p">:</span> <span class="s2">"Q3 Revenue by Region"</span><span class="p">,</span>
             <span class="s2">"knownIssues"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"y-axis does not start at zero"</span><span class="p">,</span>
                             <span class="s2">"missing error bars"</span><span class="p">]</span>
         <span class="p">}}</span>
    <span class="p">]</span>
<span class="p">}</span>
</code></pre></div>
</div>

### 24.5.3　上下文传递：哪些该共享，哪些应保留 {#context-passing-what-to-share-vs.-what-to-keep-private}

<p>多 Agent 系统中一个关键的设计决策是<em>上下文范围控制</em>：要把多少对话历史和内部状态传递给子 Agent。</p>

<div class="admonition note"><p class="admonition-title">上下文范围控制原则</p>
<p>最小化上下文（Minimal Context）</p>

<p>只传递子 Agent 完成其任务所必需的内容。可降低 Token 用量、延迟以及敏感信息泄露的风险。</p>

<p>摘要化上下文（Summarized Context）</p>

<p>不要传递原始对话历史，而是传递结构化摘要：目标、约束、已做出的决策以及相关事实。</p>

<p>私有状态（Private State）</p>

<p>内部推理、中间草稿以及用户的个人身份信息（PII），除非确有需要，通常<em>不应</em>转发给子 Agent。</p>

<p>关联 ID（Correlation IDs）</p>

<p>始终传递 <code>correlationId</code>，以便子 Agent 的行为能够在日志和审计轨迹中追溯到发起的工作流。</p>
</div>

### 24.5.4　会话串联与关联 ID {#conversation-threading-and-correlation-ids}

<p>在复杂工作流中，许多任务可能同时处于进行中。<strong>关联 ID</strong> 用于跨 Agent 串联相关任务：</p>

<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>

<span class="k">class</span><span class="w"> </span><span class="nc">WorkflowContext</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在多 Agent 工作流中携带关联元数据。"""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">workflow_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span> <span class="o">=</span> <span class="n">workflow_id</span> <span class="ow">or</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">span_id</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">parent_span_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">child_context</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="s2">"WorkflowContext"</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""为子任务创建子上下文。"""</span>
        <span class="n">child</span> <span class="o">=</span> <span class="n">WorkflowContext</span><span class="p">(</span><span class="n">workflow_id</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span><span class="p">)</span>
        <span class="n">child</span><span class="o">.</span><span class="n">parent_span_id</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">span_id</span>
        <span class="k">return</span> <span class="n">child</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">to_metadata</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
        <span class="k">return</span> <span class="p">{</span>
            <span class="s2">"x-workflow-id"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span><span class="p">,</span>
            <span class="s2">"x-span-id"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">span_id</span><span class="p">,</span>
            <span class="s2">"x-parent-span-id"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">parent_span_id</span>
        <span class="p">}</span>

<span class="c1"># 使用方式：在每次 A2A 任务提交时附加该元数据</span>
<span class="n">ctx</span> <span class="o">=</span> <span class="n">WorkflowContext</span><span class="p">()</span>
<span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span>
    <span class="n">message</span><span class="o">=</span><span class="n">message</span><span class="p">,</span>
    <span class="n">metadata</span><span class="o">=</span><span class="n">ctx</span><span class="o">.</span><span class="n">to_metadata</span><span class="p">()</span>
<span class="p">)</span>
<span class="c1"># 子任务使用子上下文以便追踪</span>
<span class="n">sub_ctx</span> <span class="o">=</span> <span class="n">ctx</span><span class="o">.</span><span class="n">child_context</span><span class="p">()</span>
</code></pre></div>

## 24.6　协调协议 {#sec:a2a:coordination}

<p>除了点对点通信，多 Agent 系统还能从更高层次的<strong>协调协议</strong>中受益——这些是支持集体决策与问题求解的结构化交互模式。</p>

### 24.6.1　合同网协议（Contract Net Protocol） {#contract-net-protocol}

<p><strong>合同网协议（Contract Net Protocol, CNP）</strong> <a class="book-citation" id="cite-smith1980contract-1" href="../../references.md#ref-smith1980contract" aria-label="参考文献 334">[334]</a> 是一种经典的多 Agent 协调机制，可被改造用于基于 LLM 的系统：</p>

<ol>
<li><p><strong>宣告（Announcement）</strong>：管理者 Agent 向所有潜在承包者 Agent 广播任务公告，其中包含任务需求和评估标准。</p></li>
<li><p><strong>竞标（Bidding）</strong>：承包者 Agent 根据自身能力评估任务，并提交包含预计完成时间、置信度和资源需求的竞标。</p></li>
<li><p><strong>授标（Award）</strong>：管理者选择中标方（对于并行子任务则可选择多个中标方），并授予合同。</p></li>
<li><p><strong>执行与汇报（Execution and Reporting）</strong>：承包者执行任务并将结果汇报给管理者。</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">合同网协议实现</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">dataclasses</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ContractNetManager</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""为任务分配实现合同网协议（Contract Net Protocol）。"""</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">allocate_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">:</span> <span class="n">Task</span><span class="p">,</span>
                            <span class="n">candidate_agents</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">AgentCard</span><span class="p">])</span> <span class="o">-&gt;</span> <span class="n">AgentCard</span><span class="p">:</span>
        <span class="c1"># 阶段 1：向所有候选 Agent 宣告任务</span>
        <span class="n">announcement</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"task-announcement"</span><span class="p">,</span>
            <span class="s2">"task"</span><span class="p">:</span> <span class="n">dataclasses</span><span class="o">.</span><span class="n">asdict</span><span class="p">(</span><span class="n">task</span><span class="p">),</span>
            <span class="s2">"deadline"</span><span class="p">:</span> <span class="p">(</span><span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">(</span><span class="n">timezone</span><span class="o">.</span><span class="n">utc</span><span class="p">)</span> <span class="o">+</span> <span class="n">timedelta</span><span class="p">(</span><span class="n">seconds</span><span class="o">=</span><span class="mi">10</span><span class="p">))</span><span class="o">.</span><span class="n">isoformat</span><span class="p">(),</span>
            <span class="s2">"evaluationCriteria"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"confidence"</span><span class="p">,</span> <span class="s2">"estimatedTime"</span><span class="p">,</span> <span class="s2">"cost"</span><span class="p">]</span>
        <span class="p">}</span>

        <span class="c1"># 阶段 2：收集竞标</span>
        <span class="n">bids</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_request_bid</span><span class="p">(</span><span class="n">agent</span><span class="p">,</span> <span class="n">announcement</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="n">candidate_agents</span>
        <span class="p">],</span> <span class="n">return_exceptions</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

        <span class="n">valid_bids</span> <span class="o">=</span> <span class="p">[(</span><span class="n">agent</span><span class="p">,</span> <span class="n">bid</span><span class="p">)</span> <span class="k">for</span> <span class="n">agent</span><span class="p">,</span> <span class="n">bid</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">candidate_agents</span><span class="p">,</span> <span class="n">bids</span><span class="p">)</span>
                      <span class="k">if</span> <span class="ow">not</span> <span class="nb">isinstance</span><span class="p">(</span><span class="n">bid</span><span class="p">,</span> <span class="ne">Exception</span><span class="p">)</span> <span class="ow">and</span> <span class="n">bid</span> <span class="ow">is</span> <span class="ow">not</span> <span class="kc">None</span><span class="p">]</span>

        <span class="k">if</span> <span class="ow">not</span> <span class="n">valid_bids</span><span class="p">:</span>
            <span class="k">raise</span> <span class="ne">RuntimeError</span><span class="p">(</span><span class="sa">f</span><span class="s2">"No agents bid on task </span><span class="si">{</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

        <span class="c1"># 阶段 3：授予最佳竞标方（置信度最高、用时最少）</span>
        <span class="k">def</span><span class="w"> </span><span class="nf">score_bid</span><span class="p">(</span><span class="n">agent_bid</span><span class="p">):</span>
            <span class="n">_</span><span class="p">,</span> <span class="n">bid</span> <span class="o">=</span> <span class="n">agent_bid</span>
            <span class="k">return</span> <span class="n">bid</span><span class="p">[</span><span class="s2">"confidence"</span><span class="p">]</span> <span class="o">-</span> <span class="mf">0.1</span> <span class="o">*</span> <span class="n">bid</span><span class="p">[</span><span class="s2">"estimatedSeconds"</span><span class="p">]</span>

        <span class="n">winner_agent</span><span class="p">,</span> <span class="n">winning_bid</span> <span class="o">=</span> <span class="nb">max</span><span class="p">(</span><span class="n">valid_bids</span><span class="p">,</span> <span class="n">key</span><span class="o">=</span><span class="n">score_bid</span><span class="p">)</span>

        <span class="c1"># 通知中标方与落标方</span>
        <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_award_contract</span><span class="p">(</span><span class="n">winner_agent</span><span class="p">,</span> <span class="n">task</span><span class="p">)</span>
        <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_reject_bid</span><span class="p">(</span><span class="n">agent</span><span class="p">,</span> <span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">agent</span><span class="p">,</span> <span class="n">_</span> <span class="ow">in</span> <span class="n">valid_bids</span> <span class="k">if</span> <span class="n">agent</span> <span class="o">!=</span> <span class="n">winner_agent</span>
        <span class="p">])</span>

        <span class="k">return</span> <span class="n">winner_agent</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_request_bid</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">agent</span><span class="p">:</span> <span class="n">AgentCard</span><span class="p">,</span>
                           <span class="n">announcement</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""请求某个 Agent 对任务进行竞标。"""</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">result</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span>
                <span class="n">agent_url</span><span class="o">=</span><span class="n">agent</span><span class="o">.</span><span class="n">url</span><span class="p">,</span>
                <span class="n">message</span><span class="o">=</span><span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
                         <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"data"</span><span class="p">,</span> <span class="s2">"data"</span><span class="p">:</span> <span class="n">announcement</span><span class="p">}]}</span>
            <span class="p">)</span>
            <span class="k">return</span> <span class="n">result</span><span class="o">.</span><span class="n">artifacts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">parts</span><span class="p">[</span><span class="mi">0</span><span class="p">][</span><span class="s2">"data"</span><span class="p">]</span>
        <span class="k">except</span> <span class="ne">Exception</span><span class="p">:</span>
            <span class="k">return</span> <span class="kc">None</span>
</code></pre></div>
</div>

### 24.6.2　黑板系统（Blackboard Systems） {#blackboard-systems}

<p><strong>黑板系统</strong> <a class="book-citation" id="cite-hayes1985blackboard-1" href="../../references.md#ref-hayes1985blackboard" aria-label="参考文献 135">[135]</a>提供一个共享工作区（即“黑板”），Agent 在上面发布部分解、观察和假设。其他 Agent 监视黑板，在能够提供价值时进行贡献——这是一种<em>机会主义</em>的问题求解方式。</p>

<p>黑板系统非常适合那些求解路径事先未知、不同 Agent 在不同阶段做出贡献的问题——例如科学假设生成、复杂调试或多源情报分析。</p>

### 24.6.3　共识协议（Consensus Protocols） {#consensus-protocols}

<p>当多个 Agent 必须就一项决策达成一致（例如执行哪个计划、结果是否正确），<strong>共识协议</strong>提供了结构化的投票机制：</p>

<p>简单多数投票（Simple Majority Voting）</p>

<p>每个 Agent 投票；获得 <span class="arithmatex">\(&gt; 50\%\)</span> 选票的选项胜出。速度快，但当 Agent 共享同一基础模型时，容易受到相关性错误的影响。</p>

<p>加权投票（Weighted Voting）</p>

<p>按 Agent 的置信度或历史准确率对选票加权。更稳健，但需要经过良好校准的置信度估计。</p>

<p>基于法定人数（Quorum-Based）</p>

<p>一项决策需要 <span class="arithmatex">\(n\)</span> 个 Agent 中至少 <span class="arithmatex">\(k\)</span> 个达成一致。提供容错能力：最多可有 <span class="arithmatex">\(n-k\)</span> 个 Agent 失败或不同意而不会阻塞流程。</p>

<p>德尔菲方法（Delphi Method）</p>

<p>Agent 先投票，看到匿名化的结果后修改投票，并反复迭代直至收敛。可减少锚定偏差，鼓励真正的深入审议。</p>

<div class="highlight"><pre><span></span><code><span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">quorum_vote</span><span class="p">(</span><span class="n">agents</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">AgentCard</span><span class="p">],</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                      <span class="n">options</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">],</span> <span class="n">quorum</span><span class="p">:</span> <span class="nb">int</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""在 Agent 之间进行法定人数投票。返回获胜选项或 None。"""</span>
    <span class="n">votes</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
        <span class="n">ask_agent_to_vote</span><span class="p">(</span><span class="n">agent</span><span class="p">,</span> <span class="n">question</span><span class="p">,</span> <span class="n">options</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">agent</span> <span class="ow">in</span> <span class="n">agents</span>
    <span class="p">])</span>

    <span class="n">counts</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">int</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
    <span class="k">for</span> <span class="n">vote</span> <span class="ow">in</span> <span class="n">votes</span><span class="p">:</span>
        <span class="k">if</span> <span class="n">vote</span> <span class="ow">in</span> <span class="n">options</span><span class="p">:</span>
            <span class="n">counts</span><span class="p">[</span><span class="n">vote</span><span class="p">]</span> <span class="o">=</span> <span class="n">counts</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="n">vote</span><span class="p">,</span> <span class="mi">0</span><span class="p">)</span> <span class="o">+</span> <span class="mi">1</span>

    <span class="c1"># 返回第一个达到法定人数的选项</span>
    <span class="k">for</span> <span class="n">option</span><span class="p">,</span> <span class="n">count</span> <span class="ow">in</span> <span class="nb">sorted</span><span class="p">(</span><span class="n">counts</span><span class="o">.</span><span class="n">items</span><span class="p">(),</span> <span class="n">key</span><span class="o">=</span><span class="k">lambda</span> <span class="n">x</span><span class="p">:</span> <span class="o">-</span><span class="n">x</span><span class="p">[</span><span class="mi">1</span><span class="p">]):</span>
        <span class="k">if</span> <span class="n">count</span> <span class="o">&gt;=</span> <span class="n">quorum</span><span class="p">:</span>
            <span class="k">return</span> <span class="n">option</span>
    <span class="k">return</span> <span class="kc">None</span>  <span class="c1"># 未达到法定人数</span>
</code></pre></div>

### 24.6.4　领导者选举（Leader Election） {#leader-election}

<p>在动态的多 Agent 系统中，可能需要在运行时选举一位<strong>领导者</strong>（编排器）——例如原编排器失败时，或当 Agent 在没有预先指派协调者的情况下自组织时。经典的分布式系统算法（Bully、Ring）可被改造用于 Agent 网络：Agent 间交换能力分数或优先级令牌，以选举出当前可用 Agent 中能力最强的作为领导者。</p>

## 24.7　A2A 与 MCP：互补的协议 {#sec:a2a:vs_mcp}

<p>A2A 与 <strong>模型上下文协议（Model Context Protocol, MCP）</strong> <a class="book-citation" id="cite-anthropic-mcp-2024-1" href="../../references.md#ref-anthropic-mcp-2024" aria-label="参考文献 9">[9]</a> 之间的关系常被误解。这两种协议是<em>互补</em>的，并非竞争关系：</p>

<div class="admonition note"><p class="admonition-title">核心区别</p>
<ul>
<li><p><strong>MCP</strong> 是<em>纵向</em>协议：它把一个 Agent 向下延伸到数据库、API、文件系统和代码执行器的世界。只有 Agent 在推理；MCP 端点是确定性服务。</p></li>
<li><p><strong>A2A</strong> 是<em>横向</em>协议：它把一个具备推理能力的 Agent 与另一个相连。两端都是能够推理、规划和使用 Tool 的智能体。</p></li>
</ul>
</div>

<span id="table-24-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>MCP</strong></th><th scope="col"><strong>A2A</strong></th></tr></thead>
<tbody><tr><td><strong>参与方</strong></td><td>Agent <span class="arithmatex">\(\leftrightarrow\)</span> Tool/资源</td><td>Agent <span class="arithmatex">\(\leftrightarrow\)</span> Agent</td></tr>
<tr><td><strong>智能性</strong></td><td>仅一端（Agent）具备智能</td><td>双方均具备智能</td></tr>
<tr><td><strong>状态性</strong></td><td>通常为无状态 Tool 调用</td><td>带生命周期的有状态任务</td></tr>
<tr><td><strong>流式传输</strong></td><td>有限（Tool 结果）</td><td>一等的 SSE 流式传输</td></tr>
<tr><td><strong>发现机制</strong></td><td>Tool 清单</td><td>Agent Card</td></tr>
<tr><td><strong>认证模型</strong></td><td>服务端控制</td><td>双向认证、OAuth 2.0</td></tr>
<tr><td><strong>典型延迟</strong></td><td>毫秒级</td><td>秒到分钟级</td></tr>
<tr><td><strong>典型用例</strong></td><td>“搜索网络”、“运行 SQL”</td><td>“委派给专家 Agent”</td></tr></tbody>
</table>
</div>

### 24.7.1　何时使用哪一种 {#when-to-use-which}

<ul>
<li><p>当远程端点是一个确定性函数时使用 <strong>MCP</strong>：数据库查询、API 调用、代码执行沙箱。Agent 完全掌控交互过程。</p></li>
<li><p>当远程端点需要对请求进行<em>推理</em>时使用 <strong>A2A</strong>：解释模糊指令、做出判断、使用自有 Tool 或进行多轮对话。</p></li>
<li><p>在同一系统中<strong>同时使用两者</strong>：编排 Agent 通过 A2A 将任务委派给专家 Agent，而每个专家 Agent 通过 MCP 访问自己的 Tool。</p></li>
</ul>

### 24.7.2　组合架构 {#combined-architecture}

<p>在生产环境的多 Agent 系统中，A2A 与 MCP 在不同层次上协同工作：<strong>A2A</strong> 处理 Agent 之间的委派与协调（同级之间的横向通信），而 <strong>MCP</strong> 处理每个 Agent 到其 Tool 与数据源的连接（与能力的纵向集成）。这种关注点分离是构建可扩展 Agent 架构的关键。</p>

<figure id="fig:combined-a2a-mcp"><a href="../../assets/figures/fig_068_combined-a2a-mcp.png" target="_blank" rel="noopener" aria-label="查看图 24.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_068_combined-a2a-mcp.png" alt="A2A 与 MCP 组合架构。编排器通过 A2A 将任务委派给专家 Agent；每个 Agent 则通过 MCP 服务器访问自己的 Tool。" loading="lazy"></a><figcaption><span class="figure-number">图 24.1　</span>A2A 与 MCP 组合架构。编排器通过 A2A 将任务委派给专家 Agent；每个 Agent 则通过 MCP 服务器访问自己的 Tool。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">要点</p>
<ul>
<li><p><strong>用 A2A 进行委派</strong>：当一个 Agent 需要它本身不具备的能力时，会通过 A2A 任务消息委派给另一个 Agent。每个 Agent 都是带有自身 Agent Card 的自包含服务。</p></li>
<li><p><strong>用 MCP 访问 Tool</strong>：每个 Agent 通过 MCP 服务器连接其 Tool。这意味着 Tool 永远不会直接暴露给其他 Agent——只能通过所属 Agent 的接口访问。</p></li>
<li><p><strong>信任边界的分离</strong>：编排器信任专家 Agent（通过 A2A 认证进行核实）。每个专家信任其自有的 MCP 服务器（本地或已认证）。不存在可传递的 Tool 访问。</p></li>
<li><p><strong>独立扩缩容</strong>：代码密集型负载可扩展 CodeAgent 实例；数据型负载可扩展 DataAgent。编排器始终保持轻量。</p></li>
</ul>
</div>

## 24.8　多 Agent 系统中的安全与信任 {#sec:a2a:security}

<p>多 Agent 系统带来了独特的安全挑战。当 Agent A 委派给 Agent B、Agent B 又委派给 Agent C 时，信任链必须被精心管理。</p>

### 24.8.1　Agent 身份验证 {#agent-identity-verification}

<p>每个 Agent 都必须具有可验证的身份。可选方案包括：</p>

<ul>
<li><p>由可信身份提供方签发的 <strong>JWT 令牌</strong> <a class="book-citation" id="cite-rfc7519-1" href="../../references.md#ref-rfc7519" aria-label="参考文献 172">[172]</a>，携带 Agent ID、颁发者和过期时间。接收方 Agent 使用提供方的公钥进行验证。</p></li>
<li><p>由内部 CA 颁发的 <strong>mTLS 证书</strong> <a class="book-citation" id="cite-rfc8705-1" href="../../references.md#ref-rfc8705" aria-label="参考文献 36">[36]</a>，同时提供认证与传输加密。</p></li>
<li><p>适用于不存在单一可信权威的跨组织场景的 <strong>去中心化标识符（Decentralized Identifiers, DIDs）</strong> <a class="book-citation" id="cite-w3c-did-2022-1" href="../../references.md#ref-w3c-did-2022" aria-label="参考文献 61">[61]</a>。</p></li>
</ul>

### 24.8.2　消息完整性与加密 {#message-integrity-and-encryption}

<ul>
<li><p>所有 A2A 通信都应通过 <strong>TLS 1.3</strong> <a class="book-citation" id="cite-rfc8446-1" href="../../references.md#ref-rfc8446" aria-label="参考文献 307">[307]</a> 进行，以防止窃听和中间人攻击。</p></li>
<li><p>对于敏感负载，<strong>端到端加密</strong>（例如 JWE）可以确保中间基础设施（负载均衡器、代理）无法读取消息内容。</p></li>
<li><p><strong>消息签名</strong>（JWS）提供不可否认性：接收方 Agent 可以证明某条特定消息确实来自某位特定发送者。</p></li>
</ul>

### 24.8.3　授权 scope {#authorization-scopes}

<p>并非每个 Agent 都应该能够要求其他任何 Agent 做任何事。OAuth 2.0 授权 scope <a class="book-citation" id="cite-rfc6749-1" href="../../references.md#ref-rfc6749" aria-label="参考文献 133">[133]</a> 定义了边界：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># DataAgent 的 OAuth 2.0 scope 示例</span>
<span class="n">SCOPES</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"data:read"</span><span class="p">:</span>        <span class="s2">"Read data from connected databases"</span><span class="p">,</span>
    <span class="s2">"data:write"</span><span class="p">:</span>       <span class="s2">"Write or modify data in connected databases"</span><span class="p">,</span>
    <span class="s2">"data:export"</span><span class="p">:</span>      <span class="s2">"Export data to external systems"</span><span class="p">,</span>
    <span class="s2">"analysis:run"</span><span class="p">:</span>     <span class="s2">"Execute statistical analyses"</span><span class="p">,</span>
    <span class="s2">"analysis:schedule"</span><span class="p">:</span><span class="s2">"Schedule recurring analyses"</span><span class="p">,</span>
    <span class="s2">"admin:config"</span><span class="p">:</span>     <span class="s2">"Modify agent configuration"</span>
<span class="p">}</span>

<span class="c1"># ReportingAgent 可能仅持有：data:read, analysis:run</span>
<span class="c1"># ETL 流水线 Agent 可能持有：data:read, data:write, data:export</span>
<span class="c1"># 只有人类管理员持有：admin:config</span>

<span class="k">class</span><span class="w"> </span><span class="nc">A2AServer</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">verify_authorization</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">required_scope</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""验证调用方 Agent 是否持有所需 scope。"""</span>
        <span class="n">claims</span> <span class="o">=</span> <span class="n">jwt</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">token</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">public_key</span><span class="p">,</span> <span class="n">algorithms</span><span class="o">=</span><span class="p">[</span><span class="s2">"RS256"</span><span class="p">])</span>
        <span class="n">granted_scopes</span> <span class="o">=</span> <span class="n">claims</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"scope"</span><span class="p">,</span> <span class="s2">""</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
        <span class="k">if</span> <span class="n">required_scope</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">granted_scopes</span><span class="p">:</span>
            <span class="k">raise</span> <span class="ne">PermissionError</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Caller lacks required scope '</span><span class="si">{</span><span class="n">required_scope</span><span class="si">}</span><span class="s2">'. "</span>
                <span class="sa">f</span><span class="s2">"Granted: </span><span class="si">{</span><span class="n">granted_scopes</span><span class="si">}</span><span class="s2">"</span>
            <span class="p">)</span>
        <span class="k">return</span> <span class="kc">True</span>
</code></pre></div>

### 24.8.4　审计轨迹与问责 {#audit-trails-and-accountability}

<div class="admonition warning"><p class="admonition-title">问责鸿沟</p>
<p>在一条 Agent 委派链中，<em>谁</em>应对某个行为负责可能会变得模糊。如果 Agent A 让 Agent B 删除一个文件，而 Agent B 照做了，那么应由谁负责？每一次 A2A 交互都必须记录：调用方 Agent 的身份、任务描述、所用授权令牌、时间戳以及结果。这种审计轨迹对于事件响应、合规和调试都至关重要。</p>
</div>

<p>每个 A2A 服务器都应输出结构化的审计日志：</p>

<div class="highlight"><pre><span></span><code><span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">A2AAuditEvent</span><span class="p">:</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="nb">str</span>          <span class="c1"># ISO 8601</span>
    <span class="n">workflow_id</span><span class="p">:</span> <span class="nb">str</span>        <span class="c1"># 顶层工作流的关联 ID</span>
    <span class="n">span_id</span><span class="p">:</span> <span class="nb">str</span>            <span class="c1"># 当前任务的 span</span>
    <span class="n">parent_span_id</span><span class="p">:</span> <span class="nb">str</span>     <span class="c1"># 调用任务的 span（用于委派链）</span>
    <span class="n">caller_agent_id</span><span class="p">:</span> <span class="nb">str</span>    <span class="c1"># 经验证的调用方 Agent 身份</span>
    <span class="n">callee_agent_id</span><span class="p">:</span> <span class="nb">str</span>    <span class="c1"># 当前 Agent 的身份</span>
    <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">skill_invoked</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">authorization_scopes</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>
    <span class="n">outcome</span><span class="p">:</span> <span class="nb">str</span>            <span class="c1"># "completed" | "failed" | "rejected"</span>
    <span class="n">duration_ms</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">error_code</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span>
</code></pre></div>

## 24.9　实现示例：多 Agent 研究工作流 {#sec:a2a:implementation}

<p>下面的示例演示了一个完整的、基于 A2A 的多 Agent 研究工作流：一个 <code>OrchestratorAgent</code> 分解研究问题，将子任务委派给专家 Agent，并综合它们的结果。</p>

<div class="highlight"><pre><span></span><code><span class="sd">"""</span>
<span class="sd">使用 A2A 协议实现的多 Agent 研究工作流。</span>
<span class="sd">演示内容：Agent Card、A2A 客户端/服务器、任务生命周期、</span>
<span class="sd">多轮交互以及 Agent 间交接。</span>
<span class="sd">"""</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">asyncio</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">uuid</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">collections.abc</span><span class="w"> </span><span class="kn">import</span> <span class="n">AsyncIterator</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">datetime</span><span class="w"> </span><span class="kn">import</span> <span class="n">datetime</span><span class="p">,</span> <span class="n">timedelta</span><span class="p">,</span> <span class="n">timezone</span>

<span class="kn">import</span><span class="w"> </span><span class="nn">httpx</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi</span><span class="w"> </span><span class="kn">import</span> <span class="n">FastAPI</span><span class="p">,</span> <span class="n">HTTPException</span><span class="p">,</span> <span class="n">Request</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi.responses</span><span class="w"> </span><span class="kn">import</span> <span class="n">StreamingResponse</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pydantic</span><span class="w"> </span><span class="kn">import</span> <span class="n">BaseModel</span><span class="p">,</span> <span class="n">Field</span>

<span class="c1"># -- 数据模型 ------------------------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Part</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="nb">type</span><span class="p">:</span> <span class="nb">str</span>           <span class="c1"># "text" | "file" | "data"</span>
    <span class="n">text</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">data</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">mimeType</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">uri</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Message</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">role</span><span class="p">:</span> <span class="nb">str</span>           <span class="c1"># "user" | "agent"</span>
    <span class="n">parts</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">Part</span><span class="p">]</span>

<span class="k">class</span><span class="w"> </span><span class="nc">TaskStatus</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">state</span><span class="p">:</span> <span class="nb">str</span>          <span class="c1"># submitted | working | input-required | completed | failed</span>
    <span class="n">message</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="nb">str</span> <span class="o">=</span> <span class="n">Field</span><span class="p">(</span>
        <span class="n">default_factory</span><span class="o">=</span><span class="k">lambda</span><span class="p">:</span> <span class="n">datetime</span><span class="o">.</span><span class="n">now</span><span class="p">(</span><span class="n">timezone</span><span class="o">.</span><span class="n">utc</span><span class="p">)</span><span class="o">.</span><span class="n">isoformat</span><span class="p">()</span>
    <span class="p">)</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Artifact</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="n">parts</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">Part</span><span class="p">]</span>
    <span class="n">index</span><span class="p">:</span> <span class="nb">int</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">append</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">False</span>
    <span class="n">lastChunk</span><span class="p">:</span> <span class="nb">bool</span> <span class="o">=</span> <span class="kc">True</span>

<span class="k">class</span><span class="w"> </span><span class="nc">Task</span><span class="p">(</span><span class="n">BaseModel</span><span class="p">):</span>
    <span class="nb">id</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">status</span><span class="p">:</span> <span class="n">TaskStatus</span>
    <span class="n">messages</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">Message</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">artifacts</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="n">Artifact</span><span class="p">]</span> <span class="o">=</span> <span class="p">[]</span>
    <span class="n">metadata</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">=</span> <span class="p">{}</span>

<span class="c1"># -- A2A 客户端（HTTP/REST 绑定） ---------------------------------------------</span>
<span class="c1"># 注意：A2A v1.0 定义了三种协议绑定：JSON-RPC 2.0、gRPC，</span>
<span class="c1"># 以及 HTTP+JSON/REST。此示例为提高可读性使用 REST 绑定。</span>

<span class="k">class</span><span class="w"> </span><span class="nc">A2AClient</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""用于向兼容 A2A 的 Agent 发送任务的客户端。"""</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">agent_url</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">auth_token</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span> <span class="o">=</span> <span class="n">agent_url</span><span class="o">.</span><span class="n">rstrip</span><span class="p">(</span><span class="s2">"/"</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">headers</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"Authorization"</span><span class="p">:</span> <span class="sa">f</span><span class="s2">"Bearer </span><span class="si">{</span><span class="n">auth_token</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
            <span class="s2">"Content-Type"</span><span class="p">:</span> <span class="s2">"application/json"</span>
        <span class="p">}</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">get_agent_card</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""获取该 Agent 的能力卡片。"""</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="n">resp</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">get</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span><span class="si">}</span><span class="s2">/.well-known/agent.json"</span><span class="p">,</span>
                <span class="n">headers</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">headers</span>
            <span class="p">)</span>
            <span class="n">resp</span><span class="o">.</span><span class="n">raise_for_status</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">resp</span><span class="o">.</span><span class="n">json</span><span class="p">()</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">send_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">message</span><span class="p">:</span> <span class="n">Message</span><span class="p">,</span>
                        <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">,</span>
                        <span class="n">metadata</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Task</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""提交一个任务，返回初始的 Task 对象。"""</span>
        <span class="n">payload</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"id"</span><span class="p">:</span> <span class="n">task_id</span> <span class="ow">or</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">()),</span>
            <span class="s2">"message"</span><span class="p">:</span> <span class="n">message</span><span class="o">.</span><span class="n">model_dump</span><span class="p">(),</span>
            <span class="s2">"metadata"</span><span class="p">:</span> <span class="n">metadata</span> <span class="ow">or</span> <span class="p">{}</span>
        <span class="p">}</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="n">resp</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">post</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span><span class="si">}</span><span class="s2">/tasks/send"</span><span class="p">,</span>
                <span class="n">json</span><span class="o">=</span><span class="n">payload</span><span class="p">,</span>
                <span class="n">headers</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">headers</span><span class="p">,</span>
                <span class="n">timeout</span><span class="o">=</span><span class="mf">30.0</span>
            <span class="p">)</span>
            <span class="n">resp</span><span class="o">.</span><span class="n">raise_for_status</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">Task</span><span class="p">(</span><span class="o">**</span><span class="n">resp</span><span class="o">.</span><span class="n">json</span><span class="p">())</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">stream_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">message</span><span class="p">:</span> <span class="n">Message</span><span class="p">,</span>
                          <span class="n">metadata</span><span class="p">:</span> <span class="nb">dict</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AsyncIterator</span><span class="p">[</span><span class="nb">dict</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""提交任务并以 SSE 事件流的形式接收结果。"""</span>
        <span class="n">payload</span> <span class="o">=</span> <span class="p">{</span>
            <span class="s2">"id"</span><span class="p">:</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">()),</span>
            <span class="s2">"message"</span><span class="p">:</span> <span class="n">message</span><span class="o">.</span><span class="n">model_dump</span><span class="p">(),</span>
            <span class="s2">"metadata"</span><span class="p">:</span> <span class="n">metadata</span> <span class="ow">or</span> <span class="p">{}</span>
        <span class="p">}</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="k">async</span> <span class="k">with</span> <span class="n">client</span><span class="o">.</span><span class="n">stream</span><span class="p">(</span>
                <span class="s2">"POST"</span><span class="p">,</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span><span class="si">}</span><span class="s2">/tasks/sendSubscribe"</span><span class="p">,</span>
                <span class="n">json</span><span class="o">=</span><span class="n">payload</span><span class="p">,</span>
                <span class="n">headers</span><span class="o">=</span><span class="p">{</span><span class="o">**</span><span class="bp">self</span><span class="o">.</span><span class="n">headers</span><span class="p">,</span> <span class="s2">"Accept"</span><span class="p">:</span> <span class="s2">"text/event-stream"</span><span class="p">},</span>
                <span class="n">timeout</span><span class="o">=</span><span class="mf">300.0</span>
            <span class="p">)</span> <span class="k">as</span> <span class="n">response</span><span class="p">:</span>
                <span class="k">async</span> <span class="k">for</span> <span class="n">line</span> <span class="ow">in</span> <span class="n">response</span><span class="o">.</span><span class="n">aiter_lines</span><span class="p">():</span>
                    <span class="k">if</span> <span class="n">line</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"data: "</span><span class="p">):</span>
                        <span class="n">event_data</span> <span class="o">=</span> <span class="n">json</span><span class="o">.</span><span class="n">loads</span><span class="p">(</span><span class="n">line</span><span class="p">[</span><span class="mi">6</span><span class="p">:])</span>
                        <span class="k">yield</span> <span class="n">event_data</span>
                        <span class="k">if</span> <span class="n">event_data</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"final"</span><span class="p">):</span>
                            <span class="k">break</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">get_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Task</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""轮询任务状态。"""</span>
        <span class="k">async</span> <span class="k">with</span> <span class="n">httpx</span><span class="o">.</span><span class="n">AsyncClient</span><span class="p">()</span> <span class="k">as</span> <span class="n">client</span><span class="p">:</span>
            <span class="n">resp</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">get</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">agent_url</span><span class="si">}</span><span class="s2">/tasks/</span><span class="si">{</span><span class="n">task_id</span><span class="si">}</span><span class="s2">"</span><span class="p">,</span>
                <span class="n">headers</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">headers</span>
            <span class="p">)</span>
            <span class="n">resp</span><span class="o">.</span><span class="n">raise_for_status</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">Task</span><span class="p">(</span><span class="o">**</span><span class="n">resp</span><span class="o">.</span><span class="n">json</span><span class="p">())</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">wait_for_completion</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task</span><span class="p">:</span> <span class="n">Task</span><span class="p">,</span>
                                  <span class="n">poll_interval</span><span class="p">:</span> <span class="nb">float</span> <span class="o">=</span> <span class="mf">2.0</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Task</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""轮询直至任务到达终止状态。"""</span>
        <span class="n">terminal_states</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"completed"</span><span class="p">,</span> <span class="s2">"failed"</span><span class="p">,</span> <span class="s2">"canceled"</span><span class="p">}</span>
        <span class="k">while</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">terminal_states</span><span class="p">:</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="n">poll_interval</span><span class="p">)</span>
            <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">get_task</span><span class="p">(</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">task</span>

<span class="c1"># -- A2A 服务器（FastAPI） ----------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ResearchAgent</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    一个专门的研究 Agent，能够检索文献并对给定主题</span>
<span class="sd">    的研究结果进行总结。</span>
<span class="sd">    """</span>

    <span class="n">AGENT_CARD</span> <span class="o">=</span> <span class="p">{</span>
        <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"ResearchAgent"</span><span class="p">,</span>
        <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Searches academic literature and synthesizes research findings."</span><span class="p">,</span>
        <span class="s2">"url"</span><span class="p">:</span> <span class="s2">"https://research-agent.example.com/a2a"</span><span class="p">,</span>
        <span class="s2">"version"</span><span class="p">:</span> <span class="s2">"1.0.0"</span><span class="p">,</span>
        <span class="s2">"capabilities"</span><span class="p">:</span> <span class="p">{</span>
            <span class="s2">"streaming"</span><span class="p">:</span> <span class="kc">True</span><span class="p">,</span>
            <span class="s2">"pushNotifications"</span><span class="p">:</span> <span class="kc">False</span><span class="p">,</span>
            <span class="s2">"stateTransitionHistory"</span><span class="p">:</span> <span class="kc">True</span>
        <span class="p">},</span>
        <span class="s2">"authentication"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"schemes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"Bearer"</span><span class="p">]},</span>
        <span class="s2">"skills"</span><span class="p">:</span> <span class="p">[{</span>
            <span class="s2">"id"</span><span class="p">:</span> <span class="s2">"literature-search"</span><span class="p">,</span>
            <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"Literature Search"</span><span class="p">,</span>
            <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Search and summarize academic papers on a topic."</span><span class="p">,</span>
            <span class="s2">"tags"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"research"</span><span class="p">,</span> <span class="s2">"literature"</span><span class="p">,</span> <span class="s2">"academic"</span><span class="p">,</span> <span class="s2">"papers"</span><span class="p">],</span>
            <span class="s2">"examples"</span><span class="p">:</span> <span class="p">[</span>
                <span class="s2">"Summarize recent papers on transformer attention mechanisms."</span><span class="p">,</span>
                <span class="s2">"What does the literature say about RLHF for code generation?"</span>
            <span class="p">],</span>
            <span class="s2">"inputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">],</span>
            <span class="s2">"outputModes"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"text"</span><span class="p">,</span> <span class="s2">"data"</span><span class="p">]</span>
        <span class="p">}]</span>
    <span class="p">}</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Task</span><span class="p">]</span> <span class="o">=</span> <span class="p">{}</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">app</span> <span class="o">=</span> <span class="n">FastAPI</span><span class="p">(</span><span class="n">title</span><span class="o">=</span><span class="s2">"ResearchAgent A2A Server"</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_register_routes</span><span class="p">()</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_register_routes</span><span class="p">(</span><span class="bp">self</span><span class="p">):</span>
        <span class="nd">@self</span><span class="o">.</span><span class="n">app</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"/.well-known/agent.json"</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">agent_card</span><span class="p">():</span>
            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">AGENT_CARD</span>

        <span class="nd">@self</span><span class="o">.</span><span class="n">app</span><span class="o">.</span><span class="n">post</span><span class="p">(</span><span class="s2">"/tasks/send"</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">send_task</span><span class="p">(</span><span class="n">request</span><span class="p">:</span> <span class="n">Request</span><span class="p">):</span>
            <span class="n">body</span> <span class="o">=</span> <span class="k">await</span> <span class="n">request</span><span class="o">.</span><span class="n">json</span><span class="p">()</span>
            <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_create_and_run_task</span><span class="p">(</span><span class="n">body</span><span class="p">)</span>
            <span class="k">return</span> <span class="n">task</span><span class="o">.</span><span class="n">model_dump</span><span class="p">()</span>

        <span class="nd">@self</span><span class="o">.</span><span class="n">app</span><span class="o">.</span><span class="n">post</span><span class="p">(</span><span class="s2">"/tasks/sendSubscribe"</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">send_subscribe</span><span class="p">(</span><span class="n">request</span><span class="p">:</span> <span class="n">Request</span><span class="p">):</span>
            <span class="n">body</span> <span class="o">=</span> <span class="k">await</span> <span class="n">request</span><span class="o">.</span><span class="n">json</span><span class="p">()</span>
            <span class="k">return</span> <span class="n">StreamingResponse</span><span class="p">(</span>
                <span class="bp">self</span><span class="o">.</span><span class="n">_stream_task</span><span class="p">(</span><span class="n">body</span><span class="p">),</span>
                <span class="n">media_type</span><span class="o">=</span><span class="s2">"text/event-stream"</span>
            <span class="p">)</span>

        <span class="nd">@self</span><span class="o">.</span><span class="n">app</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"/tasks/</span><span class="si">{task_id}</span><span class="s2">"</span><span class="p">)</span>
        <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">get_task</span><span class="p">(</span><span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
            <span class="k">if</span> <span class="n">task_id</span> <span class="ow">not</span> <span class="ow">in</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">:</span>
                <span class="k">raise</span> <span class="n">HTTPException</span><span class="p">(</span><span class="n">status_code</span><span class="o">=</span><span class="mi">404</span><span class="p">,</span> <span class="n">detail</span><span class="o">=</span><span class="s2">"Task not found"</span><span class="p">)</span>
            <span class="k">return</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task_id</span><span class="p">]</span><span class="o">.</span><span class="n">model_dump</span><span class="p">()</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_create_and_run_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">body</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">Task</span><span class="p">:</span>
        <span class="n">task_id</span> <span class="o">=</span> <span class="n">body</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"id"</span><span class="p">,</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">()))</span>
        <span class="n">message</span> <span class="o">=</span> <span class="n">Message</span><span class="p">(</span><span class="o">**</span><span class="n">body</span><span class="p">[</span><span class="s2">"message"</span><span class="p">])</span>

        <span class="n">task</span> <span class="o">=</span> <span class="n">Task</span><span class="p">(</span>
            <span class="nb">id</span><span class="o">=</span><span class="n">task_id</span><span class="p">,</span>
            <span class="n">status</span><span class="o">=</span><span class="n">TaskStatus</span><span class="p">(</span><span class="n">state</span><span class="o">=</span><span class="s2">"submitted"</span><span class="p">),</span>
            <span class="n">messages</span><span class="o">=</span><span class="p">[</span><span class="n">message</span><span class="p">],</span>
            <span class="n">metadata</span><span class="o">=</span><span class="n">body</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"metadata"</span><span class="p">,</span> <span class="p">{})</span>
        <span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task_id</span><span class="p">]</span> <span class="o">=</span> <span class="n">task</span>

        <span class="c1"># 异步执行</span>
        <span class="n">asyncio</span><span class="o">.</span><span class="n">create_task</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_execute_task</span><span class="p">(</span><span class="n">task_id</span><span class="p">))</span>
        <span class="k">return</span> <span class="n">task</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_execute_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="n">task</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task_id</span><span class="p">]</span>
        <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="n">TaskStatus</span><span class="p">(</span><span class="n">state</span><span class="o">=</span><span class="s2">"working"</span><span class="p">)</span>

        <span class="k">try</span><span class="p">:</span>
            <span class="c1"># 从消息中提取研究问题</span>
            <span class="n">question</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">messages</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">parts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">text</span>

            <span class="c1"># 模拟文献检索（实际生产中替换为真实搜索 Tool）</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>  <span class="c1"># 模拟延迟</span>
            <span class="n">findings</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_search_literature</span><span class="p">(</span><span class="n">question</span><span class="p">)</span>

            <span class="c1"># 生成产物</span>
            <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span> <span class="o">=</span> <span class="p">[</span><span class="n">Artifact</span><span class="p">(</span><span class="n">parts</span><span class="o">=</span><span class="p">[</span>
                <span class="n">Part</span><span class="p">(</span><span class="nb">type</span><span class="o">=</span><span class="s2">"text"</span><span class="p">,</span> <span class="n">text</span><span class="o">=</span><span class="n">findings</span><span class="p">[</span><span class="s2">"summary"</span><span class="p">]),</span>
                <span class="n">Part</span><span class="p">(</span><span class="nb">type</span><span class="o">=</span><span class="s2">"data"</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="p">{</span><span class="s2">"papers"</span><span class="p">:</span> <span class="n">findings</span><span class="p">[</span><span class="s2">"papers"</span><span class="p">],</span>
                                        <span class="s2">"query"</span><span class="p">:</span> <span class="n">question</span><span class="p">})</span>
            <span class="p">])]</span>
            <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="n">TaskStatus</span><span class="p">(</span><span class="n">state</span><span class="o">=</span><span class="s2">"completed"</span><span class="p">)</span>

        <span class="k">except</span> <span class="ne">Exception</span> <span class="k">as</span> <span class="n">e</span><span class="p">:</span>
            <span class="n">task</span><span class="o">.</span><span class="n">status</span> <span class="o">=</span> <span class="n">TaskStatus</span><span class="p">(</span><span class="n">state</span><span class="o">=</span><span class="s2">"failed"</span><span class="p">,</span> <span class="n">message</span><span class="o">=</span><span class="nb">str</span><span class="p">(</span><span class="n">e</span><span class="p">))</span>

        <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task_id</span><span class="p">]</span> <span class="o">=</span> <span class="n">task</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_search_literature</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">dict</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""占位实现：生产环境中调用真实的搜索 API。"""</span>
        <span class="k">return</span> <span class="p">{</span>
            <span class="s2">"summary"</span><span class="p">:</span> <span class="sa">f</span><span class="s2">"Based on a search of recent literature regarding "</span>
                       <span class="sa">f</span><span class="s2">"'</span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="s2">', key findings include: ..."</span><span class="p">,</span>
            <span class="s2">"papers"</span><span class="p">:</span> <span class="p">[</span>
                <span class="p">{</span><span class="s2">"title"</span><span class="p">:</span> <span class="s2">"Attention Is All You Need"</span><span class="p">,</span> <span class="s2">"year"</span><span class="p">:</span> <span class="mi">2017</span><span class="p">,</span>
                 <span class="s2">"relevance"</span><span class="p">:</span> <span class="mf">0.95</span><span class="p">},</span>
                <span class="p">{</span><span class="s2">"title"</span><span class="p">:</span> <span class="s2">"RLHF: Training Language Models to Follow Instructions"</span><span class="p">,</span>
                 <span class="s2">"year"</span><span class="p">:</span> <span class="mi">2022</span><span class="p">,</span> <span class="s2">"relevance"</span><span class="p">:</span> <span class="mf">0.88</span><span class="p">}</span>
            <span class="p">]</span>
        <span class="p">}</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">_stream_task</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">body</span><span class="p">:</span> <span class="nb">dict</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">AsyncIterator</span><span class="p">[</span><span class="nb">str</span><span class="p">]:</span>
        <span class="n">task</span> <span class="o">=</span> <span class="k">await</span> <span class="bp">self</span><span class="o">.</span><span class="n">_create_and_run_task</span><span class="p">(</span><span class="n">body</span><span class="p">)</span>

        <span class="c1"># 推送状态更新</span>
        <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'id'</span><span class="p">:</span><span class="w"> </span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span><span class="w"> </span><span class="s1">'status'</span><span class="p">:</span><span class="w"> </span><span class="p">{</span><span class="s1">'state'</span><span class="p">:</span><span class="w"> </span><span class="s1">'submitted'</span><span class="p">},</span><span class="w"> </span><span class="s1">'final'</span><span class="p">:</span><span class="w"> </span><span class="kc">False</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
        <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'id'</span><span class="p">:</span><span class="w"> </span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span><span class="w"> </span><span class="s1">'status'</span><span class="p">:</span><span class="w"> </span><span class="p">{</span><span class="s1">'state'</span><span class="p">:</span><span class="w"> </span><span class="s1">'working'</span><span class="p">},</span><span class="w"> </span><span class="s1">'final'</span><span class="p">:</span><span class="w"> </span><span class="kc">False</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

        <span class="c1"># 等待完成</span>
        <span class="k">while</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="ow">not</span> <span class="ow">in</span> <span class="p">(</span><span class="s2">"completed"</span><span class="p">,</span> <span class="s2">"failed"</span><span class="p">,</span> <span class="s2">"canceled"</span><span class="p">):</span>
            <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">sleep</span><span class="p">(</span><span class="mf">0.5</span><span class="p">)</span>
            <span class="n">task</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">tasks</span><span class="p">[</span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">]</span>

        <span class="c1"># 推送产物</span>
        <span class="k">if</span> <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span><span class="p">:</span>
            <span class="k">for</span> <span class="n">part</span> <span class="ow">in</span> <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">parts</span><span class="p">:</span>
                <span class="n">event</span> <span class="o">=</span> <span class="p">{</span>
                    <span class="s2">"id"</span><span class="p">:</span> <span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span>
                    <span class="s2">"artifact"</span><span class="p">:</span> <span class="p">{</span>
                        <span class="s2">"parts"</span><span class="p">:</span> <span class="p">[</span><span class="n">part</span><span class="o">.</span><span class="n">model_dump</span><span class="p">()],</span>
                        <span class="s2">"index"</span><span class="p">:</span> <span class="mi">0</span><span class="p">,</span>
                        <span class="s2">"append"</span><span class="p">:</span> <span class="kc">False</span><span class="p">,</span>
                        <span class="s2">"lastChunk"</span><span class="p">:</span> <span class="kc">True</span>
                    <span class="p">},</span>
                    <span class="s2">"final"</span><span class="p">:</span> <span class="kc">False</span>
                <span class="p">}</span>
                <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">(</span><span class="n">event</span><span class="p">)</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

        <span class="c1"># 最终状态</span>
        <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'id'</span><span class="p">:</span><span class="w"> </span><span class="n">task</span><span class="o">.</span><span class="n">id</span><span class="p">,</span><span class="w"> </span><span class="s1">'status'</span><span class="p">:</span><span class="w"> </span><span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">model_dump</span><span class="p">(),</span><span class="w"> </span><span class="s1">'final'</span><span class="p">:</span><span class="w"> </span><span class="kc">True</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

<span class="c1"># -- 编排器：多 Agent 工作流 --------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">ResearchOrchestrator</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    编排一个多 Agent 研究工作流：</span>
<span class="sd">    1. 将研究问题分解为子问题</span>
<span class="sd">    2. 将每个子问题分派给 ResearchAgent</span>
<span class="sd">    3. 将结果综合为最终报告</span>
<span class="sd">    """</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">research_agent_url</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">auth_token</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">research_client</span> <span class="o">=</span> <span class="n">A2AClient</span><span class="p">(</span><span class="n">research_agent_url</span><span class="p">,</span> <span class="n">auth_token</span><span class="p">)</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span> <span class="o">=</span> <span class="nb">str</span><span class="p">(</span><span class="n">uuid</span><span class="o">.</span><span class="n">uuid4</span><span class="p">())</span>

    <span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">run</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">research_question</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Starting workflow </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Question: </span><span class="si">{</span><span class="n">research_question</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

        <span class="c1"># 第 1 步：分解为子问题</span>
        <span class="n">sub_questions</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_decompose</span><span class="p">(</span><span class="n">research_question</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Decomposed into </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">sub_questions</span><span class="p">)</span><span class="si">}</span><span class="s2"> sub-questions"</span><span class="p">)</span>

        <span class="c1"># 第 2 步：并行分派子问题</span>
        <span class="n">tasks</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">research_client</span><span class="o">.</span><span class="n">send_task</span><span class="p">(</span>
                <span class="n">message</span><span class="o">=</span><span class="n">Message</span><span class="p">(</span><span class="n">role</span><span class="o">=</span><span class="s2">"user"</span><span class="p">,</span> <span class="n">parts</span><span class="o">=</span><span class="p">[</span><span class="n">Part</span><span class="p">(</span><span class="nb">type</span><span class="o">=</span><span class="s2">"text"</span><span class="p">,</span> <span class="n">text</span><span class="o">=</span><span class="n">q</span><span class="p">)]),</span>
                <span class="n">metadata</span><span class="o">=</span><span class="p">{</span><span class="s2">"workflowId"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">workflow_id</span><span class="p">,</span> <span class="s2">"subQuestion"</span><span class="p">:</span> <span class="n">i</span><span class="p">}</span>
            <span class="p">)</span>
            <span class="k">for</span> <span class="n">i</span><span class="p">,</span> <span class="n">q</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="n">sub_questions</span><span class="p">)</span>
        <span class="p">])</span>

        <span class="c1"># 第 3 步：等待所有任务完成</span>
        <span class="n">completed_tasks</span> <span class="o">=</span> <span class="k">await</span> <span class="n">asyncio</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="o">*</span><span class="p">[</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">research_client</span><span class="o">.</span><span class="n">wait_for_completion</span><span class="p">(</span><span class="n">task</span><span class="p">)</span>
            <span class="k">for</span> <span class="n">task</span> <span class="ow">in</span> <span class="n">tasks</span>
        <span class="p">])</span>

        <span class="c1"># 第 4 步：检查失败情况</span>
        <span class="n">failed</span> <span class="o">=</span> <span class="p">[</span><span class="n">t</span> <span class="k">for</span> <span class="n">t</span> <span class="ow">in</span> <span class="n">completed_tasks</span> <span class="k">if</span> <span class="n">t</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="o">==</span> <span class="s2">"failed"</span><span class="p">]</span>
        <span class="k">if</span> <span class="n">failed</span><span class="p">:</span>
            <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Warning: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">failed</span><span class="p">)</span><span class="si">}</span><span class="s2"> sub-tasks failed"</span><span class="p">)</span>

        <span class="c1"># 第 5 步：综合结果</span>
        <span class="n">findings</span> <span class="o">=</span> <span class="p">[]</span>
        <span class="k">for</span> <span class="n">task</span><span class="p">,</span> <span class="n">question</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">completed_tasks</span><span class="p">,</span> <span class="n">sub_questions</span><span class="p">):</span>
            <span class="k">if</span> <span class="n">task</span><span class="o">.</span><span class="n">status</span><span class="o">.</span><span class="n">state</span> <span class="o">==</span> <span class="s2">"completed"</span> <span class="ow">and</span> <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span><span class="p">:</span>
                <span class="n">summary</span> <span class="o">=</span> <span class="n">task</span><span class="o">.</span><span class="n">artifacts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">parts</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">text</span>
                <span class="n">findings</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="sa">f</span><span class="s2">"### </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="se">\n</span><span class="si">{</span><span class="n">summary</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

        <span class="n">report</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_synthesize</span><span class="p">(</span><span class="n">research_question</span><span class="p">,</span> <span class="n">findings</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"[Orchestrator] Workflow complete. Report: </span><span class="si">{</span><span class="nb">len</span><span class="p">(</span><span class="n">report</span><span class="p">)</span><span class="si">}</span><span class="s2"> chars"</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">report</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_decompose</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""将复杂问题分解为聚焦的子问题。"""</span>
        <span class="c1"># 生产环境中：使用 LLM 进行分解</span>
        <span class="k">return</span> <span class="p">[</span>
            <span class="sa">f</span><span class="s2">"What are the foundational methods for: </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="s2">?"</span><span class="p">,</span>
            <span class="sa">f</span><span class="s2">"What are the most recent advances in: </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="s2">?"</span><span class="p">,</span>
            <span class="sa">f</span><span class="s2">"What are the open challenges and limitations in: </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="s2">?"</span>
        <span class="p">]</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_synthesize</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">question</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">findings</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">])</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""将子结论综合为一份连贯的报告。"""</span>
        <span class="c1"># 生产环境中：使用 LLM 进行综合</span>
        <span class="n">sections</span> <span class="o">=</span> <span class="s2">"</span><span class="se">\n\n</span><span class="s2">"</span><span class="o">.</span><span class="n">join</span><span class="p">(</span><span class="n">findings</span><span class="p">)</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"# Research Report: </span><span class="si">{</span><span class="n">question</span><span class="si">}</span><span class="se">\n\n</span><span class="si">{</span><span class="n">sections</span><span class="si">}</span><span class="s2">"</span>

<span class="c1"># -- 入口 ---------------------------------------------------------------------</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">main</span><span class="p">():</span>
    <span class="n">orchestrator</span> <span class="o">=</span> <span class="n">ResearchOrchestrator</span><span class="p">(</span>
        <span class="n">research_agent_url</span><span class="o">=</span><span class="s2">"https://research-agent.example.com/a2a"</span><span class="p">,</span>
        <span class="n">auth_token</span><span class="o">=</span><span class="s2">"eyJhbGciOiJSUzI1NiJ9..."</span>
    <span class="p">)</span>
    <span class="n">report</span> <span class="o">=</span> <span class="k">await</span> <span class="n">orchestrator</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
        <span class="s2">"Reinforcement learning from human feedback for large language models"</span>
    <span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">report</span><span class="p">)</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="n">asyncio</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">main</span><span class="p">())</span>
</code></pre></div>

## 24.10　小结 {#sec:a2a:summary}

<div class="admonition note"><p class="admonition-title">关键要点：Agent 到 Agent 通信</p>
<ol>
<li><p><strong>A2A 让规模化的专业化成为可能</strong>：通过将任务路由给专家 Agent，多 Agent 系统可同时获得深度与广度。（第 <a href="../25-multi-agent-systems/index.md#sec:multi-agent-systems">25</a> 章将深入介绍多 Agent 架构。）</p></li>
<li><p><strong>Google 的 A2A 协议</strong>为可互操作的 Agent 通信提供了一个面向生产的开放标准，涵盖 Agent Card、任务生命周期管理、SSE 流式传输和企业级认证。</p></li>
<li><p><strong>通信模式</strong>从简单的请求-响应，到复杂的协商和基于拍卖的分配——可根据任务复杂度和延迟需求来选择。</p></li>
<li><p><strong>A2A 与 MCP 互补</strong>：A2A 连接 Agent 与 Agent；MCP 连接 Agent 与 Tool。大多数生产系统两者并用。</p></li>
<li><p><strong>安全不可妥协</strong>：Agent 身份验证、授权 scope 和审计轨迹在任何多 Agent 部署中都不可或缺。</p></li>
<li><p><strong>协调协议</strong>（合同网、黑板、共识）在简单委派之外，提供了用于集体决策的结构化机制。</p></li>
<li><p><strong>基于关联 ID 的可观测性</strong>对于调试和审计跨越多个 Agent 与 Tool 的复杂多 Agent 工作流至关重要。</p></li>
</ol>
</div>

<div class="admonition question"><p class="admonition-title">A2A 中的开放研究问题</p>
<ul>
<li><p>Agent 应如何处理来自层级中多个编排器的<em>相互冲突的指令</em>？哪些冲突解决机制最为有效？</p></li>
<li><p>Agent 能否通过经验<em>学习</em>出更好的路由与委派策略，而不再依赖静态的能力声明？</p></li>
<li><p>如何防范<em>Prompt 注入攻击</em>——即恶意 Agent 通过在消息中嵌入对抗性指令来操纵下游 Agent？</p></li>
<li><p>上下文传递的合理<em>隐私边界</em>在哪里——子 Agent 应该看到多少对话历史，又如何在技术上强制执行这些边界？</p></li>
<li><p>当 Agent 网络扩展到成百上千个 Agent 时，如何在不形成瓶颈或破坏一致性的前提下维持<em>连贯的全局状态</em>？</p></li>
</ul>
</div>
