# 27.6　流式与实时模式 {#subsec:streaming-patterns}

<p>流式（Stream）是 Agent 化 UI 的基础：它把体验从“等待一个结果”转变为“观察 Agent 在工作”。本节涵盖关键的流式模式及其实现考量。</p>

## 27.6.1　Token 流 {#token-streaming}

<p>Token 流在 Token 被生成时增量传输 LLM 输出，而不等待完整响应。常用的两种传输机制：</p>

<ul>
<li><p><strong>服务器推送事件（Server-Sent Events, SSE）</strong>12：从服务器到客户端的单向 HTTP 流。每个事件携带一块 Token。SSE 简单，运行于标准 HTTP/1.1 之上，并由浏览器自动重连。它是 LLM 流式 API 的主导机制（OpenAI、Anthropic、Google 都使用 SSE）。</p></li>
<li><p><strong>WebSockets</strong>：双向持久连接。实现更复杂，但在客户端需要在流中途发送数据的交互式流场景下是必要的（例如打断 Agent、在生成中途提供反馈）。</p></li>
</ul>

<p class="code-caption"><span class="code-number">代码 27.3　</span>基于 FastAPI 的 SSE Token 流</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">fastapi</span><span class="w"> </span><span class="kn">import</span> <span class="n">FastAPI</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">fastapi.responses</span><span class="w"> </span><span class="kn">import</span> <span class="n">StreamingResponse</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">AsyncOpenAI</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">json</span>

<span class="n">app</span> <span class="o">=</span> <span class="n">FastAPI</span><span class="p">()</span>
<span class="n">client</span> <span class="o">=</span> <span class="n">AsyncOpenAI</span><span class="p">()</span>

<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">token_stream</span><span class="p">(</span><span class="n">prompt</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
    <span class="c1"># 产出 SSE 格式 Token 块的生成器。</span>
<span class="w">    </span><span class="sd">"""Generator that yields SSE-formatted token chunks."""</span>
    <span class="n">stream</span> <span class="o">=</span> <span class="k">await</span> <span class="n">client</span><span class="o">.</span><span class="n">chat</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
        <span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">,</span>
        <span class="n">messages</span><span class="o">=</span><span class="p">[{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="n">prompt</span><span class="p">}],</span>
        <span class="n">stream</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
    <span class="p">)</span>
    <span class="k">async</span> <span class="k">for</span> <span class="n">chunk</span> <span class="ow">in</span> <span class="n">stream</span><span class="p">:</span>
        <span class="n">delta</span> <span class="o">=</span> <span class="n">chunk</span><span class="o">.</span><span class="n">choices</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">delta</span>
        <span class="k">if</span> <span class="n">delta</span><span class="o">.</span><span class="n">content</span><span class="p">:</span>
            <span class="c1"># SSE 格式："data: &lt;json&gt;\n\n"</span>
            <span class="c1"># SSE format: "data: &lt;json&gt;\n\n"</span>
            <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'token'</span><span class="p">:</span><span class="w"> </span><span class="n">delta</span><span class="o">.</span><span class="n">content</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>
        <span class="k">elif</span> <span class="n">chunk</span><span class="o">.</span><span class="n">choices</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">finish_reason</span><span class="p">:</span>
            <span class="k">yield</span> <span class="sa">f</span><span class="s2">"data: </span><span class="si">{</span><span class="n">json</span><span class="o">.</span><span class="n">dumps</span><span class="p">({</span><span class="s1">'done'</span><span class="p">:</span><span class="w"> </span><span class="kc">True</span><span class="p">})</span><span class="si">}</span><span class="se">\n\n</span><span class="s2">"</span>

<span class="nd">@app</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"/stream"</span><span class="p">)</span>
<span class="k">async</span> <span class="k">def</span><span class="w"> </span><span class="nf">stream_endpoint</span><span class="p">(</span><span class="n">prompt</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
    <span class="k">return</span> <span class="n">StreamingResponse</span><span class="p">(</span>
        <span class="n">token_stream</span><span class="p">(</span><span class="n">prompt</span><span class="p">),</span>
        <span class="n">media_type</span><span class="o">=</span><span class="s2">"text/event-stream"</span><span class="p">,</span>
        <span class="n">headers</span><span class="o">=</span><span class="p">{</span><span class="s2">"Cache-Control"</span><span class="p">:</span> <span class="s2">"no-cache"</span><span class="p">,</span> <span class="s2">"X-Accel-Buffering"</span><span class="p">:</span> <span class="s2">"no"</span><span class="p">},</span>
    <span class="p">)</span>
</code></pre></div>

## 27.6.2　工具调用流 {#tool-call-streaming}

<p>现代 LLM API 支持流式工具调用：工具名与参数被增量地流式传输，使 UI 能够在工具尚未真正被调用之前就显示“Agent 正在调用 <code>search_web</code>，查询：‘climate change 2024’…”。这需要解析不完整的 JSON，可通过流式 JSON 解析器实现。</p>

<p>工具调用流的模式：</p>

<ul>
<li><p><strong>渐进式参数展示</strong>：在参数流入时即展示，即便调用尚未完成。</p></li>
<li><p><strong>并行工具调用指示器</strong>：当模型同时调用多个工具时，将它们全部标记为待处理，并在结果到达时分别更新。</p></li>
<li><p><strong>工具结果流</strong>：有些工具（例如代码执行、网页抓取）本身就能流式输出结果；应将其渐进式管道传输至 UI。</p></li>
</ul>

## 27.6.3　多 Agent 流 {#multi-agent-streaming}

<p>在多 Agent 系统中，可能有多个 Agent 同时产出输出。UI 必须处理并行流：</p>

<ul>
<li><p><strong>带 Agent 标签的流</strong>：每条流都标记 Agent 身份；UI 将其渲染到不同的泳道或面板中。</p></li>
<li><p><strong>流合并</strong>：对于 supervisor-子 Agent 模式，supervisor 的流可能与子 Agent 流交错；UI 必须维护一致的顺序。</p></li>
<li><p><strong>背压（Backpressure）</strong>：如果 UI 渲染速度跟不上流的到达速度（例如多个 Agent 同时生成），背压机制必须防止缓冲区溢出。策略包括：丢弃中间 Token（仅展示最新）、批量更新或暂停较慢的流。</p></li>
</ul>

## 27.6.4　乐观 UI 更新 {#optimistic-ui-updates}

<p>乐观 UI 更新（optimistic UI updates）通过在服务器确认前立即将用户操作反映到 UI 上，提升感知响应性：</p>

<ul>
<li><p>当用户发送消息时，它会（乐观地）立即出现在聊天历史中，而请求仍在传输中。</p></li>
<li><p>当审批门控被接受时，UI 立即将该操作显示为“已批准”，并开始展示 Agent 的后续步骤，甚至先于服务器处理该批准。</p></li>
<li><p>如果服务器返回错误，乐观更新将被回滚并显示错误状态。</p></li>
</ul>

## 27.6.5　背压处理 {#backpressure-handling}

<p>在高吞吐量的 Agent 化场景下，数据到达速率可能超出 UI 的渲染能力。管理背压的策略：</p>

<ul>
<li><p><strong>Token 批处理</strong>：将 Token 缓冲 50–100ms 后批量渲染而非逐个渲染，降低 DOM 更新频率。</p></li>
<li><p><strong>虚拟滚动</strong>：对于长输出，仅渲染可视区域的内容，丢弃屏幕外的 DOM 节点。</p></li>
<li><p><strong>节流更新</strong>：对于指标与状态展示，无论入流速率如何，都以固定速率（例如 10 Hz）更新。</p></li>
<li><p><strong>渐进式细节</strong>：在高吞吐期间显示摘要视图；完整细节按需提供。</p></li>
</ul>
