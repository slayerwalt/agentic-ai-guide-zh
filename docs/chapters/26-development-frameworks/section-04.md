# 26.4　开源 Agent 工具链 {#subsec:open-source-tooling}

<p>除主要的商业框架外，围绕 Agent 开发的具体环节也涌现出丰富的开源工具生态。相比全栈框架，这些工具往往提供更高的灵活度和透明度。</p>

<div class="admonition note"><p class="admonition-title">开源 Agent 哲学</p>
<p>开源 Agent 工具优先追求可组合性而非便利性。它们并不强加一套完整架构，而是提供定义良好的构件，由开发者按需自由组装。</p>
</div>

## 26.4.1　模块化 Agent 架构 {#modular-agent-architectures}

<p>模块化方案把 Agent 系统拆解为可独立替换的组件：</p>

<figure id="fig:modular-arch"><a href="../../assets/figures/fig_074_modular-arch.png" target="_blank" rel="noopener" aria-label="查看图 26.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_074_modular-arch.png" alt="模块化 Agent 架构。编排器将任务派发给各核心服务；每个服务自管其存储。虚线表示可选的跨服务通信。" loading="lazy"></a><figcaption><span class="figure-number">图 26.3　</span>模块化 Agent 架构。编排器将任务派发给各核心服务；每个服务自管其存储。虚线表示可选的跨服务通信。</figcaption></figure>

## 26.4.2　关键开源构件 {#key-open-source-building-blocks}

#### Prompt 管理。 {#prompt-management.}

<ul>
<li><p><strong>Promptflow</strong><sup id="note-ref-1"><a href="#note-1" aria-label="注 1">1</a></sup>（Microsoft）：可视化的 Prompt 工程与评测</p></li>
<li><p><strong>Guidance</strong><sup id="note-ref-2"><a href="#note-2" aria-label="注 2">2</a></sup>（Microsoft）：代码与 Prompt 交织的约束生成</p></li>
<li><p><strong>LMQL</strong> <a class="book-citation" id="cite-beurerkellner2023lmql-1" href="../../references.md#ref-beurerkellner2023lmql" aria-label="参考文献 26">[26]</a>：类 SQL 的 LLM Prompt 查询语言，支持约束</p></li>
<li><p><strong>Outlines</strong> <a class="book-citation" id="cite-willard2023outlines-1" href="../../references.md#ref-willard2023outlines" aria-label="参考文献 378">[378]</a>：基于正则与 JSON schema 约束的结构化生成</p></li>
</ul>

#### Tool 注册表。 {#tool-registries.}

<ul>
<li><p><strong>Composio</strong><sup id="note-ref-3"><a href="#note-3" aria-label="注 3">3</a></sup>：250+ 个预置 Tool 集成，自带 OAuth 管理</p></li>
<li><p><strong>Toolhouse</strong><sup id="note-ref-4"><a href="#note-4" aria-label="注 4">4</a></sup>：带沙箱的托管 Tool 执行</p></li>
<li><p><strong>E2B</strong><sup id="note-ref-5"><a href="#note-5" aria-label="注 5">5</a></sup>：面向 Agent 代码运行的代码执行沙箱</p></li>
</ul>

#### 记忆存储。 {#memory-stores.}

<ul>
<li><p><strong>Mem0</strong><sup id="note-ref-6"><a href="#note-6" aria-label="注 6">6</a></sup>：带自动摘要的自适应记忆层</p></li>
<li><p><strong>Zep</strong><sup id="note-ref-7"><a href="#note-7" aria-label="注 7">7</a></sup>：带时间感知（temporal awareness）的长期记忆</p></li>
<li><p><strong>Letta</strong> <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a>（原名 MemGPT）：具备自管理记忆层级的 Agent</p></li>
</ul>

#### 评测框架。 {#evaluation-harnesses.}

<ul>
<li><p><strong>RAGAS</strong><sup id="note-ref-8"><a href="#note-8" aria-label="注 8">8</a></sup>：面向 RAG 的专用评测指标</p></li>
<li><p><strong>DeepEval</strong><sup id="note-ref-9"><a href="#note-9" aria-label="注 9">9</a></sup>：面向 LLM 输出的单元测试框架</p></li>
<li><p><strong>Promptfoo</strong><sup id="note-ref-10"><a href="#note-10" aria-label="注 10">10</a></sup>：基于 CLI 的 Prompt 评测与红队测试</p></li>
<li><p><strong>AgentBench</strong><sup id="note-ref-11"><a href="#note-11" aria-label="注 11">11</a></sup>：面向 Agent 能力的标准化基准</p></li>
</ul>

#### 自托管 Agent 运行时。 {#self-hosted-agent-runtimes.}

<p><strong>OpenClaw</strong><sup id="note-ref-12"><a href="#note-12" aria-label="注 12">12</a></sup> 是一个自托管的网关，通过模块化的 <em>skill</em> 系统把 LLM 接入真实世界的 Tool。与上文的开发框架不同，OpenClaw 强调的是<em>部署</em>层：多渠道集成（Slack、Discord、WhatsApp、Teams）、事件驱动的常驻执行、沙箱化的 Tool 运行，以及面向高影响动作的审批门控。其架构将 <em>tools</em>（低层动作，例如 shell 命令或 API 调用）与 <em>skills</em>（更高层的能力，用规划逻辑来编排 tools）分离，使得在无需重写核心代码的前提下，扩展 Agent 的能力面变得轻而易举。</p>

## 26.4.3　互操作标准 {#interoperability-standards}

<p>Agent 生态正在向若干互操作标准收敛：</p>

<ul>
<li><p><strong>模型上下文协议（Model Context Protocol, MCP）</strong> <a class="book-citation" id="cite-anthropic-mcp-2024-1" href="../../references.md#ref-anthropic-mcp-2024" aria-label="参考文献 9">[9]</a>：Anthropic 提出的 Tool 与资源暴露开放标准，使任何 MCP 兼容 Tool 都能与任何 MCP 兼容 Agent 协作（详见第 <a href="../22-mcp/index.md#sec:mcp">22</a> 章）</p></li>
<li><p><strong>智能体到智能体协议（Agent-to-Agent, A2A）</strong> <a class="book-citation" id="cite-google-a2a-2025-1" href="../../references.md#ref-google-a2a-2025" aria-label="参考文献 113">[113]</a>：Google 提出的 Agent 间通信与任务委派开放标准（详见第 <a href="../24-a2a/index.md#sec:a2a">24</a> 章）</p></li>
<li><p><strong>Tool 的 OpenAPI 化</strong>：用 OpenAPI 规范定义 Tool 接口，实现 Tool 的自动发现与集成（见下文）</p></li>
</ul>

#### OpenAPI 作为 Tool 接口层。 {#openapi-as-a-tool-interface-layer.}

<p>OpenAPI 规范13（前身为 Swagger）以机器可读形式描述 REST API——端点、参数、请求/响应 schema 以及认证要求。Agent 框架越来越多地把 OpenAPI 规范当作<em>零代码 Tool 定义</em>层：不再为每个 API 手写 Tool 包装器，Agent 在运行时解析规范并自动生成可调用 Tool。</p>

<p>转换流水线工作流程如下：</p>

<ol>
<li><p><strong>解析</strong>：读取 OpenAPI 规范（JSON/YAML），并解析 <code>$ref</code> 引用。</p></li>
<li><p><strong>发现</strong>：提取每个操作（<code>GET /pets/{id}</code>、<code>POST /orders</code> 等）。</p></li>
<li><p><strong>生成</strong>：把每个操作转换为函数调用 schema——Tool 名取自 <code>operationId</code>，描述取自 <code>summary</code>，参数取自规范的 <code>parameters</code> 与 <code>requestBody</code> 字段。</p></li>
<li><p><strong>执行</strong>：当 LLM 发出 Tool 调用时，根据其提供的参数构造 HTTP 请求（URL、header、查询参数、body）并发送。</p></li>
<li><p><strong>返回</strong>：将 API 响应回灌到 Agent 的上下文中。</p></li>
</ol>

<p class="code-caption"><span class="code-number">代码 26.15　</span>从 OpenAPI 规范自动生成 Agent Tool</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">openapi_toolset</span><span class="w"> </span><span class="kn">import</span> <span class="n">OpenAPIToolset</span>  <span class="c1"># 例如 google.adk、LangChain 等</span>

<span class="c1"># 加载任意 OpenAPI 3.x 规范——可以是本地文件或远程 URL</span>
<span class="n">spec</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">openapi: "3.0.3"</span>
<span class="s2">info:</span>
<span class="s2">  title: Weather API</span>
<span class="s2">  version: "1.0"</span>
<span class="s2">paths:</span>
<span class="s2">  /forecast:</span>
<span class="s2">    get:</span>
<span class="s2">      operationId: get_forecast</span>
<span class="s2">      summary: Get weather forecast for a location</span>
<span class="s2">      parameters:</span>
<span class="s2">        - name: city</span>
<span class="s2">          in: query</span>
<span class="s2">          required: true</span>
<span class="s2">          schema: {type: string}</span>
<span class="s2">        - name: days</span>
<span class="s2">          in: query</span>
<span class="s2">          schema: {type: integer, default: 3}</span>
<span class="s2">      responses:</span>
<span class="s2">        '200':</span>
<span class="s2">          description: Forecast data</span>
<span class="s2">"""</span>

<span class="c1"># 一行代码：规范 -&gt; 可直接使用的 Tool</span>
<span class="n">toolset</span> <span class="o">=</span> <span class="n">OpenAPIToolset</span><span class="p">(</span><span class="n">spec_str</span><span class="o">=</span><span class="n">spec</span><span class="p">,</span> <span class="n">spec_str_type</span><span class="o">=</span><span class="s2">"yaml"</span><span class="p">)</span>
<span class="n">tools</span> <span class="o">=</span> <span class="n">toolset</span><span class="o">.</span><span class="n">get_tools</span><span class="p">()</span>  <span class="c1"># [RestApiTool("get_forecast", ...)]</span>

<span class="c1"># 接入任意 Agent 框架</span>
<span class="n">agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span><span class="n">model</span><span class="o">=</span><span class="s2">"gpt-4o"</span><span class="p">,</span> <span class="n">tools</span><span class="o">=</span><span class="n">tools</span><span class="p">)</span>
<span class="c1"># LLM 看到的是：function get_forecast(city: str, days: int = 3) -&gt; dict</span>
<span class="c1"># 并可在规划过程中自主调用它</span>
</code></pre></div>

<p>该模式被 Google ADK14、Semantic Kernel（作为「OpenAPI 插件」）、LangChain 的 <code>OpenAPIToolkit</code> 以及 <code>openapi-llm</code>15 等独立库支持。其关键优势在于：任何已有 API 文档的组织都可以让自家 API 在不写额外代码的情况下被 Agent 调用——规范<em>就是</em> Tool 定义。</p>

<div class="footnotes"><hr><ol start="1">
<li id="note-1"><p><a href="https://github.com/microsoft/promptflow">https://github.com/microsoft/promptflow</a></p><a href="#note-ref-1" aria-label="返回注 1">↩</a></li>
<li id="note-2"><p><a href="https://github.com/guidance-ai/guidance">https://github.com/guidance-ai/guidance</a></p><a href="#note-ref-2" aria-label="返回注 2">↩</a></li>
<li id="note-3"><p><a href="https://composio.dev">https://composio.dev</a></p><a href="#note-ref-3" aria-label="返回注 3">↩</a></li>
<li id="note-4"><p><a href="https://toolhouse.ai">https://toolhouse.ai</a></p><a href="#note-ref-4" aria-label="返回注 4">↩</a></li>
<li id="note-5"><p><a href="https://e2b.dev">https://e2b.dev</a></p><a href="#note-ref-5" aria-label="返回注 5">↩</a></li>
<li id="note-6"><p><a href="https://mem0.ai">https://mem0.ai</a></p><a href="#note-ref-6" aria-label="返回注 6">↩</a></li>
<li id="note-7"><p><a href="https://www.getzep.com">https://www.getzep.com</a></p><a href="#note-ref-7" aria-label="返回注 7">↩</a></li>
<li id="note-8"><p><a href="https://github.com/explodinggradients/ragas">https://github.com/explodinggradients/ragas</a></p><a href="#note-ref-8" aria-label="返回注 8">↩</a></li>
<li id="note-9"><p><a href="https://github.com/confident-ai/deepeval">https://github.com/confident-ai/deepeval</a></p><a href="#note-ref-9" aria-label="返回注 9">↩</a></li>
<li id="note-10"><p><a href="https://github.com/promptfoo/promptfoo">https://github.com/promptfoo/promptfoo</a></p><a href="#note-ref-10" aria-label="返回注 10">↩</a></li>
<li id="note-11"><p><a href="https://github.com/THUDM/AgentBench">https://github.com/THUDM/AgentBench</a></p><a href="#note-ref-11" aria-label="返回注 11">↩</a></li>
<li id="note-12"><p><a href="https://github.com/open-claw/open-claw">https://github.com/open-claw/open-claw</a></p><a href="#note-ref-12" aria-label="返回注 12">↩</a></li>
</ol></div>
