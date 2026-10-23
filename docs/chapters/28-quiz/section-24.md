# 28.24　MCP 协议问题 {#mcp-protocol-questions}

<section class="quiz-question" id="q-source-93"><h2 class="question-title">Q：解释 MCP 的 N+M 架构，以及它对 Agent 生态系统为何重要</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>N<span class="arithmatex">\(\times\)</span>M 问题</strong>：没有 MCP 时，<span class="arithmatex">\(N\)</span> 个 Agent 框架必须各自实现与 <span class="arithmatex">\(M\)</span> 个工具的集成 = 总共 <span class="arithmatex">\(N \times M\)</span> 次集成。新增一个工具需要 <span class="arithmatex">\(N\)</span> 次实现。</p>

<p><strong>MCP 的 N+M 方案</strong>：标准化接口。每个 Agent 实现一个 MCP 客户端（共 <span class="arithmatex">\(N\)</span> 个）。每个工具实现一个 MCP 服务器（共 <span class="arithmatex">\(M\)</span> 个）。总集成数 = <span class="arithmatex">\(N + M\)</span>。</p>

<p><strong>具体示例</strong>：5 个 Agent 框架（LangChain/AutoGen/CrewAI/Claude/custom）<span class="arithmatex">\(\times\)</span> 20 个工具（GitHub/Slack/DB/filesystem/…）= 没有 MCP 时 100 次集成。有 MCP 时：5 个客户端 + 20 个服务器 = 25 次实现。</p>

<p><strong>为何重要</strong>：</p>

<ol>
<li><p><strong>工具复用</strong>：构建一次工具服务器；可被任何兼容 MCP 的 Agent 使用</p></li>
<li><p><strong>Agent 可移植性</strong>：从 Claude 切换到自定义 Agent 而无需重写工具集成</p></li>
<li><p><strong>生态成长</strong>：降低新增工具的门槛激励社区构建更多</p></li>
<li><p><strong>可组合性</strong>：运行时将多个服务器动态连接到一个 Agent</p></li>
</ol>

<p><strong>类比</strong>：USB 标准化了外设连接。USB 之前：每种设备都有专有接口。USB 之后：一个口适配所有。MCP 对 Agent–工具连接做了同样的事。</p>

<p><em><strong>复习：</strong>第 <a href="../22-mcp/index.md">22</a> 章（模型上下文协议）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-94"><h2 class="question-title">Q：MCP 的四个核心原语是什么，何时分别使用？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<span id="table-28-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>原语</strong></th><th scope="col"><strong>方向</strong></th><th scope="col"><strong>用途</strong></th><th scope="col"><strong>示例</strong></th></tr></thead>
<tbody><tr><td>Tools</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务器</td><td>执行动作</td><td><code>create_issue</code>；<code>query_db</code></td></tr>
<tr><td>Resources</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务器</td><td>读取数据</td><td>文件内容；数据库记录</td></tr>
<tr><td>Prompts</td><td>客户端 <span class="arithmatex">\(\to\)</span> 服务器</td><td>获取模板</td><td>“总结这个 PR”模板</td></tr>
<tr><td>Sampling</td><td>服务器 <span class="arithmatex">\(\to\)</span> 客户端</td><td>请求 LLM 生成</td><td>服务器请求 LLM 分类</td></tr></tbody>
</table>
</div>

<p><strong>关键区分</strong>：</p>

<ul>
<li><p><strong>Tools vs Resources</strong>：Tools 有<em>副作用</em>（创建/修改/删除）。Resources 是<em>只读</em>的。这对安全很重要——Agent 可自由读取 Resources，但调用 Tools 必须获得批准。</p></li>
<li><p><strong>Sampling</strong> 方向反转：通常是客户端（Agent）调用服务器（工具）。在 Sampling 中，服务器请求客户端的 LLM 帮忙。用例：代码分析服务器需要 LLM 解释一段代码。</p></li>
<li><p><strong>Prompts</strong> 是元数据（可复用模板），不是执行。它们帮助 Agent 构造更好的工具调用。</p></li>
</ul>

<p><em><strong>复习：</strong>第 <a href="../22-mcp/index.md">22</a> 章（模型上下文协议）。</em></p>
</details></section>
