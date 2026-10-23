# 第 24 章 · 智能体到智能体通信（Agent-to-Agent, A2A） {#sec:a2a}

<p>随着大语言模型从孤立的助手演化为由专业化 Agent 组成的协作网络，<em>Agent 之间如何对话</em>这一问题已变得与单个 Agent 内部如何推理同等重要。本章将介绍那些使多 Agent 系统能够协调、委派并共同解决任何单一 Agent 都无法独立处理的问题的协议、模式与工程实践。</p>

## 本章内容

<ul class="chapter-outline">
<li id="sec:a2a:motivation"><a href="section-01.md">24.1　动机：为何 Agent 必须相互通信</a></li>
<li id="sec:a2a:google"><a href="section-02.md">24.2　Google A2A 协议</a></li>
<li id="sec:a2a:patterns"><a href="section-03.md">24.3　通信模式</a></li>
<li id="sec:a2a:discovery"><a href="section-04.md">24.4　Agent 发现与路由</a></li>
<li id="sec:a2a:messages"><a href="section-05.md">24.5　消息格式与 Schema</a></li>
<li id="sec:a2a:coordination"><a href="section-06.md">24.6　协调协议</a></li>
<li id="sec:a2a:vs_mcp"><a href="section-07.md">24.7　A2A 与 MCP：互补的协议</a></li>
<li id="sec:a2a:security"><a href="section-08.md">24.8　多 Agent 系统中的安全与信任</a></li>
<li id="sec:a2a:implementation"><a href="section-09.md">24.9　实现示例：多 Agent 研究工作流</a></li>
<li id="sec:a2a:summary"><a href="section-10.md">24.10　小结</a></li>
</ul>
