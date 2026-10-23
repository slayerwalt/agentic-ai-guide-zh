# 第 22 章 · 模型上下文协议（Model Context Protocol, MCP） {#sec:mcp}

<p>工具增强型语言模型的兴起带来了一个碎片化问题：每一个 Agent 框架、每一个 LLM 提供商、每一次企业部署都在发明自己的一套机制来把模型与外部工具和数据源连接起来。<strong>模型上下文协议（Model Context Protocol, MCP）</strong> <a class="book-citation" id="cite-anthropic-mcp-2024-1" href="../../references.md#ref-anthropic-mcp-2024" aria-label="参考文献 9">[9]</a> 由 Anthropic 于 2024 年末提出，是一个旨在彻底解决这一问题的开放标准——为 AI 应用与它们所需工具之间提供一个通用的、与厂商无关的接口。</p>

## 本章内容

<ul class="chapter-outline">
<li id="sec:mcp:motivation"><a href="section-01.md">22.1　动机：工具集成问题</a></li>
<li id="sec:mcp:architecture"><a href="section-02.md">22.2　架构概览</a></li>
<li id="sec:mcp:primitives"><a href="section-03.md">22.3　核心原语</a></li>
<li id="sec:mcp:protocol"><a href="section-04.md">22.4　协议规范</a></li>
<li id="sec:mcp:tools"><a href="section-05.md">22.5　Tool 定义与发现</a></li>
<li id="sec:mcp:security"><a href="section-06.md">22.6　安全模型</a></li>
<li id="sec:mcp:implementation"><a href="section-07.md">22.7　实现模式</a></li>
<li id="sec:mcp:ecosystem"><a href="section-08.md">22.8　MCP 生态</a></li>
<li id="sec:mcp:alternatives"><a href="section-09.md">22.9　MCP vs.~其他方案</a></li>
<li id="sec:mcp:training"><a href="section-10.md">22.10　MCP 在 Agent 训练中的作用</a></li>
<li id="sec:mcp:summary"><a href="section-11.md">22.11　小结</a></li>
</ul>
