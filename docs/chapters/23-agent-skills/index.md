# 第 23 章 · Agent Skills {#sec:agent-skills}

<p>随着 Agent 从单体式的 prompt-加-工具系统演化为模块化架构，一个关键的设计问题浮现出来：<em>Agent 的能力应如何被组织、发现与组合？</em>答案越来越收敛到 <strong>skill</strong>（技能）这一概念——一些离散、可复用的行为单元，可以在不重新训练的前提下被加载、组合与替换。</p>

<p>这一思路由 Voyager <a class="book-citation" id="cite-wang2023voyager-1" href="../../references.md#ref-wang2023voyager" aria-label="参考文献 362">[362]</a> 推广开来：该工作展示了一个在 Minecraft 中运行的 LLM Agent 可以不断积累一个可执行代码 skill 的库，每个 skill 都经过验证并被存储以供后续复用。同样的原则也适用于生产级 Agent：skill 以可组合、可版本化的形式封装领域专长，其可扩展性远超任何单个 prompt 所能容纳。Skill 通常会包装 MCP server（见第 <a href="../22-mcp/index.md#sec:mcp">22</a> 章）以获得工具访问能力，将 skill 抽象与标准化工具层连接起来。</p>

## 本章内容

<ul class="chapter-outline">
<li id="what-is-a-skill"><a href="section-01.md">23.1　什么是 Skill？</a></li>
<li id="skill-architecture-patterns"><a href="section-02.md">23.2　Skill 架构模式</a></li>
<li id="case-study-anthropics-agent-design"><a href="section-03.md">23.3　案例研究：Anthropic 的 Agent 设计</a></li>
<li id="skill-lifecycle"><a href="section-04.md">23.4　Skill 生命周期</a></li>
<li id="skill-registries-and-marketplaces"><a href="section-05.md">23.5　Skill 注册表与市场</a></li>
<li id="skills-vs.-fine-tuning"><a href="section-06.md">23.6　Skill vs.~微调</a></li>
</ul>
