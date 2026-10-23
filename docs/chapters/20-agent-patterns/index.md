# 第 20 章 · Agent 设计模式 {#sec:agent-design-patterns}

<p>构建有效的 Agent 不仅需要强大的模型和一组工具。<em>架构</em>——即如何编排 LLM、如何分解任务、控制流如何在各组件间流动——决定了 Agent 是否可靠、可调试且具备成本效益。本章介绍从 Anthropic、OpenAI、Google 以及开源社区的生产部署中沉淀出的经典设计模式。</p>

<div class="admonition tip"><p class="admonition-title">何时使用 Agent 而非工作流</p>
<p>并非每个任务都需要一个自主 Agent。关键区别在于：</p>

<ul>
<li><p><strong>工作流（Workflow）</strong>：预定义的控制流，在特定步骤调用 LLM。可预测、可测试、成本更低。当任务结构已知时使用。</p></li>
<li><p><strong>Agent</strong>：LLM 动态决定下一步做什么。灵活、可应对新情况。当任务需要自适应决策时使用。</p></li>
</ul>

<p><strong>从工作流开始。</strong> 只有当任务确实需要动态路由或开放式探索时，才升级到 Agent。</p>
</div>

## 本章内容

<ul class="chapter-outline">
<li id="workflow-patterns"><a href="section-01.md">20.1　工作流模式</a></li>
<li id="autonomous-agent-patterns"><a href="section-02.md">20.2　自主 Agent 模式</a></li>
<li id="design-principles"><a href="section-03.md">20.3　设计原则</a></li>
<li id="pattern-selection-guide"><a href="section-04.md">20.4　模式选择指南</a></li>
</ul>
