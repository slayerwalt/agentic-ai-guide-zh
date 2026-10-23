# 第 26 章 · Agent 开发框架 {#sec:agent-development-frameworks}

<p>从研究原型迁移到生产级 Agent 系统，是当前 AI 开发中最具挑战性的工程任务之一。学术论文往往在受控环境中展示惊艳的能力，而真实部署却暴露出远超任务性能本身的众多问题：对抗性输入下的可靠性、内部推理过程的可观测性、复杂多步骤 Workflow 的可测试性，以及在每天数百万请求规模下进行服务时的运维开销。本节梳理 Agent 开发框架的全景图——这些为应对上述挑战而出现的工具、库与平台——并就如何构建、测试、部署及迭代生产级 Agent 系统给出实用指引。</p>

## 本章内容

<ul class="chapter-outline">
<li id="subsec:engineering-gap"><a href="section-01.md">26.1　动机：工程能力的鸿沟</a></li>
<li id="subsec:agent-lifecycle"><a href="section-02.md">26.2　Agent 开发生命周期</a></li>
<li id="subsec:frameworks"><a href="section-03.md">26.3　主流框架深度剖析</a></li>
<li id="subsec:open-source-tooling"><a href="section-04.md">26.4　开源 Agent 工具链</a></li>
<li id="subsec:agent-testing"><a href="section-05.md">26.5　Agent 测试与评测</a></li>
<li id="subsec:observability"><a href="section-06.md">26.6　可观测性与调试</a></li>
<li id="subsec:production-deployment"><a href="section-07.md">26.7　生产部署模式</a></li>
<li id="subsec:framework-comparison-dup2"><a href="section-08.md">26.8　框架对比</a></li>
<li id="subsec:implementation-example"><a href="section-09.md">26.9　完整实现示例：生产级研究型 Agent</a></li>
<li id="subsec:agent-dev-summary"><a href="section-10.md">26.10　小结</a></li>
</ul>
