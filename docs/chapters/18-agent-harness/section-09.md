# 18.9　框架对比 {#subsec:framework-comparison}

<p class="table-caption" id="table-18-1"><span class="table-number">表 18.1　</span>主流 Agent 编排框架对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>框架</strong></th><th scope="col"><strong>灵活性</strong></th><th scope="col"><strong>复杂度</strong></th><th scope="col"><strong>生产</strong></th><th scope="col"><strong>多 Agent</strong></th><th scope="col"><strong>最适合</strong></th></tr></thead>
<tbody><tr><td>LangChain</td><td>H</td><td>H</td><td>M</td><td>M</td><td>快速原型、链式工作流</td></tr>
<tr><td>LangGraph</td><td>H</td><td>H</td><td>H</td><td>H</td><td>复杂有状态工作流</td></tr>
<tr><td>AutoGen</td><td>M</td><td>M</td><td>M</td><td>H</td><td>多 Agent 对话</td></tr>
<tr><td>CrewAI</td><td>M</td><td>L</td><td>M</td><td>H</td><td>基于角色的团队</td></tr>
<tr><td>OAI Assistants</td><td>L</td><td>L</td><td>H</td><td>L</td><td>简单托管 Agent</td></tr>
<tr><td>OpenAI Swarm</td><td>M</td><td>L</td><td>L</td><td>H</td><td>Handoff 模式</td></tr>
<tr><td>Custom</td><td>H</td><td>H</td><td>H</td><td>H</td><td>完全控制、无厂商绑定</td></tr></tbody>
</table>
</div>

<p><strong>图例：</strong> H = 高，M = 中，L = 低。<strong>灵活性</strong>、<strong>复杂度</strong>、<strong>生产</strong> = 生产就绪度。</p>

<ul>
<li><p><strong>LangChain</strong> <a class="book-citation" id="cite-chase2022langchain-1" href="../../references.md#ref-chase2022langchain" aria-label="参考文献 41">[41]</a>1 提供丰富的集成生态，但学习曲线陡峭，其抽象可能掩盖实际发生的事情。</p></li>
<li><p><strong>LangGraph</strong> <a class="book-citation" id="cite-langchain2024langgraph-1" href="../../references.md#ref-langchain2024langgraph" aria-label="参考文献 154">[154]</a>2 为 LangChain 增加了显式的基于图的控制流，使复杂的多步 Agent 更易管理。</p></li>
<li><p><strong>AutoGen</strong> <a class="book-citation" id="cite-wu2023autogen-1" href="../../references.md#ref-wu2023autogen" aria-label="参考文献 385">[385]</a>3 擅长多 Agent 对话与嵌套对话，对人机协同模式提供良好支持。</p></li>
<li><p><strong>CrewAI</strong> <a class="book-citation" id="cite-moura2023crewai-1" href="../../references.md#ref-moura2023crewai" aria-label="参考文献 261">[261]</a>4 提供了高层、基于角色的抽象（“Agent 团队”），易于上手，但对自定义模式的灵活性较差。</p></li>
<li><p><strong>OpenAI Assistants API</strong>5 完全托管（无需运维基础设施），但定制化有限且存在厂商锁定。</p></li>
<li><p><strong>OpenAI Swarm</strong> <a class="book-citation" id="cite-openai2024swarm-1" href="../../references.md#ref-openai2024swarm" aria-label="参考文献 276">[276]</a>6 是一个轻量、教学性的框架，用于演示 handoff 模式；不适合生产使用。</p></li>
<li><p><strong>自研 harness</strong> 提供最大化的控制权，是具有特定需求的生产系统的正确选择，但需要大量工程投入。</p></li>
</ul>

<div class="admonition question"><p class="admonition-title">何时使用框架，何时自研？</p>
<p>当满足以下条件时使用框架：处于原型阶段、用例契合框架的抽象，或需要快速集成多种工具。在以下情况下自研：你有严格的延迟/成本要求、框架抽象的“泄漏”导致 bug、你需要对 Context 管理进行细粒度控制，或你正在构建一个以 agent harness 为核心差异化竞争力的产品。</p>
</div>
