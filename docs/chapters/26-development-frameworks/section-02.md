# 26.2　Agent 开发生命周期 {#subsec:agent-lifecycle}

<p>结构化的开发生命周期能帮助团队系统地从概念走向生产。图 <a href="#fig:agent-lifecycle">26.1</a> 展示了五个主要阶段。</p>

<figure id="fig:agent-lifecycle"><a href="../../assets/figures/fig_072_agent-lifecycle.png" target="_blank" rel="noopener" aria-label="查看图 26.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_072_agent-lifecycle.png" alt="Agent 开发生命周期。每个阶段的反馈环确保持续改进。" loading="lazy"></a><figcaption><span class="figure-number">图 26.1　</span>Agent 开发生命周期。每个阶段的反馈环确保持续改进。</figcaption></figure>

## 26.2.1　阶段 1：设计 {#phase-1-design}

<p>设计阶段在尚未写下一行代码之前，先确定 Agent 的<em>能力边界（capability envelope）</em>——它能做什么、不能做什么。</p>

<p><strong>定义能力。</strong> 从能力矩阵开始：以结构化清单列出 Agent 应处理的任务、必须拒绝的边界情况、以及明确不在范围内的行为。该文档将成为后续评测标准的依据。</p>

<p><strong>Tool 选择。</strong> 每个 Tool 都应有明确的目的、定义良好的输入输出，以及失败模式规范。过度配置 Tool 是常见错误：Tool 过多的 Agent 会因 Tool 选择混乱而表现退化，同时延迟也会上升。</p>

<p><strong>约束规范。</strong> 生产 Agent 需要显式约束：每次请求允许的最大 Tool 调用次数、网页浏览的允许域名、数据访问权限以及输出格式要求。这些约束应同时编入系统 Prompt <em>并且</em>以程序方式强制执行。</p>

## 26.2.2　阶段 2：实现 {#phase-2-implementation}

<p>实现阶段涉及三件相互交织的事：Prompt 工程、Tool 集成和编排逻辑。</p>

<p><strong>Prompt 工程。</strong> 生产 Agent 的系统 Prompt 是活的文档，需要版本控制、结构化测试和谨慎的变更管理。常用技术包括思维链（Chain-of-Thought, CoT）脚手架、少样本（Few-shot）示例、显式输出格式说明以及角色（persona）定义。</p>

<p><strong>Tool 集成。</strong> 每个 Tool 都实现为带类型接口的函数，具备完善的错误处理，并在可能时保证幂等性。Tool 的描述（LLM 用于判断何时调用）与 Tool 实现本身同等重要。</p>

<p><strong>编排。</strong> 编排层管理 Agent 循环：调用 LLM、解析 Tool 调用、执行 Tool、更新状态、并决定何时终止。框架选型（见 <a href="section-03.md#subsec:frameworks">26.3</a> 节）会显著影响该层的组织方式。</p>

## 26.2.3　阶段 3：测试 {#phase-3-testing}

<p>Agent 测试将在 <a href="section-05.md#subsec:agent-testing">26.5</a> 节深入介绍。核心原则是<em>在多个粒度上测试</em>：单个 Tool、完整 Agent 循环以及端到端的用户场景。</p>

## 26.2.4　阶段 4：部署 {#phase-4-deployment}

<p>部署相关问题见 <a href="section-07.md#subsec:production-deployment">26.7</a> 节。关键抉择包括同步与异步执行、状态持久化策略以及扩展架构。</p>

## 26.2.5　阶段 5：迭代 {#phase-5-iteration}

<p>迭代阶段闭合「生产行为」与「系统改进」之间的反馈环。它需要：</p>

<ul>
<li><p><strong>失败日志</strong>：每一次 Agent 失败都连带完整上下文（输入、轨迹、错误）一起记录</p></li>
<li><p><strong>失败分类</strong>：按类型（Tool 错误、推理错误、幻觉、循环）对失败进行分类，以识别系统性问题</p></li>
<li><p><strong>Prompt 更新</strong>：Prompt 改动在部署前先通过回归测试套件检验</p></li>
<li><p><strong>微调</strong>：当 Prompt 工程触及瓶颈时，在精选轨迹上微调可以提升表现</p></li>
<li><p><strong>A/B 测试</strong>：以严谨的统计方法在生产流量上验证新版 Agent</p></li>
</ul>
