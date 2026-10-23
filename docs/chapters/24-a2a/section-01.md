# 24.1　动机：为何 Agent 必须相互通信 {#sec:a2a:motivation}

<div class="admonition tip"><p class="admonition-title">专业化的必然性</p>
<p>单一的通才型 Agent 面临一个根本性矛盾：知识的广度与能力的深度。真实世界的任务——例如法律文档审阅、多步科学研究、企业级软件开发——同时需要两者。Agent 间通信通过让一个<em>由专业化 Agent 组成的网络</em>进行协作来化解这一矛盾：每个 Agent 贡献自己的长处，同时把短板委派出去。</p>
</div>

<p>推动结构化 Agent 间通信需求的力量主要有以下几股：</p>

#### 认知负载与上下文限制。 {#cognitive-load-and-context-limits.}

<p>每个 LLM 都在有限的上下文窗口内运行。复杂的工作流——涉及数百份文档、Tool 调用和推理步骤——很快会超出单一 Agent 的内存承载力。通过将任务分解到多个 Agent，每个 Agent 都在可管理的上下文中工作，而编排型 Agent 只需维护高层状态。</p>

#### 专业化与专家能力。 {#specialization-and-expertise.}

<p>不同的 Agent 可针对特定领域进行微调、Prompt 设计或 Tool 配置：例如可访问编译器和测试运行器的 <code>CodeAgent</code>、可访问判例数据库的 <code>LegalAgent</code>、配备统计库的 <code>DataAgent</code>。将子任务路由给合适的专家可同时提升质量与效率。</p>

#### 并行性与吞吐量。 {#parallelism-and-throughput.}

<p>彼此独立的子任务可以同时分派给多个 Agent。一个研究编排器可能并行地把文献检索扇出到五个专业化 Agent，然后再综合它们的结果——大幅缩短墙钟时间。</p>

#### 故障隔离与韧性。 {#fault-isolation-and-resilience.}

<p>当某个 Agent 失败时，设计良好的多 Agent 系统可以换用另一个 Agent 重试、回退到更简单的方案，或者升级到人工介入——而不会让整个工作流崩溃。</p>

#### 委派与交接。 {#delegation-and-handoff.}

<p>随着上下文转移，长时间运行的任务可能需要在 Agent 之间交接。最初的 <code>PlannerAgent</code> 分解目标，将子任务交给 <code>ExecutorAgents</code>，最后由 <code>ReviewerAgent</code> 校验输出——每个 Agent 只接收它所需要的那部分上下文。</p>

<div class="admonition note"><p class="admonition-title">A2A 通信的核心需求</p>
<ol>
<li><p><strong>可发现性</strong>：Agent 必须能够找到其他 Agent 并理解它们的能力。</p></li>
<li><p><strong>互操作性</strong>：由不同团队或厂商构建的 Agent 必须使用共同的协议进行通信。</p></li>
<li><p><strong>异步性</strong>：长时间运行的任务不得阻塞调用方；结果应通过回调或轮询返回。</p></li>
<li><p><strong>安全性</strong>：Agent 之间必须相互认证，并强制执行授权边界。</p></li>
<li><p><strong>可观测性</strong>：每一次消息交换都必须可追踪，以便调试和审计。</p></li>
</ol>
</div>
