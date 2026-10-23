# 18.1　什么是 Agent Harness？ {#subsec:what-is-harness}

<div class="admonition note"><p class="admonition-title">定义：Agent Harness</p>
<p><strong>agent harness</strong> 是一种运行时基础设施，它包裹 LLM，将其从一个无状态的文本补全引擎转变为有状态、目标导向的 Agent，能够进行多步推理、工具使用、记忆检索以及与外部系统交互。</p>
</div>

<p>harness 强制实施清晰的 <strong>关注点分离</strong>（separation of concerns）：</p>

<ul>
<li><p><strong>推理（Reasoning）</strong>——完全委托给 LLM；harness 不对模型输出进行二次猜测。</p></li>
<li><p><strong>执行（Execution）</strong>——harness 负责派发工具调用、管理 I/O 并强制沙箱隔离。</p></li>
<li><p><strong>记忆（Memory）</strong>——harness 维护短期（context window）、工作（scratchpad）以及长期（向量库 / 数据库）记忆。</p></li>
<li><p><strong>通信（Communication）</strong>——harness 处理 Agent、用户与外部服务之间的消息路由。</p></li>
<li><p><strong>可观测性（Observability）</strong>——harness 对每一步进行埋点，用于日志、追踪与调试。</p></li>
</ul>

<figure id="fig:harness-arch"><a href="../../assets/figures/fig_053_harness-arch.png" target="_blank" rel="noopener" aria-label="查看图 18.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_053_harness-arch.png" alt="agent harness 的高层架构。LLM 只负责推理；所有执行、记忆、路由与可观测性都由 harness 管理。" loading="lazy"></a><figcaption><span class="figure-number">图 18.1　</span>agent harness 的高层架构。LLM 只负责推理；所有执行、记忆、路由与可观测性都由 harness 管理。</figcaption></figure>

<div class="admonition tip"><p class="admonition-title">为何要分离关注点？</p>
<p>语言模型本质上是一个函数 <span class="arithmatex">\(f_\theta : \text{tokens} \to \text{tokens}\)</span>。它没有持久状态、无法调用 API，也没有时间感知。harness 就是为模型提供“身体”的“操作系统”——持久记忆、执行器（工具）以及调度器（编排器） <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a>。正如操作系统将硬件从应用程序中抽象出来，harness 将基础设施从模型中抽象出来。</p>
</div>
