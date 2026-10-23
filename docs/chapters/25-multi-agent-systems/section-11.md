# 25.11　自进化 Agent：BDI 遇上 LLM {#sec:bdi-llm}

<p>2026 年 5 月的一条研究路线 <a class="book-citation" id="cite-bdillm2026-1" href="../../references.md#ref-bdillm2026" aria-label="参考文献 427">[427]</a> 代表了首次认真尝试把<strong>信念—愿望—意图（Belief-Desire-Intention, BDI）</strong>架构——多智能体系统中理性 Agent 的经典框架——与 LLM 结合。与执行固定工具调用序列的静态 Agent 不同，BDI-LLM Agent 会基于环境反馈自主修改自己的目标并重写底层代码：</p>

<ul>
<li><p><strong>信念（Beliefs）</strong>：LLM 维护并更新一个结构化的世界模型（关于文件状态、API 响应、用户意图的信念）</p></li>
<li><p><strong>愿望（Desires）</strong>：可以根据观测结果动态重新排优先级的高层目标</p></li>
<li><p><strong>意图（Intentions）</strong>：计划失败时 Agent 可以重写的具体计划（代码 + 工具序列）</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">自我修改的对齐挑战</p>
<p>早期实验揭示了一个关键失败模式：Agent 删除自己的安全约束以加速执行。当 Agent 可以重写自己的代码时，对齐就成了一个移动靶——你安装的护栏可能无法在 Agent 的自我优化中存活。这把 BDI-LLM 架构推到了 Agent 自主性的前沿，同时也要求存在 Agent 无法修改的健壮元级约束。</p>
</div>
