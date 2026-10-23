# 19.13　历史背景与相关工作 {#sec:loop-history}

<p>循环工程并非凭空出现。它把一条跨越数年的研究谱系产品化了：</p>

<ol>
<li><p><strong>ReAct</strong>（2022） <a class="book-citation" id="cite-yao2023react-1" href="../../references.md#ref-yao2023react" aria-label="参考文献 408">[408]</a>：确立了所有现代循环都继承的推理与行动交错模式。</p></li>
<li><p><strong>Reflexion</strong>（2023） <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a>：加入情景记忆与自我批评，证明 Agent 可以在不更新权重的情况下跨尝试改进。</p></li>
<li><p><strong>AutoGPT</strong>（2023） <a class="book-citation" id="cite-significantgravitas2023autogpt-1" href="../../references.md#ref-significantgravitas2023autogpt" aria-label="参考文献 330">[330]</a>：第一个被广泛使用的自主 Agent 循环，同时展示了无人值守运行的潜力与陷阱（失控执行、高成本、不可靠输出）。</p></li>
<li><p><strong>Self-Refine</strong>（2023） <a class="book-citation" id="cite-madaan2023selfrefine-1" href="../../references.md#ref-madaan2023selfrefine" aria-label="参考文献 246">[246]</a>：把“先生成后批评”的迭代精化循环形式化。</p></li>
<li><p><strong>“Ralph 循环”</strong>（2025）：非正式的基于 bash 的单行脚本，让编码 Agent 在简单的重试循环中运行——产品化循环原语的前身。</p></li>
<li><p><strong>产品化循环</strong>（2026）：主要平台（Codex、Claude Code）直接内嵌 <code>/goal</code> 与 <code>/loop</code> 命令，无需定制基础设施就能使用复杂的循环模式 <a class="book-citation" id="cite-osmani2026loop-1" href="../../references.md#ref-osmani2026loop" aria-label="参考文献 279">[279]</a>。</p></li>
</ol>

<p>从 AutoGPT 到现代循环工程的演进展示了一条成熟之路：早期系统缺少终止逻辑、验证与成本控制——它们是没有工程的循环。2026 年的形式化补上了让无人值守运行变得安全的纪律：显式终止、确定性验证、预算约束与制造者—检查者分离。</p>
