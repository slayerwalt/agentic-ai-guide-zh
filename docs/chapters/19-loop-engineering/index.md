# 第 19 章 · 循环工程（Loop Engineering） {#sec:loop-engineering}

<p>从业者与 LLM 智能体交互方式的演变遵循了一条惊人一致的轨迹。2022–2024 年，首要技能是<em>提示工程（prompt engineering）</em>：精心组织措辞，从单次模型调用中引出想要的回答。到 2025 年，焦点转向<em>上下文工程（context engineering）</em>——策展模型在推理时看到的全部 token，包括检索到的文档、工具输出与对话历史 <a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a>。上一章我们讨论了<em>harness 工程</em>：设计围绕 Agent 的运行时环境——工具、沙箱、记忆与护栏。循环工程 <a class="book-citation" id="cite-osmani2026loop-1" href="../../references.md#ref-osmani2026loop" aria-label="参考文献 279">[279]</a> 是这一演进中的下一层：设计<em>迭代控制结构</em>，让 Agent 自主地朝目标前进，无需人类在每一轮输入下一条指令。</p>

<div class="admonition note"><p class="admonition-title">从提示到循环：工程演进的四个阶段</p>
<ol>
<li><p><strong>提示工程</strong>：优化你对模型<em>说了什么</em>（2022–2024）</p></li>
<li><p><strong>上下文工程</strong>：优化模型<em>看到了什么</em>（2025）</p></li>
<li><p><strong>Harness 工程</strong>：优化 Agent<em>运行所在</em>的环境（2025–2026）</p></li>
<li><p><strong>循环工程</strong>：优化让 Agent 持续朝目标工作的<em>循环</em>（2026）</p></li>
</ol>

<p>每一层都包裹住前一层而不取代它。你仍然要写提示；仍然要策展上下文；仍然要构建 harness。循环工程让这一切运转起来。</p>
</div>

<p>这个说法诞生于 2026 年 6 月：Peter Steinberger <a class="book-citation" id="cite-steinberger2026openclaw-1" href="../../references.md#ref-steinberger2026openclaw" aria-label="参考文献 338">[338]</a> 主张开发者应当停止直接提示编码 Agent，转而设计能够提示这些 Agent 的系统——这一观点得到了 Boris Cherny（Anthropic 的 Claude Code 负责人）的印证，他提到自己的角色已经完全转变为编写协调模型行动的外部执行循环 <a class="book-citation" id="cite-cherny2026loops-1" href="../../references.md#ref-cherny2026loops" aria-label="参考文献 52">[52]</a>。这不仅仅是工具偏好。它反映了一个结构性现实：当一次 Agent 运行可能持续一小时并修改数十个文件时，杠杆率最高的工程不再在提示里——而在那个让 Agent 始终保持高效、被验证、不偏离目标的循环里。</p>

## 本章内容

<ul class="chapter-outline">
<li id="sec:context-engineering"><a href="section-01.md">19.1　上下文工程：循环之下的那一层</a></li>
<li id="sec:loop-as-rl"><a href="section-02.md">19.2　把循环工程视为推理时强化学习</a></li>
<li id="sec:loop-anatomy"><a href="section-03.md">19.3　生产级循环的解剖</a></li>
<li id="sec:loop-pseudocode"><a href="section-04.md">19.4　循环的伪代码</a></li>
<li id="sec:loop-patterns"><a href="section-05.md">19.5　循环模式</a></li>
<li id="sec:verification-engineering"><a href="section-06.md">19.6　验证工程</a></li>
<li id="sec:termination-engineering"><a href="section-07.md">19.7　终止工程</a></li>
<li id="sec:loop-failures"><a href="section-08.md">19.8　失败模式与反模式</a></li>
<li id="sec:production-loops"><a href="section-09.md">19.9　生产级循环架构</a></li>
<li id="sec:loop-context-management"><a href="section-10.md">19.10　长时程循环中的上下文管理</a></li>
<li id="sec:when-not-loops"><a href="section-11.md">19.11　何时不该使用循环</a></li>
<li id="sec:loop-economics"><a href="section-12.md">19.12　循环的经济学</a></li>
<li id="sec:loop-history"><a href="section-13.md">19.13　历史背景与相关工作</a></li>
<li id="sec:loop-summary"><a href="section-14.md">19.14　小结</a></li>
</ul>
