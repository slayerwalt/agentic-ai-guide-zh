# 30.3　延伸阅读 {#further-reading}

## 30.3.1　基础论文 {#foundational-papers}

<ul>
<li><p><strong>Attention Is All You Need</strong> <a class="book-citation" id="cite-vaswani2017attention-1" href="../../references.md#ref-vaswani2017attention" aria-label="参考文献 357">[357]</a> — Transformer 架构。</p></li>
<li><p><strong>RLHF / InstructGPT</strong> <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a> — 首个大规模 RLHF 部署。</p></li>
<li><p><strong>PPO</strong> <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a> — 近端策略优化（Proximal Policy Optimization）。</p></li>
<li><p><strong>DPO</strong> <a class="book-citation" id="cite-rafailov2023direct-1" href="../../references.md#ref-rafailov2023direct" aria-label="参考文献 302">[302]</a> — 直接偏好优化（Direct Preference Optimization）。</p></li>
<li><p><strong>GRPO / DeepSeek-R1</strong> <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a><a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> — 组相对策略优化与涌现推理。</p></li>
<li><p><strong>ReAct</strong> <a class="book-citation" id="cite-yao2023react-1" href="../../references.md#ref-yao2023react" aria-label="参考文献 408">[408]</a> — LLM Agent 的 Reasoning + Acting 框架。</p></li>
<li><p><strong>Toolformer</strong> <a class="book-citation" id="cite-schick2023toolformer-1" href="../../references.md#ref-schick2023toolformer" aria-label="参考文献 315">[315]</a> — 教 LLM 学会使用工具。</p></li>
<li><p><strong>RAG</strong> <a class="book-citation" id="cite-lewis2020retrieval-1" href="../../references.md#ref-lewis2020retrieval" aria-label="参考文献 208">[208]</a> — 检索增强生成（Retrieval-Augmented Generation）。</p></li>
</ul>

## 30.3.2　系统与扩展 {#systems-and-scaling}

<ul>
<li><p><strong>Megatron-LM</strong> <a class="book-citation" id="cite-shoeybi2019megatron-1" href="../../references.md#ref-shoeybi2019megatron" aria-label="参考文献 327">[327]</a> — 张量并行与流水线并行。</p></li>
<li><p><strong>DeepSpeed ZeRO</strong> <a class="book-citation" id="cite-rajbhandari2020zero-1" href="../../references.md#ref-rajbhandari2020zero" aria-label="参考文献 304">[304]</a> — 显存高效的分布式训练。</p></li>
<li><p><strong>vLLM</strong> <a class="book-citation" id="cite-kwon2023efficient-1" href="../../references.md#ref-kwon2023efficient" aria-label="参考文献 195">[195]</a> — 为高效 LLM 服务而生的 PagedAttention。</p></li>
<li><p><strong>FlashAttention</strong> <a class="book-citation" id="cite-dao2022flashattention-1" href="../../references.md#ref-dao2022flashattention" aria-label="参考文献 65">[65]</a> — IO 感知的精确 attention。</p></li>
</ul>

## 30.3.3　智能体 AI {#agentic-ai}

<ul>
<li><p><strong>Building Effective Agents</strong> <a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a> — 设计模式与原则。</p></li>
<li><p><strong>Voyager</strong> <a class="book-citation" id="cite-wang2023voyager-1" href="../../references.md#ref-wang2023voyager" aria-label="参考文献 362">[362]</a> — Minecraft 中带有技能库的开放式 Agent。</p></li>
<li><p><strong>SWE-bench</strong> <a class="book-citation" id="cite-jimenez2024swebench-1" href="../../references.md#ref-jimenez2024swebench" aria-label="参考文献 169">[169]</a> — 自主软件工程基准。</p></li>
<li><p><strong>OSWorld</strong> <a class="book-citation" id="cite-xie2024osworld-1" href="../../references.md#ref-xie2024osworld" aria-label="参考文献 397">[397]</a> — 完整的计算机使用基准。</p></li>
<li><p><strong>GAIA</strong> <a class="book-citation" id="cite-mialon2023gaia-1" href="../../references.md#ref-mialon2023gaia" aria-label="参考文献 254">[254]</a> — 面向真实任务的通用 AI 助手基准。</p></li>
<li><p><strong>MemGPT</strong> <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a> — 受操作系统启发、面向无界上下文的记忆管理。</p></li>
<li><p><strong>Model Context Protocol</strong> <a class="book-citation" id="cite-anthropic-mcp-2024-1" href="../../references.md#ref-anthropic-mcp-2024" aria-label="参考文献 9">[9]</a> — 工具集成的开放标准。</p></li>
<li><p><strong>Agent-to-Agent Protocol</strong> <a class="book-citation" id="cite-google-a2a-2025-1" href="../../references.md#ref-google-a2a-2025" aria-label="参考文献 113">[113]</a> — 智能体间通信标准。</p></li>
</ul>

## 30.3.4　对齐与安全 {#alignment-and-safety}

<ul>
<li><p><strong>Constitutional AI</strong> <a class="book-citation" id="cite-bai2022constitutional-1" href="../../references.md#ref-bai2022constitutional" aria-label="参考文献 16">[16]</a> — 自监督对齐。</p></li>
<li><p><strong>Sleeper Agents</strong> <a class="book-citation" id="cite-hubinger2024sleeper-1" href="../../references.md#ref-hubinger2024sleeper" aria-label="参考文献 153">[153]</a> — 关于欺骗性对齐的担忧。</p></li>
<li><p><strong>Reflexion</strong> <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> — 从口头自我反思中学习。</p></li>
<li><p><strong>Indirect Prompt Injection</strong> <a class="book-citation" id="cite-greshake2023indirect-1" href="../../references.md#ref-greshake2023indirect" aria-label="参考文献 120">[120]</a> — LLM 集成应用的安全风险。</p></li>
</ul>

## 30.3.5　在线资源 {#online-resources}

<ul>
<li><p><strong>HuggingFace TRL</strong>：<a href="https://github.com/huggingface/trl">https://github.com/huggingface/trl</a> — 生产级 RL 库。</p></li>
<li><p><strong>LangGraph</strong>：<a href="https://github.com/langchain-ai/langgraph">https://github.com/langchain-ai/langgraph</a> — Agent 工作流图。</p></li>
<li><p><strong>OpenAI Agents SDK</strong>：<a href="https://github.com/openai/openai-agents-python">https://github.com/openai/openai-agents-python</a> — 官方 Agent 框架。</p></li>
<li><p><strong>DeepSpeed-Chat</strong>：<a href="https://github.com/microsoft/DeepSpeedExamples">https://github.com/microsoft/DeepSpeedExamples</a> — 端到端 RLHF 流水线。</p></li>
<li><p><strong>DSPy</strong>：<a href="https://github.com/stanfordnlp/dspy">https://github.com/stanfordnlp/dspy</a> — 声明式 prompt 优化。</p></li>
<li><p><strong>AutoGen</strong>：<a href="https://github.com/microsoft/autogen">https://github.com/microsoft/autogen</a> — 多智能体对话框架。</p></li>
</ul>

<p><em>“预测未来最好的方式，就是把它创造出来。”</em><br>— Alan Kay</p>
