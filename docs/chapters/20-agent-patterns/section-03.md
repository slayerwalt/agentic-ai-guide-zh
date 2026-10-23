# 20.3　设计原则 {#design-principles}

<p>以下原则提炼自 Anthropic 的《构建有效 Agent》指南 <a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a>，适用于所有模式：</p>

<ol>
<li><p><strong>保持简单。</strong> 使用能跑通的最简架构。仅在确有必要时增加复杂度。一个能解决问题的 Prompt 链，永远胜过一个“也许能解决”的多 Agent 系统。</p></li>
<li><p><strong>透明优于聪明。</strong> 每一步都应可检查。避免隐藏状态或隐式推理。当 Agent 失败时，你需要理解<em>原因</em>——不透明的架构使调试无从下手。</p></li>
<li><p><strong>提供良好工具。</strong> 文档完善、类型清晰、错误信息明确的工具是效能放大器。描述含糊的工具会被误用；具备精确 Schema 和使用指南的工具会被正确选择。</p></li>
<li><p><strong>为失败做规划。</strong> 每次工具调用都可能失败。在 harness 层构建重试逻辑、回退方案和优雅降级，使模型无需推理基础设施故障。</p></li>
<li><p><strong>使用结构化输出。</strong> 受约束的生成（JSON Schema、函数调用）防止解析失败。一个产出需要正则解析的自由文本的 Agent 是脆弱的；一个产出经校验 JSON 的 Agent 是鲁棒的。</p></li>
<li><p><strong>使用多样化输入测试。</strong> Agent 行为比单轮对话更易变。同一 Prompt 在不同运行中可能产生不同的工具调用序列。进行对抗性测试，覆盖边界情况、模糊请求和畸形输入。</p></li>
</ol>
