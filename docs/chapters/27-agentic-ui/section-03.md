# 27.3　Agent 的关键 UI 组件 {#subsec:ui-components}

<p>无论采取何种总体范式，Agent 化 UI 都共享一组反复出现的组件。本节梳理其中最重要的部分，并给出各自的设计指导。</p>

## 27.3.1　思考与推理展示 {#thought-and-reasoning-display}

<p>现代 LLM——尤其是采用思维链（Chain-of-Thought，CoT）或延展思考训练的模型（例如 OpenAI o1/o3、带扩展思考的 Anthropic Claude）——在产出最终响应之前会生成大量内部推理。呈现这些推理是一把双刃剑：它提升了透明度，但冗长的内部独白也可能淹没用户。</p>

<p>最佳实践：</p>

<ul>
<li><p><strong>可折叠的推理区块</strong>：显示摘要（“思考了 12 秒”），并为想看细节的用户提供展开按钮。</p></li>
<li><p><strong>渐进式披露</strong>：默认仅展示最终结论；推理可按需查看。</p></li>
<li><p><strong>结构化推理</strong>：如果模型产出结构化思考（假设、证据、结论），应以视觉层次来呈现，而非堆成大段文字墙。</p></li>
<li><p><strong>推理与响应的区分</strong>：在视觉上清晰区分内部推理（可能包含错误或失败尝试）与最终响应。</p></li>
</ul>

## 27.3.2　工具使用可视化 {#tool-use-visualization}

<p>工具调用是 Agent 与外部世界交互的主要机制。将其可视化对建立信任与进行调试至关重要。</p>

<div class="admonition note"><p class="admonition-title">工具调用的解剖</p>
<p>每次工具调用都有四个值得展示的组成部分：(1) <strong>工具名</strong>与图标；(2) <strong>输入参数</strong>（可能是较大的 JSON）；(3) <strong>输出/结果</strong>（可能很大）；(4) <strong>耗时</strong>（延迟）。UI 必须在完整性与可读性之间取得平衡。</p>
</div>

<p>工具可视化的设计模式：</p>

<ul>
<li><p><strong>内联工具卡片</strong>：消息流中的紧凑卡片，显示工具名、一行输入摘要与状态（运行中/成功/错误）。可展开查看完整细节。</p></li>
<li><p><strong>工具时间线</strong>：横向时间线显示一轮中的全部工具调用及其耗时，便于识别瓶颈。</p></li>
<li><p><strong>输入/输出 diff</strong>：对于修改状态的工具（例如文件编辑），展示前后对比 diff。</p></li>
<li><p><strong>工具图标与品牌</strong>：常用工具（网页搜索、代码执行、文件系统、API）使用可辨识的图标，便于快速扫视。</p></li>
<li><p><strong>错误高亮</strong>：失败的工具调用以红色显示，附带错误消息与任何重试尝试。</p></li>
</ul>

## 27.3.3　进度指示器 {#progress-indicators}

<p>多步 Agent 任务需要丰富的进度反馈：</p>

<ul>
<li><p><strong>步骤级进度</strong>：编号的计划步骤列表，每完成一项即打钩。对于动态计划，步骤可随 Agent 调整而增删。</p></li>
<li><p><strong>Token 流指示器</strong>：生成期间的闪烁光标或动画省略号；为高级用户提供每秒 Token 数计数器。</p></li>
<li><p><strong>预计完成</strong>：在可行时，依据任务复杂度与历史表现给出预计到达时间（ETA），并以合适的不确定性表达（“大约 2–5 分钟”）。</p></li>
<li><p><strong>子任务嵌套</strong>：对于层级化任务，提供带可展开子任务的树状进度视图。</p></li>
<li><p><strong>取消</strong>：清晰可见的“停止”按钮，能优雅地终止 Agent 并总结迄今为止已完成的工作。</p></li>
</ul>

## 27.3.4　审批门控 {#approval-gates}

<p>审批门控（approval gates）是人机协同（Human-in-the-Loop）控制的主要机制。它必须设计得既<em>富含信息</em>（给用户足够的上下文以做出良好决策），又不会令人<em>疲劳</em>（无需对每个琐碎操作都请求批准）。</p>

<div class="admonition warning"><p class="admonition-title">审批门控中的告警疲劳</p>
<p>如果 Agent 请求批准过于频繁，用户将不加阅读地反射式批准——使门控的目的落空。分层审批策略（参见 <a href="section-07.md#subsec:hitl-design">27.7</a> 节）是维持有效监督的关键。</p>
</div>

<p>审批门控的 UI 元素：</p>

<ul>
<li><p><strong>操作摘要</strong>：用通俗语言描述 Agent 想要做什么（“向 john@example.com 发送附带报告的邮件”）。</p></li>
<li><p><strong>风险指示器</strong>：操作可逆性的视觉信号（绿色 = 易于撤销，黄色 = 难以撤销，红色 = 不可逆）。</p></li>
<li><p><strong>批准 / 拒绝 / 修改</strong>：三选项界面；“修改”会在批准前打开操作参数的编辑器。</p></li>
<li><p><strong>上下文面板</strong>：可展开区域，展示 Agent 想执行该操作的缘由（相关推理、先前步骤）。</p></li>
<li><p><strong>超时行为</strong>：清晰说明用户不响应时会发生什么（Agent 暂停而非继续）。</p></li>
</ul>

## 27.3.5　上下文展示 {#context-display}

<p>Agent 会维护影响其行为的内部状态——记忆、激活的工具、检索到的文档、对话历史。让这些状态可见有助于用户理解并预测 Agent 的行为。</p>

<ul>
<li><p><strong>记忆面板</strong>：显示 Agent 当前对用户、任务以及先前交互“记住”了什么。用户可编辑。</p></li>
<li><p><strong>激活工具列表</strong>：Agent 当前可用的工具，配有启用/禁用开关。</p></li>
<li><p><strong>检索到的上下文</strong>：当前位于 Agent 上下文窗口中的文档或数据块，附带来源引用。</p></li>
<li><p><strong>Token 预算指示器</strong>：上下文窗口已消耗多少，帮助用户判断何时开启新会话。</p></li>
</ul>

## 27.3.6　错误与恢复 UI {#error-and-recovery-ui}

<p>Agent 会失败——工具返回错误、模型出现幻觉、计划变得不可行。UI 必须优雅地处理失败：</p>

<ul>
<li><p><strong>错误卡片</strong>：内联显示失败信息，包括错误类型、消息以及 Agent 的解读。</p></li>
<li><p><strong>重试控件</strong>：手动重试按钮，可选地调整参数。</p></li>
<li><p><strong>备选方案</strong>：当主方案失败时，Agent 提出备选；UI 将其作为可选项呈现。</p></li>
<li><p><strong>部分结果</strong>：若多步任务中途失败，UI 显示已完成的步骤及其输出，保留部分价值。</p></li>
<li><p><strong>升级路径</strong>：当 Agent 无法继续时，提供通往人工支持或手动完成的清晰路径。</p></li>
</ul>

## 27.3.7　置信度指示器 {#confidence-indicators}

<p>LLM 是带有（已校准或未校准的）不确定性的概率系统。呈现置信度有助于用户判断何时该信任、何时该核实：</p>

<ul>
<li><p><strong>措辞犹豫的展示</strong>：高亮诸如“我不确定”或“你或许需要核实”之类的措辞，以引起对低置信度论断的注意。</p></li>
<li><p><strong>来源质量指示</strong>：对检索到的信息显示来源新近度、权威性与相关性评分。</p></li>
<li><p><strong>显式不确定性请求</strong>：“你有多自信？”按钮促使 Agent 自评并解释其不确定性。</p></li>
<li><p><strong>核实建议</strong>：对于高风险输出，Agent 主动建议核实步骤（“我建议独立核对这个计算”）。</p></li>
</ul>
