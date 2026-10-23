# 27.2　Agent 的 UI 范式 {#subsec:ui-paradigms}

<p>没有任何单一的 UI 范式能适应所有 Agent 化用例。合适的界面取决于任务时长、所需的人类介入程度、输出类型以及用户的专业水平。其谱系从完全对话式的聊天界面，跨越到几乎无需人类交互的完全自主仪表盘。</p>

## 27.2.1　基于聊天的界面 {#chat-based-interfaces}

<p>聊天范式——消息气泡、文本输入框和滚动历史——仍然是与 LLM 交互最为熟悉的入口。其优势在于学习曲线平缓与自然语言的灵活性。在 Agent 化场景下，聊天界面通常会扩展以下能力：</p>

<ul>
<li><p><strong>流式响应</strong>：Token 在生成时即逐步出现，提供即时反馈并降低感知延迟。通过服务器推送事件（Server-Sent Events, SSE）或 WebSockets 实现。</p></li>
<li><p><strong>内联工具指示器</strong>：消息流中的小徽章或可展开区段，用于显示何时调用了工具（例如 “<code>[Searched the web for: climate change 2024]</code>”）。</p></li>
<li><p><strong>输入指示与状态消息</strong>：“Agent is thinking…”、“Running Python code…”、“Fetching results…” 等提示在延迟间隙保持用户的知情感。</p></li>
<li><p><strong>消息线程化</strong>：对于多轮 Agent 任务，可折叠的子线程能容纳中间步骤，而不会让主对话变得杂乱。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">聊天 UI 对 Agent 的局限</p>
<p>聊天界面将本质上并行的过程串行化呈现。当一个 Agent 同时分发任务到五个工具时，线性的消息流会错误地表达实际的执行图。对于复杂的 Agent 化工作流，聊天应被更丰富的范式所扩展——甚至取代。</p>
</div>

## 27.2.2　画布与产物式界面 {#canvas-and-artifact-based-interfaces}

<p>画布范式（canvas paradigm）由 Claude Artifacts1 与 ChatGPT Canvas2 推广，引入了<em>分屏</em>布局：左侧承载对话，右侧（“canvas” 或 “artifact panel”）将生成的内容——代码、文档、图表、电子表格——以可实时编辑的产物（artifact）形式呈现。</p>

<p>关键特征：</p>

<ul>
<li><p><strong>持久化产物</strong>：生成的内容跨轮次持久存在，可以通过自然语言指令迭代优化（“把图表改成蓝色”、“给这个函数加上错误处理”）。</p></li>
<li><p><strong>原位编辑</strong>：用户可以直接编辑产物，Agent 也能观察并响应这些编辑。</p></li>
<li><p><strong>版本历史</strong>：产物维护修订历史，支持回滚到任意先前状态。</p></li>
<li><p><strong>多产物工作区</strong>：进阶实现支持多个并存的产物（例如一个代码文件、其测试套件以及一个文档页面）。</p></li>
</ul>

<p>画布范式特别适合<em>共创</em>（co-creation）任务：写作、编程、数据分析与设计——这些场景的产出是文档或产物，而非对话式的回答。</p>

## 27.2.3　工作流可视化 {#workflow-visualization}

<p>对于执行结构化计划（步骤序列或步骤图）的 Agent，工作流可视化 UI 使计划显式化并可追踪。该范式常见于：</p>

<ul>
<li><p><strong>Agent 化流水线</strong>（LangGraph、AutoGen、CrewAI）：将 Agent 的执行图渲染为有向无环图（Directed Acyclic Graph, DAG）或流程图，节点代表步骤，边代表数据流或控制流。</p></li>
<li><p><strong>任务分解视图</strong>：将 Agent 的高层计划呈现为清单或甘特图式的时间线，每个子任务可展开以显示其自身的步骤。</p></li>
<li><p><strong>实时进度追踪</strong>：节点在执行时改变颜色或显示加载图标；完成的节点展示输出；失败的节点展示错误细节。</p></li>
</ul>

<p>LangGraph Studio3 是该范式的典范，它为 LangGraph Agent 提供基于图的调试器与可视化工具。用户可以检查每个节点的状态、回放执行过程，并注入修改后的状态以测试备选路径。</p>

## 27.2.4　仪表盘与监控界面 {#dashboard-and-monitoring-interfaces}

<p>对于长时间运行或生产环境中的 Agent，仪表盘 UI 提供运维视角：</p>

<ul>
<li><p><strong>实时状态</strong>：哪些 Agent 正在运行、空闲或失败；当前任务与步骤。</p></li>
<li><p><strong>资源指标</strong>：Token 消耗、API 调用计数、延迟直方图、成本估算。</p></li>
<li><p><strong>队列管理</strong>：待处理任务、优先级排序、限流状态。</p></li>
<li><p><strong>告警与异常检测</strong>：异常行为（过多的重试、成本激增、反复失败）以通知形式呈现。</p></li>
<li><p><strong>历史分析</strong>：任务完成率、平均时长、错误频率随时间的变化。</p></li>
</ul>

<p>仪表盘 UI 通常基于 Grafana4、自定义 React 仪表盘或 Streamlit 等工具构建，面向<em>运维人员</em>而非最终用户。</p>

## 27.2.5　协作型界面 {#collaborative-interfaces}

<p>协作型 UI 将 Agent 视为共享工作区（文档、代码库或设计画布）中与人类协作者并列的同侪贡献者。关键特性包括：</p>

<ul>
<li><p><strong>在线指示</strong>：Agent 以带名字的光标或头像出现在共享工作区中。</p></li>
<li><p><strong>变更归属</strong>：Agent 所做的编辑在视觉上与人类编辑相区分（例如用颜色编码的 diff）。</p></li>
<li><p><strong>内联建议</strong>：Agent 以修订标记或评论的形式提出变更，人类可以接受、拒绝或修改。</p></li>
<li><p><strong>冲突解决</strong>：当 Agent 与人类同时编辑同一区域时，UI 会呈现冲突并辅助解决。</p></li>
</ul>

<p>该范式正在 Cursor5（带 AI 的协作式代码编辑）、Notion AI6 以及集成 Gemini 的 Google Docs7 等工具中兴起。</p>

## 27.2.6　带检查点的自主模式 {#autonomous-with-checkpoints}

<p>在自主性谱系的另一端，一些 Agent 几乎完全独立运行——浏览网页、编写代码、执行命令——仅在预定义的、需要人类批准的<em>检查点</em>（checkpoint）处现身。该范式被用于：</p>

<ul>
<li><p><strong>计算机使用 Agent</strong>（Anthropic Computer Use8、OpenAI Operator9）：Agent 控制浏览器或桌面；UI 显示实时屏幕画面，并在不可逆操作前暂停等待批准。</p></li>
<li><p><strong>带门控的自动化流水线</strong>：CI/CD 风格的工作流——Agent 完成一个阶段后等待人类“合并”方可继续。</p></li>
<li><p><strong>定时 Agent</strong>：按计划运行并异步报告结果的 Agent，通过基于通知的 UI 来审阅输出并批准后续操作。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">实践中的检查点 UI</p>
<p>被指派“清理我的邮件收件箱”的 Agent 可能会自主分类并归档 500 封邮件，然后暂停并给出摘要：“我发现 23 封邮件来自您 6 个月未打开的邮件列表。是否需要全部、部分或都不退订？”用户审阅列表、做出选择，Agent 继续执行。这种模式——以人类决策点穿插的自主执行——在效率与可控之间取得平衡。</p>
</div>
