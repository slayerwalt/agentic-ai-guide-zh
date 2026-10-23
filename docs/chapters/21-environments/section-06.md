# 21.6　环境–Agent 接口模式 {#sec:interface-patterns}

<p>图 <a href="#fig:env-agent-interface">21.2</a> 展示了实践中使用的四种主要接口模式。</p>

<figure id="fig:env-agent-interface"><a href="../../assets/figures/fig_065_env-agent-interface.png" target="_blank" rel="noopener" aria-label="查看图 21.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_065_env-agent-interface.png" alt="四种 Agent–环境接口模式。(a) 基于文本是 LLM 最常用的方式；(b) 结构化 JSON 可实现精确解析；(c) 多模态结合截图与可访问性树用于 GUI 任务；(d) 流式接口支持没有离散回合边界的实时交互。" loading="lazy"></a><figcaption><span class="figure-number">图 21.2　</span>四种 Agent–环境接口模式。(a) 基于文本是 LLM 最常用的方式；(b) 结构化 JSON 可实现精确解析；(c) 多模态结合截图与可访问性树用于 GUI 任务；(d) 流式接口支持没有离散回合边界的实时交互。</figcaption></figure>

#### 基于文本的观察/动作。 {#text-based-observationaction.}

<p>Agent 接收字符串观察并产出字符串动作。环境解析该动作（如从 <code>&lt;tool&gt;...&lt;/tool&gt;</code> 块中抽取工具调用）并以字符串形式返回结果。这是兼容性最佳的模式：任何 LLM 无需特殊架构即可参与。</p>

#### 结构化 JSON 观察/动作。 {#structured-json-observationaction.}

<p>观察和动作是符合既定 Schema 的 JSON 对象。这支持严格校验（在执行前拒绝畸形动作）、结构化日志，以及更便利的程序化轨迹分析。代价是 Agent 必须可靠地产出有效 JSON，这要求微调或受约束解码。</p>

#### 多模态（截图 + 可访问性树）。 {#multimodal-screenshot-accessibility-tree.}

<p>用于计算机使用类和 Web 环境。观察为元组 <code>(screenshot: PIL.Image, a11y_tree: dict)</code>。截图提供视觉上下文；可访问性树提供元素标识符，使得动作可不依赖像素级坐标。该混合方式比纯截图控制更鲁棒。</p>

#### 流式 vs. 回合式交互。 {#streaming-vs.-turn-based-interaction.}

<p>当前大多数环境采用回合式模型：Agent 产出完整动作，环境执行后返回下一个观察。流式环境允许 Agent 在观察到达时即接收部分观察（如长时间运行命令的输出），并在流过程中中断或重定向执行。这更接近人类与计算机交互的方式，但对 Agent 架构要求更复杂。</p>
