# 第 23 章 · Agent Skills {#sec:agent-skills}

<p>随着 Agent 从单体式的 prompt-加-工具系统演化为模块化架构，一个关键的设计问题浮现出来：<em>Agent 的能力应如何被组织、发现与组合？</em>答案越来越收敛到 <strong>skill</strong>（技能）这一概念——一些离散、可复用的行为单元，可以在不重新训练的前提下被加载、组合与替换。</p>

<p>这一思路由 Voyager <a class="book-citation" id="cite-wang2023voyager-1" href="../../references.md#ref-wang2023voyager" aria-label="参考文献 362">[362]</a> 推广开来：该工作展示了一个在 Minecraft 中运行的 LLM Agent 可以不断积累一个可执行代码 skill 的库，每个 skill 都经过验证并被存储以供后续复用。同样的原则也适用于生产级 Agent：skill 以可组合、可版本化的形式封装领域专长，其可扩展性远超任何单个 prompt 所能容纳。Skill 通常会包装 MCP server（见第 <a href="../22-mcp/index.md#sec:mcp">22</a> 章）以获得工具访问能力，将 skill 抽象与标准化工具层连接起来。</p>

## 23.1　什么是 Skill？ {#what-is-a-skill}

<p>一个 <strong>skill</strong> 是一个自包含的能力模块，赋予 Agent 在某个特定领域或任务上的专长。与仅暴露单个函数的原始 tool 不同，一个 skill 涵盖：</p>

<ul>
<li><p><strong>系统 prompt 增强</strong>：注入到 Agent 上下文中的领域特定指令、约束与人设要素。</p></li>
<li><p><strong>Tool 绑定</strong>：该 skill 所需的一个或多个 tool（API、MCP server、本地命令）。</p></li>
<li><p><strong>知识</strong>：Agent 正确执行该 skill 所需的参考资料、示例或 few-shot 演示。</p></li>
<li><p><strong>工作流逻辑</strong>：引导 Agent 完成复杂任务的多步骤流程、决策树或条件流。</p></li>
<li><p><strong>安全护栏</strong>：该 skill 特有的安全约束、输出格式要求与校验规则。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">Skill vs. Tool vs. Agent</p>
<span id="table-23-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>概念</strong></th><th scope="col"><strong>范围</strong></th><th scope="col"><strong>示例</strong></th></tr></thead>
<tbody><tr><td><strong>Tool</strong></td><td>单次函数调用</td><td><code>web_search(query)</code></td></tr>
<tr><td><strong>Skill</strong></td><td>完整的能力（prompts + tools + 知识）</td><td>“Research Analyst” skill</td></tr>
<tr><td><strong>Agent</strong></td><td>拥有多个 skill 的自主实体</td><td>一个编程助手</td></tr></tbody>
</table>
</div>

<p>Tool 是一把锤子。Skill 是知道<em>如何搭起房屋的框架</em>。Agent 则是决定使用哪些 skill 的木匠。</p>
</div>

## 23.2　Skill 架构模式 {#skill-architecture-patterns}

### 23.2.1　静态 Skill 加载 {#static-skill-loading}

<p>最简单的模式：根据配置在 Agent 初始化时加载 skill。Agent 始终可以访问其全部 skill。</p>

<div class="highlight"><pre><span></span><code><span class="c1"># 伪代码 —— 与具体框架无关的模式</span>
<span class="n">agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"claude-sonnet-4-20250514"</span><span class="p">,</span>
    <span class="n">skills</span><span class="o">=</span><span class="p">[</span><span class="s2">"code-review"</span><span class="p">,</span> <span class="s2">"documentation"</span><span class="p">,</span> <span class="s2">"testing"</span><span class="p">],</span>
    <span class="c1"># 每个 skill 都会向 Agent 添加 prompts、tools 与知识</span>
<span class="p">)</span>
</code></pre></div>

<p><strong>优点：</strong>简单、可预测、低延迟。<br><strong>缺点：</strong>未被使用的 skill 会浪费上下文窗口；无法扩展到数百个 skill。</p>

### 23.2.2　动态 Skill 发现 {#dynamic-skill-discovery}

<p>Agent 根据当前任务选择激活哪些 skill。一个 skill 路由器（通常是一个轻量分类器或基于 embedding 的匹配器）来判断相关性：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># 伪代码 —— 与具体框架无关的模式</span>
<span class="n">relevant_skills</span> <span class="o">=</span> <span class="n">skill_router</span><span class="o">.</span><span class="n">match</span><span class="p">(</span>
    <span class="n">user_request</span><span class="o">=</span><span class="n">message</span><span class="p">,</span>
    <span class="n">available_skills</span><span class="o">=</span><span class="n">skill_registry</span><span class="p">,</span>
    <span class="n">max_skills</span><span class="o">=</span><span class="mi">3</span>
<span class="p">)</span>
<span class="n">agent</span><span class="o">.</span><span class="n">activate</span><span class="p">(</span><span class="n">relevant_skills</span><span class="p">)</span>
</code></pre></div>

<p><strong>优点：</strong>可扩展到大型 skill 库；上下文效率高。<br><strong>缺点：</strong>路由错误可能漏掉相关 skill；引入额外延迟。</p>

### 23.2.3　层级化 Skill 组合 {#hierarchical-skill-composition}

<p>Skill 可以依赖于其他 skill，从而构成一个有向无环图（DAG）。一个高层 skill（例如“部署应用”）可能会调用一些子 skill（“跑测试”、“构建 Docker 镜像”、“更新 DNS”）：</p>

<ul>
<li><p>Skill 显式声明其依赖</p></li>
<li><p>编排器在执行前解析依赖图</p></li>
<li><p>子 skill 可被多个父 skill 共享</p></li>
</ul>

## 23.3　案例研究：Anthropic 的 Agent 设计 {#case-study-anthropics-agent-design}

<p>Anthropic 对 Agent 架构的方法 <a class="book-citation" id="cite-anthropic2024buildingagents-1" href="../../references.md#ref-anthropic2024buildingagents" aria-label="参考文献 10">[10]</a> 是关于生产环境中基于 skill 的 Agent 设计最清晰的阐述之一。其设计哲学强调<strong>简单优于复杂</strong>、<strong>可组合的构建块优于单体框架</strong>。（这些模式也会在第 <a href="../20-agent-patterns/index.md#sec:agent-design-patterns">20</a> 章从编排视角进行讨论。）</p>

### 23.3.1　核心原则 {#core-principles}

<ol>
<li><p><strong>从最简单的方案起步。</strong>在尝试更简单的方案（单次 LLM 调用、检索 + 生成）并确认其不足之前，不要急于使用 Agent 化模式。</p></li>
<li><p><strong>Workflow 与 Agent 的区分。</strong>Anthropic 将以下两者加以区分：</p>

<ul>
<li><p><strong>Workflows</strong>：对 LLM 调用进行预先编排——确定性的控制流，在特定节点包含 LLM 步骤。更可预测、更易调试。</p></li>
<li><p><strong>Agents</strong>：由 LLM 动态决定下一步该做什么——工具选择、迭代次数与停止条件都由模型驱动。更灵活，但更难控制。</p></li>
</ul></li>
<li><p><strong>以增强型 LLM 作为原子单元。</strong>原语从来不是裸模型——它总是与其检索源、可调用工具和持久化记忆捆绑在一起的模型。这一组合单元在实践中就是“配备了 skill 的模型”。</p></li>
</ol>

### 23.3.2　构建块模式 {#building-block-patterns}

<p>Anthropic 总结了五种可组合的 workflow 模式，可作为 skill 模板使用：</p>

<p class="table-caption" id="table-23-1"><span class="table-number">表 23.1　</span>Anthropic 提出的可组合 Agent 模式。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>机制</strong></th><th scope="col"><strong>适用场景</strong></th></tr></thead>
<tbody><tr><td><strong>Prompt Chaining</strong></td><td>顺序的 LLM 调用，每一步的输出作为下一步的输入。步骤之间设有关卡，校验中间结果。</td><td>具有清晰分解的多步骤转换任务</td></tr>
<tr><td><strong>Routing</strong></td><td>一个分类器或 LLM 根据任务类型将输入分发给专门的处理者（skill）。</td><td>需要不同领域专长的多个独立任务类别</td></tr>
<tr><td><strong>Parallelization</strong></td><td>多个 LLM 调用同时进行——可以是分段（拆分任务），也可以是投票（同一任务、汇总结果）。</td><td>相互独立的子任务；或通过共识获得置信度</td></tr>
<tr><td><strong>Orchestrator–Workers</strong></td><td>一个中心 LLM 将任务拆分为子任务，分派给 worker LLM，再综合结果。</td><td>子任务难以事先预测的复杂任务</td></tr>
<tr><td><strong>Evaluator–Optimizer</strong></td><td>一个 LLM 生成，另一个 LLM 评估；反复迭代直至达到质量阈值。</td><td>具有明确质量标准的任务（代码、写作）</td></tr></tbody>
</table>
</div>

### 23.3.3　增强型 LLM {#the-augmented-llm}

<p>在 Anthropic 的框架中，基本单元不是裸模型，而是<strong>增强型 LLM</strong>：</p>



<div class="arithmatex">\[\text{Augmented LLM} = \text{Model} + \text{Retrieval} + \text{Tools} + \text{Memory}\]</div>

<p>这与 skill 概念直接对应：每个 skill 都为一个特定任务配置该模型可访问的检索源、tool 与记忆存储。Skill 边界定义了模型在某次特定调用中<em>能看到什么、能做什么</em>。</p>

### 23.3.4　实际影响 {#practical-implications-dup2}

<div class="admonition tip"><p class="admonition-title">Anthropic 的关键洞见</p>
<p>最有效的 Agent 并不是最复杂的，而是<strong>配有好用工具的简单循环</strong>：</p>

<div class="highlight"><pre><span></span><code><span class="k">while</span> <span class="ow">not</span> <span class="n">done</span><span class="p">:</span>
    <span class="n">action</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">decide</span><span class="p">(</span><span class="n">context</span><span class="p">,</span> <span class="n">tools</span><span class="p">)</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">execute</span><span class="p">(</span><span class="n">action</span><span class="p">)</span>
    <span class="n">context</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">result</span><span class="p">)</span>
    <span class="n">done</span> <span class="o">=</span> <span class="n">llm</span><span class="o">.</span><span class="n">should_stop</span><span class="p">(</span><span class="n">context</span><span class="p">)</span>
</code></pre></div>

<p>智能来自（1）模型自身的能力、（2）工具描述的质量、（3）任务表述的清晰度——而非花哨的编排逻辑。Skill 为（2）和（3）提供了结构。</p>
</div>

##### 源自 Anthropic 方法的设计建议： {#design-recommendations-from-anthropics-approach}

<ul>
<li><p><strong>保持 Agent 循环简单</strong>：避免过度工程化控制流，让模型自己决策。</p></li>
<li><p><strong>在工具质量上投入</strong>：详尽、无歧义的工具描述比复杂的路由逻辑更有价值。</p></li>
<li><p><strong>使用结构化输出</strong>：强制模型以可解析的格式（JSON、函数调用）输出决策——可减少 skill 执行错误。</p></li>
<li><p><strong>内建恢复机制</strong>：Skill 应优雅地处理错误——用不同参数重试、请求澄清，或升级到由人接管。</p></li>
<li><p><strong>限制单个 skill 的范围</strong>：试图无所不能的 skill 什么也做不好。狭窄、定义明确的 skill 比宽泛的 skill 更易组合。</p></li>
</ul>

## 23.4　Skill 生命周期 {#skill-lifecycle}

<ol>
<li><p><strong>发现</strong>：系统识别有哪些可用的 skill（注册表、市场、本地定义）。</p></li>
<li><p><strong>选择</strong>：根据用户请求匹配并加载相关 skill。</p></li>
<li><p><strong>激活</strong>：将 skill 的 prompt、tool 与知识注入到 Agent 的上下文中。</p></li>
<li><p><strong>执行</strong>：Agent 利用该 skill 的能力完成任务。</p></li>
<li><p><strong>失活</strong>：移除 skill 上下文，为后续任务释放上下文窗口空间。</p></li>
<li><p><strong>学习</strong>：执行结果可能更新该 skill 的 few-shot 示例或微调路由。</p></li>
</ol>

## 23.5　Skill 注册表与市场 {#skill-registries-and-marketplaces}

<p>生产级 skill 系统需要相应的基础设施：</p>

<ul>
<li><p><strong>Skill 清单（manifest）</strong>：一份结构化描述（名称、能力、所需工具、输入/输出 schema），用于支持自动发现与路由。</p></li>
<li><p><strong>版本控制</strong>：Skill 会不断演进；Agent 需要锁定特定版本以保证可复现性。</p></li>
<li><p><strong>依赖解析</strong>：Skill 可能需要特定的 MCP server、API key 或其他 skill。</p></li>
<li><p><strong>权限模型</strong>：并非所有 Agent 都应能访问所有 skill（安全性、成本与能力边界考虑）。</p></li>
<li><p><strong>市场</strong>：组织可以发布、分享与安装 skill——类似于代码的包管理器。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">Skill 清单示例</p>
<p>一份 skill 清单声明了编排器加载并调用一个 skill 所需的一切。目前尚不存在行业标准 schema；下面是一个示意性格式，它涵盖了真实实现（Anthropic MCP、OpenAI 函数规范、LangChain 工具定义）中常见的字段：</p>

<div class="highlight"><pre><span></span><code><span class="o">//</span> <span class="n">示意性</span> <span class="n">schema</span> <span class="err">——</span> <span class="n">并非任何具体</span> <span class="n">SDK</span> <span class="n">的格式</span>
<span class="p">{</span>
  <span class="s2">"name"</span><span class="p">:</span> <span class="s2">"code-review"</span><span class="p">,</span>
  <span class="s2">"description"</span><span class="p">:</span> <span class="s2">"Review code changes for bugs, style, and security issues"</span><span class="p">,</span>
  <span class="s2">"version"</span><span class="p">:</span> <span class="s2">"2.1.0"</span><span class="p">,</span>
  <span class="s2">"requires"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"tools"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"file_read"</span><span class="p">,</span> <span class="s2">"grep"</span><span class="p">,</span> <span class="s2">"git_diff"</span><span class="p">],</span>
    <span class="s2">"mcp_servers"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"github"</span><span class="p">],</span>
    <span class="s2">"models"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"claude-sonnet-4-20250514"</span><span class="p">]</span>
  <span class="p">},</span>
  <span class="s2">"input_schema"</span><span class="p">:</span> <span class="p">{</span>
    <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"object"</span><span class="p">,</span>
    <span class="s2">"properties"</span><span class="p">:</span> <span class="p">{</span>
      <span class="s2">"repo"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">},</span>
      <span class="s2">"pr_number"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"integer"</span><span class="p">}</span>
    <span class="p">}</span>
  <span class="p">},</span>
  <span class="s2">"prompts"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"skills/code-review/system.md"</span><span class="p">],</span>
  <span class="s2">"knowledge"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"skills/code-review/style-guide.md"</span><span class="p">]</span>
<span class="p">}</span>
</code></pre></div>
</div>

## 23.6　Skill vs. 微调 {#skills-vs.-fine-tuning}

<p>一个自然的问题：为什么要使用运行时的 skill 注入，而不是对模型进行微调？</p>

<p class="table-caption" id="table-23-2"><span class="table-number">表 23.2　</span>用于添加能力的 Skill（上下文内）与微调对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>Skill（上下文内）</strong></th><th scope="col"><strong>微调</strong></th></tr></thead>
<tbody><tr><td>部署速度</td><td>即时</td><td>数小时–数天</td></tr>
<tr><td>灵活性</td><td>运行时可替换/组合</td><td>训练时固定</td></tr>
<tr><td>上下文成本</td><td>占用上下文窗口</td><td>运行时零成本</td></tr>
<tr><td>深层行为改变</td><td>受上下文长度限制</td><td>深层参数级变化</td></tr>
<tr><td>多租户</td><td>不同用户可用不同 skill</td><td>所有用户共用同一模型</td></tr>
<tr><td>维护</td><td>更新文本文件</td><td>在新数据上重新训练</td></tr></tbody>
</table>
</div>

<p>在实践中，两种方法是互补的：微调提供<em>基础能力</em>（指令跟随、工具使用格式、推理），而 skill 在运行时叠加提供<em>任务特定的专长</em>。</p>
