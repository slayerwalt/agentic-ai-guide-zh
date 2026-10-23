# 19.3　生产级循环的解剖 {#sec:loop-anatomy}

<p>一个可用的循环需要五个结构性原语 <a class="book-citation" id="cite-osmani2026loop-1" href="../../references.md#ref-osmani2026loop" aria-label="参考文献 279">[279]</a>，外加外部状态持久化：</p>

## 19.3.1　五个原语 {#sec:loop-primitives}

<p class="table-caption" id="table-19-1"><span class="table-number">表 19.1　</span>循环工程的五个结构性原语。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>原语</strong></th><th scope="col"><strong>在循环中的角色</strong></th><th scope="col"><strong>RL 对应物</strong></th></tr></thead>
<tbody><tr><td>自动化</td><td>按计划或事件触发循环</td><td>回合启动；环境重置</td></tr>
<tr><td>工作树</td><td>隔离并行 Agent</td><td>分布式 RL 中的独立 rollout worker</td></tr>
<tr><td>技能</td><td>固化可复用能力与项目知识</td><td>策略条件化；任务特定奖励塑形</td></tr>
<tr><td>连接器</td><td>对接外部工具与系统</td><td>环境动作空间</td></tr>
<tr><td>子 Agent</td><td>分解与验证</td><td>分层 RL；critic 网络</td></tr></tbody>
</table>
</div>

#### 自动化。 {#par:automations}

<p>自动化是把一次性的 Agent 运行变成真正循环的心跳。它们定义循环<em>何时</em>、<em>为何</em>触发——按 cron 计划、响应 webhook，或由文件系统事件触发。没有自动化，你只有一个 Agent；有了它们，你拥有一个自维持的系统。生产示例包括：夜间 CI 失败分类、每小时依赖漏洞扫描，以及提交后审查。</p>

#### 工作树（隔离）。 {#par:worktrees}

<p>多个 Agent 并行运行的那一刻起，文件级冲突就成了主导性失败模式。Git 工作树——共享仓库历史的独立工作目录——提供了隔离：每个 Agent 在自己的分支上操作，不可能与兄弟 Agent 发生写入冲突。这与分布式 RL 中运行独立 rollout worker（第 <a href="../11-training-infrastructure/section-02.md#sec:parallelism">11.2</a> 节）是同一个原则：并行需要隔离。</p>

#### 技能。 {#par:skills-in-loops}

<p>技能（第 <a href="../23-agent-skills/index.md#sec:agent-skills">23</a> 章）把 Agent 本会在每个循环从头重新推导的项目知识固化下来。在循环语境中，技能充当<em>持久条件</em>——跨迭代保持稳定的约定、构建流程与约束。没有技能，每次循环迭代都从零开始；有了它们，Agent 跨运行累积知识，而不会把上下文窗口浪费在重新发现的事实上。</p>

#### 连接器。 {#par:connectors}

<p>连接器——通常通过 MCP 实现（第 <a href="../22-mcp/index.md#sec:mcp">22</a> 章）——把循环的动作空间扩展到文件系统之外。只能读写文件的循环是受限的；连接了问题跟踪器、CI 系统、预发布环境与团队通信渠道的循环可以闭合完整的反馈回路：发现问题 <span class="arithmatex">\(\to\)</span> 修复 <span class="arithmatex">\(\to\)</span> 验证 <span class="arithmatex">\(\to\)</span> 部署 <span class="arithmatex">\(\to\)</span> 通知。</p>

#### 子 Agent（制造者—检查者分离）。 {#par:sub-agents-loops}

<p>循环工程中最关键的结构性原则是把<em>产出</em>输出的 Agent 与<em>评估</em>输出的 Agent 分开。让模型给自己的输出打分，就像让学生给自己批改考卷——激励是错位的。一个专门的验证子 Agent（可以运行不同的模型或以更高的推理强度运行）提供独立的评估，让无人值守的运行变得可信。这对应 RL 中的 actor-critic 架构：actor（生成器）提出动作；critic（验证器）评估它们。</p>

## 19.3.2　外部状态：循环的记忆 {#sec:loop-state}

<p>上述每个原语都只作用于单次迭代。<em>外部状态</em>是把各次迭代在时间上连接起来的东西。由于 LLM 在两次调用之间是无状态的——模型会忘记当前上下文之外的一切——循环的连续性必须驻留在磁盘上：一个 Markdown 进度文件、一个结构化数据库，或一份受版本控制的日志。这个外部状态有三个功能：</p>

<ol>
<li><p><strong>进度追踪</strong>：尝试过什么、成功了什么、还剩什么。</p></li>
<li><p><strong>失败记忆</strong>：哪些方法试过并失败了，防止循环在相同的死胡同之间震荡。</p></li>
<li><p><strong>交接上下文</strong>：当循环升级给人类或后续运行接管时，状态文件提供完整的审计轨迹。</p></li>
</ol>

<div class="admonition warning"><p class="admonition-title">状态不是可选项</p>
<p>没有外部状态的循环是一个失忆的循环。它会重新尝试已失败的方法、丢失部分进度，并在出错时拿不出审计轨迹。状态机制不必复杂——每轮迭代后提交到 git 的一个 Markdown 文件通常就足够——但它必须存在。</p>
</div>
