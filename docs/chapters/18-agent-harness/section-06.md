# 18.6　状态管理 {#subsec:state-management}

<p>Agent 本质上是有状态的。harness 必须管理多层状态：</p>

## 18.6.1　对话状态 {#conversation-state}

<p>消息历史是主要的状态产物。每条消息包含：</p>

<ul>
<li><p><strong>角色（Role）：</strong> <code>system</code>、<code>user</code>、<code>assistant</code>、<code>tool</code>。</p></li>
<li><p><strong>内容（Content）：</strong> 文本、工具调用或工具结果。</p></li>
<li><p><strong>元数据（Metadata）：</strong> 时间戳、Token 数、重要性分数、压缩状态。</p></li>
</ul>

## 18.6.2　任务状态 {#task-state}

<p>对于长时间运行的任务，harness 跟踪：</p>

<ul>
<li><p><strong>进度：</strong> 哪些子任务已完成、进行中或待办。</p></li>
<li><p><strong>Checkpoint：</strong> 序列化的状态快照，允许失败后恢复。</p></li>
<li><p><strong>回滚（Rollback）：</strong> 当检测到错误时撤销最近 <span class="arithmatex">\(k\)</span> 个动作的能力。</p></li>
</ul>

## 18.6.3　Agent 状态 {#agent-state}

<p>Agent 的内部状态包括：</p>

<ul>
<li><p><strong>当前计划：</strong> Agent 计划执行的步骤序列。</p></li>
<li><p><strong>挂起动作：</strong> 已发出但尚未返回的工具调用。</p></li>
<li><p><strong>信念（Beliefs）：</strong> Agent 已确立的事实（例如“用户时区是 UTC+9”）。</p></li>
</ul>

## 18.6.4　持久化状态 {#persistent-state}

<p>用于跨会话连续性 <a class="book-citation" id="cite-packer2023memgpt-1" href="../../references.md#ref-packer2023memgpt" aria-label="参考文献 281">[281]</a><a class="book-citation" id="cite-wang2023voyager-1" href="../../references.md#ref-wang2023voyager" aria-label="参考文献 362">[362]</a>：</p>

<ul>
<li><p><strong>用户档案：</strong> 偏好、过往交互、关于用户已学习到的事实。</p></li>
<li><p><strong>长期记忆：</strong> 过往对话的向量数据库，可按语义相似度检索。</p></li>
<li><p><strong>任务历史：</strong> 带结果的已完成任务，用于 few-shot 检索。</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">把状态当作一等公民</p>
<p>在早期的 Agent 框架中，状态是事后才考虑的东西——一个到处传递的全局字典。生产系统将状态视作一等公民，配有显式 schema、版本控制与迁移路径。把 Agent 状态当作数据库 schema 来对待：一开始就要谨慎定义，因为日后再改动很痛苦。</p>
</div>
