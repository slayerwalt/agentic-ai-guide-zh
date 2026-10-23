# 27.8　无障碍与信任 {#subsec:ui-trust}

<p>信任不是一项功能——它是一个系统在持续按预期行为、清晰自我解释并能从失败中优雅恢复时所涌现出的属性。Agent 化 UI 必须将信任作为一等关切来设计。</p>

## 27.8.1　解释 Agent 的决策 {#explaining-agent-decisions}

<p>Agent 化 UI 中的可解释性超越了展示思维链。它要求：</p>

<ul>
<li><p><strong>决策理由</strong>：对于重要决策，Agent 不仅应解释<em>决定了什么</em>，还应解释<em>为什么</em>——考虑了哪些因素、拒绝了哪些备选、做出了哪些假设。</p></li>
<li><p><strong>来源归属</strong>：论断应链接到其来源；检索到的文档应可引用。</p></li>
<li><p><strong>反事实解释</strong>：“如果你说的是 X 而不是 Y，我会做 Z”——帮助用户理解 Agent 的决策边界。</p></li>
<li><p><strong>不确定性量化</strong>：明确说明置信度，以及驱动不确定性的因素。</p></li>
</ul>

## 27.8.2　展示置信度水平 {#showing-confidence-levels}

<p>置信度指示器必须经过校准且有意义：</p>

<ul>
<li><p><strong>语言化置信度</strong>：对大多数用户而言，自然语言表达（“我相当有信心”、“这点我不太确定”）比数值概率更易解读。</p></li>
<li><p><strong>视觉化置信度</strong>：颜色编码（绿/黄/红）、图标变体或字重可在不增加文字的情况下编码置信度。</p></li>
<li><p><strong>按论断的置信度</strong>：对于含多项论断的响应，逐项的置信度指示（例如内联脚注）比单一的响应级评分更具信息量。</p></li>
</ul>

## 27.8.3　撤销与回滚能力 {#undo-and-rollback-capabilities}

<p>每一项重要的 Agent 操作在技术可行时都应可撤销：</p>

<ul>
<li><p><strong>带撤销的操作日志</strong>：所有 Agent 操作的时序日志，每个可逆操作旁附“撤销”按钮。</p></li>
<li><p><strong>基于快照的回滚</strong>：对于有状态任务（例如代码编辑、文档写作），定期快照支持回滚到任意先前状态。</p></li>
<li><p><strong>Dry-run 模式</strong>：在执行计划之前，Agent 可模拟运行并展示预测的状态变化，允许用户在任何真实操作发生前批准或修改。</p></li>
<li><p><strong>优雅降级</strong>：当撤销不可行时（例如邮件已发出），UI 应清楚说明，并提供最佳可行的替代方案（例如发送跟进邮件）。</p></li>
</ul>

## 27.8.4　UI 中的审计轨迹 {#audit-trails-in-the-ui}

<p>对于企业与受监管的使用场景，审计轨迹（audit trails）必不可少：</p>

<ul>
<li><p><strong>不可变操作日志</strong>：每一次 Agent 操作、工具调用与人类批准都附带时间戳、用户身份与完整参数被记录。</p></li>
<li><p><strong>可导出的历史</strong>：审计轨迹可导出为 JSON、CSV 或 PDF，用于合规报告。</p></li>
<li><p><strong>Diff 视图</strong>：对于文档或代码修改，审计轨迹包含前后对比 diff。</p></li>
<li><p><strong>会话回放</strong>：可逐步回放整个 Agent 会话，用于调试或合规审查。</p></li>
</ul>

## 27.8.5　管理用户期望 {#managing-user-expectations}

<p>失准的期望是用户不信任的主要来源。Agent 化 UI 应主动管理期望：</p>

<ul>
<li><p><strong>能力披露</strong>：清晰、易获取的文档，说明 Agent 能做什么、不能做什么。</p></li>
<li><p><strong>局限承认</strong>：当 Agent 遇到超出其能力的任务时，应明确说明，而不是默默尝试并失败。</p></li>
<li><p><strong>不确定性沟通</strong>：主动沟通不确定性，而非等待用户去发现错误。</p></li>
<li><p><strong>一致的人格</strong>：一致的 Agent 身份与沟通风格能建立熟悉感与可预测性。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">通过透明度建立信任：案例研究</p>
<p>设想一个被指派预订航班的 Agent。低信任 UI 呈现：“我已为您预订航班。确认号：AA1234。”高信任 UI 则呈现：(1) 所使用搜索参数的摘要；(2) 考虑过的备选及选中该航班的原因；(3) 所执行的具体操作（对预订系统的 API 调用）；(4) 含预订链接的确认详情；(5) 在接下来 24 小时内有效的撤销选项；(6) 关于 Agent 无法做之事的说明（例如“我无法修改此预订；您需要直接致电航空公司”）。第二种 UI 占用更多屏幕空间，但建立了用户对 Agent 行为正确性的信心，并给予他们核实与（必要时）恢复所需的信息。</p>
</div>
