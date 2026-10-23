# 12.2　LLM Agent 的轨迹缓冲区 {#trajectory-buffers-for-llm-agents}

<p>在 LLM Agent 的语境下，传统的 RL 回放缓冲区经历了结构性的转变。Agent 缓冲区——通常称为<strong>轨迹缓冲区（Trajectory Buffers）</strong>、<strong>经验池（Experience Pools）</strong>或<strong>记忆库（Memory Banks）</strong>——不再存储低维数值 Tensor，而是管理复杂的文本历史、工具执行输出和显式推理步骤。</p>

## 12.2.1　LLM Agent 缓冲区的数学结构 {#mathematical-structure-of-an-llm-agent-buffer}

<p>在经典 RL 中，回放缓冲区存储扁平元组 <span class="arithmatex">\((s, a, r, s&#x27;)\)</span>。对于 LLM Agent，这扩展为高维的 Token 化文本结构：</p>



<div class="arithmatex">\[\boxed{e_t = \left( \mathcal{S}_t,\; \mathcal{A}_t,\; \mathcal{R}_t,\; \mathcal{S}_{t+1} \right)}
\tag{12.1}\]</div>

<ul>
<li><p><span class="arithmatex">\(\mathcal{S}_t\)</span>：<strong>完整上下文状态</strong>——系统 Prompt、用户目标、对话历史，以及当前环境变量（例如 HTML 源代码、目录结构、数据库 Schema）。</p></li>
<li><p><span class="arithmatex">\(\mathcal{A}_t\)</span>：Agent 的<strong>生成输出</strong>，通常由一段思维链（Chain-of-Thought，CoT）推理字符串紧接一个结构化工具调用组成：</p>



<div class="arithmatex">\[\mathcal{A}_t = \{\text{text}_{\text{reasoning}},\; \text{json}_{\text{tool\_call}}\}
\tag{12.2}\]</div></li>
<li><p><span class="arithmatex">\(\mathcal{R}_t\)</span>：<strong>评估信号</strong>，来自外部执行环境（单元测试通过、编译器标志、API 响应码），或由 LLM-as-a-judge 系统验证。</p></li>
<li><p><span class="arithmatex">\(\mathcal{S}_{t+1}\)</span>：<strong>更新后的上下文窗口</strong>，将工具输出文本或错误日志直接附加到对话历史中。</p></li>
</ul>

<div class="admonition example"><p class="admonition-title">具体 Agent 轨迹：代码调试</p>
<p><strong>步骤 1</strong>：<span class="arithmatex">\(\mathcal{S}_1\)</span> = “修复 <code>utils.py</code> 中失败的测试”<br><span class="arithmatex">\(\mathcal{A}_1\)</span> = <em>“让我先读取该文件”</em> + <code>read_file(&quot;utils.py&quot;)</code><br><span class="arithmatex">\(\mathcal{R}_1\)</span> = 0（中间步骤）<br><strong>步骤 2</strong>：<span class="arithmatex">\(\mathcal{S}_2\)</span> = [先前上下文 + 文件内容]<br><span class="arithmatex">\(\mathcal{A}_2\)</span> = <em>“Bug 在第 42 行，是一个差一错误”</em> + <code>edit_file(&quot;utils.py&quot;, ...)</code><br><span class="arithmatex">\(\mathcal{R}_2\)</span> = 0（中间步骤）<br><strong>步骤 3</strong>：<span class="arithmatex">\(\mathcal{S}_3\)</span> = [先前上下文 + 编辑确认]<br><span class="arithmatex">\(\mathcal{A}_3\)</span> = <em>“让我验证修复”</em> + <code>run_tests()</code><br><span class="arithmatex">\(\mathcal{R}_3\)</span> = +1.0（所有测试通过——稀疏终端 reward）</p>
</div>
