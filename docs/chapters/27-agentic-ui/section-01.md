# 27.1　动机：超越聊天框 {#subsec:ui-motivation}

<div class="admonition tip"><p class="admonition-title">为什么 Agent 需要专门的界面</p>
<p>聊天气泡传达的是<em>结果</em>。Agent 化 UI 传达的则是<em>过程</em>——推理、调用的工具、做出的决策，以及需要人类判断介入的节点。没有这种可见性，用户无法信任、纠正 Agent，也无法从中学习。</p>
</div>

<p>聊天界面与 Agent 化界面之间的差距，犹如自动售货机与熟练协作者之间的差距。当一个 Agent 执行 20 步研究任务、浏览网页、撰写并运行代码、综合成一份报告时，用户需要解答的问题是简单的文本响应所无法提供的：</p>

<ul>
<li><p><strong>Agent 此刻在做什么？</strong> 长时间运行的任务需要进度反馈；沉默会滋生不信任。</p></li>
<li><p><strong>Agent 为何做出这个决策？</strong> 推理过程的透明性使用户能够尽早发现错误。</p></li>
<li><p><strong>使用了哪些工具，输入了什么？</strong> 工具来源（tool provenance）对核实事实性论断与审计行为至关重要。</p></li>
<li><p><strong>我应该在哪里介入？</strong> Agent 必须呈现那些需要人类判断的决策点，同时避免用每一项微决策淹没用户。</p></li>
<li><p><strong>我能撤销吗？</strong> 不可逆操作（发送邮件、修改文件、执行代码）需要明确的确认与回滚路径。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">自动化偏差风险</p>
<p>人机自动化交互的研究一致表明，用户会过度信任自动化系统，尤其是当这些系统自信地呈现输出而不附带不确定性信号时 <a class="book-citation" id="cite-parasuraman1997humans-1" href="../../references.md#ref-parasuraman1997humans" aria-label="参考文献 284">[284]</a>。Agent 化 UI 必须通过呈现不确定性、展示推理过程，以及让质疑或覆盖 Agent 决策变得容易，来主动对抗自动化偏差（automation bias）。</p>
</div>

<p>因此，Agent 化 UI 的设计位于人机交互（Human-Computer Interaction, HCI）、可解释 AI（Explainable AI, XAI）与软件工程的交汇处。核心设计目标包括：</p>

<ol>
<li><p><strong>透明性</strong>：让 Agent 的内部状态对用户清晰可读。</p></li>
<li><p><strong>可控性</strong>：提供有意义的介入节点，而不要求持续监督。</p></li>
<li><p><strong>信任校准</strong>：帮助用户建立关于 Agent 能力与局限的准确心智模型。</p></li>
<li><p><strong>效率</strong>：最小化认知负担；在合适的时间呈现合适的信息。</p></li>
<li><p><strong>可恢复性</strong>：让错误的发现与撤销成本低廉。</p></li>
</ol>
