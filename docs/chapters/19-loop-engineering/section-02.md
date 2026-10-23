# 19.2　把循环工程视为推理时强化学习 {#sec:loop-as-rl}

<p>让循环工程与本书相关的核心洞见是：一个精心设计的 Agent 循环在结构上与在推理时运行的 RL 优化过程完全相同——只是不做模型权重的梯度更新：</p>

<span id="eq:loop-as-mdp"></span>

<div class="arithmatex">\[\begin{align*}
s_t &amp;= \text{context}(s_{t-1}, a_{t-1}, o_{t-1}) &amp;&amp; \text{(state = accumulated context)} \nonumber \\

a_t &amp;= \pi_\theta(s_t) &amp;&amp; \text{(action = model generation)} \nonumber \\

o_t &amp;= \text{env}(a_t) &amp;&amp; \text{(observation = tool/environment feedback)} \nonumber \\

r_t &amp;= \text{verify}(s_t, a_t, o_t) &amp;&amp; \text{(reward = verification signal)}

\tag{19.1}
\end{align*}\]</div>

<p>循环持续运行，直到 <span class="arithmatex">\(r_t\)</span> 超过成功阈值或触发终止条件。“策略”<span class="arithmatex">\(\pi_\theta\)</span> 不通过梯度更新；相反，被更新的是<em>状态</em>——观测、错误消息与反思被追加到上下文中，让同一个冻结模型在后续迭代中产生更好的动作。这正是 Reflexion <a class="book-citation" id="cite-shinn2023reflexion-1" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 背后的机制：模型跨迭代改进，靠的不是学习新权重，而是阅读自己的失败历史。</p>

<div class="admonition tip"><p class="admonition-title">循环—RL 对应关系</p>
<p>考虑标准 RL 目标 <span class="arithmatex">\(\max_\pi \mathbb{E}[\sum_{t=0}^T \gamma^t r_t]\)</span>。在循环工程中：</p>

<ul>
<li><p><strong>策略</strong> <span class="arithmatex">\(\pi\)</span>：冻结的 LLM，以不断累积的上下文为条件</p></li>
<li><p><strong>状态</strong> <span class="arithmatex">\(s_t\)</span>：第 <span class="arithmatex">\(t\)</span> 步时上下文窗口的内容</p></li>
<li><p><strong>动作</strong> <span class="arithmatex">\(a_t\)</span>：模型生成的输出（代码编辑、工具调用、消息）</p></li>
<li><p><strong>环境</strong>：工具、文件系统、测试运行器、linter——任何产生观测的东西</p></li>
<li><p><strong>奖励</strong> <span class="arithmatex">\(r_t\)</span>：验证信号（测试通过、linter 干净、指标提升）</p></li>
<li><p><strong>折扣</strong> <span class="arithmatex">\(\gamma\)</span>：隐含在循环预算中——越靠后的步骤成本越高、越不可能成功</p></li>
<li><p><strong>回合</strong>：从目标定义到终止的一次完整循环执行</p></li>
</ul>

<p>与训练时 RL 的关键区别：策略权重是冻结的。“学习”完全通过状态累积发生——随着循环推进，模型获得更丰富的条件信息。这类似于上下文内学习（in-context learning），但环境在主动塑造模型所见。</p>
</div>

<p>这种对应并非仅仅是个比喻。它有直接的工程后果：</p>

<ol>
<li><p><strong>奖励设计很重要</strong>：正如 RLHF 中那样，定义糟糕的奖励（验证标准）会导致 reward hacking——Agent 满足检查的字面要求却违背其意图（例如删除一个失败的测试让 CI 变绿）。</p></li>
<li><p><strong>探索—利用权衡</strong>：重复同一失败方法的循环表现出很差的探索能力。Reflexion <a class="book-citation" id="cite-shinn2023reflexion-2" href="../../references.md#ref-shinn2023reflexion" aria-label="参考文献 326">[326]</a> 这类机制通过自我批评加入显式探索，类似于 PPO 中的熵奖励。</p></li>
<li><p><strong>时域与折扣</strong>：更长的循环面对累积误差与上下文劣化，正如长时域 RL 受困于信用分配难题。预算上限就是有效的时域。</p></li>
<li><p><strong>状态表示</strong>：上下文管理（压缩、剪枝、外化）就是 RL 中状态表示设计在循环工程里的对应物——两者都决定 Agent 能否有效地对自己的处境进行推理。</p></li>
</ol>
