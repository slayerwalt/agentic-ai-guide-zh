# 17.6　多智能体系统中的记忆 {#sec:memory-multiagent}

<p>当多个 Agent 协作完成一个共享任务时，记忆变成了一种<em>协调机制</em>——而不仅是个人的知识库。负责任务分解的规划 Agent 必须把子目标传达给执行 Agent；批评者（critic）Agent 必须访问到与被评估 Agent 相同的上下文；研究型 Agent 团队必须避免重复工作。如果没有共享记忆，Agent 之间就只能通过直接消息传递一切，造成带宽瓶颈，并在对话滚出上下文时丢失信息。共享记忆通过提供所有 Agent 都可读写的、持久化且可查询的底座来解决这一问题——把隐式协调（“我希望对方记得”）转换为显式状态（“答案就在 blackboard 上”）。</p>

## 17.6.1　共享记忆池 {#shared-memory-pools}

<p>在多智能体系统中，Agent 可以在共享记忆存储 <span class="arithmatex">\(\mathcal{M}_\text{shared}\)</span> 之外，再各自保留私有存储 <span class="arithmatex">\(\mathcal{M}_i\)</span>：</p>



<div class="arithmatex">\[\text{context}_i(t) = \mathcal{R}(\mathcal{M}_i, q_i) \cup \mathcal{R}(\mathcal{M}_\text{shared}, q_i).\]</div>

<p>共享记忆使<em>隐式协调</em>成为可能：Agent <span class="arithmatex">\(A\)</span> 写入一个发现；Agent <span class="arithmatex">\(B\)</span> 无需显式通信即可检索到。</p>

## 17.6.2　Blackboard 架构 {#blackboard-architecture}

<p><em>blackboard</em>（黑板）模式 <a class="book-citation" id="cite-hayes1985blackboard-1" href="../../references.md#ref-hayes1985blackboard" aria-label="参考文献 135">[135]</a> 是一种经典的多智能体协调机制：</p>

<p>每个 Agent 都从 blackboard 读取并向其写入。一个<em>控制器</em>监控 blackboard，并在某个 Agent 的前置条件满足时将其激活。这把 Agent 彼此解耦：它们通过共享状态而非直接消息进行通信。</p>

## 17.6.3　共享知识中的共识与冲突 {#consensus-and-conflict-in-shared-knowledge}

<p>当多个 Agent 同时写入共享记忆时，冲突难以避免。常用的解决策略包括：</p>

<ul>
<li><p><strong>Last-write-wins（后写者胜出）：</strong> 简单但会丢失信息。</p></li>
<li><p><strong>版本化记忆：</strong> 维护所有写入的历史；Agent 可以查询任一版本。</p></li>
<li><p><strong>投票 / 共识：</strong> 在某个事实被提交之前，要求 <span class="arithmatex">\(n\)</span> 个 Agent 中有 <span class="arithmatex">\(k\)</span> 个达成一致。</p></li>
<li><p><strong>置信度加权合并：</strong> <span class="arithmatex">\(f_\text{merged} = \sum_i w_i f_i\)</span>，其中 <span class="arithmatex">\(w_i\)</span> 是 Agent <span class="arithmatex">\(i\)</span> 的置信度。</p></li>
<li><p><strong>指定权威：</strong> 把不同记忆区域的所有权分配给特定的 Agent。</p></li>
</ul>

<div class="admonition question"><p class="admonition-title">开放问题：分布式记忆一致性</p>
<p>在并发写入、网络分区与对抗性 Agent 的存在下，多智能体系统应当如何维持记忆一致性？经典的分布式系统方案（Paxos、Raft）可以套用但代价高昂。带有界过时（bounded staleness）的近似一致性对许多智能体任务可能已经足够——但合适的取舍仍是一个开放的研究问题。</p>
</div>
