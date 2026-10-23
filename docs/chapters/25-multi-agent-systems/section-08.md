# 25.8　挑战与解决方案 {#subsec:mas-challenges}

## 25.8.1　协调开销 {#coordination-overhead}

<p>每条 Agent 间消息都会消耗 Token——也就意味着消耗时间与金钱。在朴素实现中，即使不必要，Agent 也会不停通信。</p>

<div class="admonition note"><p class="admonition-title">何时<em>不要</em>通信</p>
<ul>
<li><p>当信息已经存在于共享黑板中时</p></li>
<li><p>当接收方 Agent 当前任务并不需要该信息时</p></li>
<li><p>当消息会重复已经发送过的信息时</p></li>
<li><p>当任务简单到单个 Agent 即可完成时</p></li>
</ul>

<p><strong>准则</strong>：仅当信息的期望价值超过消息成本时才进行通信。</p>
</div>

<p>量化通信成本：若一条消息消耗 <span class="arithmatex">\(c\)</span> 个 Token，接收方 Agent 的任务价值为 <span class="arithmatex">\(v\)</span>，则仅当任务价值的期望提升 <span class="arithmatex">\(\Delta v &gt; c \cdot \text{cost\_per\_token}\)</span> 时才进行通信。</p>

## 25.8.2　冗余 vs. 效率 {#redundancy-vs.-efficiency}

<p>多个 Agent 可能独立求解同一子问题，从而浪费算力。解决方案：</p>

<ul>
<li><p><strong>重复检测（Duplicate detection）</strong>：开始任务前先检查黑板上是否已有结果</p></li>
<li><p><strong>结果缓存</strong>：以语义键存储已完成的子任务结果以便检索</p></li>
<li><p><strong>任务锁</strong>：将任务标记为“进行中”以防止重复执行</p></li>
</ul>

## 25.8.3　归因（Attribution） {#attribution}

<p>当多智能体系统成功或失败时，是哪个 Agent 应当负责？归因对以下方面至关重要：</p>

<ul>
<li><p>RL 的 Reward 分配（功劳分配问题，credit assignment problem）</p></li>
<li><p>调试与改进</p></li>
<li><p>信任校准（决定该依赖哪些 Agent）</p></li>
</ul>

<p><strong>反事实功劳分配（counterfactual credit assignment）</strong>方法通过提问“若该 Agent 采取不同行动，结果会改变多少？”来估计每个 Agent 的贡献。</p>

<span id="eq:counterfactual-credit"></span>

<div class="arithmatex">\[\text{credit}^i = J(\pi^1, \ldots, \pi^n) - J(\pi^1, \ldots, \pi^{i}_{\text{default}}, \ldots, \pi^n)

\tag{25.13}\]</div>

## 25.8.4　可扩展性（Scalability） {#scalability}

<p>朴素的消息传递在 Agent 数量上呈 <span class="arithmatex">\(O(n^2)\)</span> 扩展。解决方案：</p>

<ul>
<li><p><strong>层级通信</strong>：Agent 仅在自身子树内通信</p></li>
<li><p><strong>基于主题的发布/订阅</strong>：Agent 仅订阅相关主题的消息</p></li>
<li><p><strong>稀疏通信图</strong>：仅连接需要交互的 Agent</p></li>
<li><p><strong>异步通信</strong>：Agent 不阻塞等待响应</p></li>
</ul>

## 25.8.5　涌现行为与安全性 {#emergent-behavior-and-safety}

<p>多智能体系统可能展现出意料之外的涌现行为——Agent 之间的交互产生没有任何单个 Agent 被设计来产出的结果。这既是特性（涌现能力），也是风险（涌现失败）。</p>

<div class="admonition warning"><p class="admonition-title">多智能体系统中的安全关切</p>
<ul>
<li><p><strong>Prompt 注入级联</strong>：对一个 Agent 的恶意输入会在系统中传播</p></li>
<li><p><strong>Reward hacking</strong>：Agent 找到违背意图的意外方式来最大化 Reward</p></li>
<li><p><strong>合谋（Collusion）</strong>：在竞争场景下，Agent 可能发展出隐式的合谋策略</p></li>
<li><p><strong>放大（Amplification）</strong>：一个 Agent 中的错误或偏差会被下游 Agent 放大</p></li>
</ul>

<p>始终包含一个<strong>安全监控 Agent（safety monitor agent）</strong>，它观察所有 Agent 间通信，一旦检测到不安全行为即可中止系统。</p>
</div>

## 25.8.6　评估（Evaluation） {#evaluation-dup4}

<p>评估多智能体系统需要多层级的指标：</p>

<p class="table-caption" id="table-25-3"><span class="table-number">表 25.3　</span>多智能体系统的多层级评估指标。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>层级</strong></th><th scope="col"><strong>指标</strong></th><th scope="col"><strong>示例</strong></th></tr></thead>
<tbody><tr><td>系统</td><td>任务完成率</td><td>正确完成任务的百分比</td></tr>
<tr><td>系统</td><td>端到端延迟</td><td>从任务到最终输出所用时间</td></tr>
<tr><td>系统</td><td>总 Token 成本</td><td>所有 Agent 消耗的 Token 总量</td></tr>
<tr><td>Agent</td><td>个体准确率</td><td>单个 Agent 的任务成功率</td></tr>
<tr><td>Agent</td><td>通信效率</td><td>有用消息 / 总消息数</td></tr>
<tr><td>Agent</td><td>贡献分数</td><td>反事实功劳（式 <a href="#eq:counterfactual-credit">25.13</a>）</td></tr>
<tr><td>涌现</td><td>协调质量</td><td>任务重叠/缺口的程度</td></tr></tbody>
</table>
</div>
