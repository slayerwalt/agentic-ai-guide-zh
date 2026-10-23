# 17.7　用强化学习训练记忆系统 {#sec:rl-memory}

## 17.7.1　记忆操作的奖励信号 {#reward-signals-for-memory-operations}

<p>记忆操作（读、写、更新、反思）可以被视为 RL 框架中的动作。挑战在于设计能激励<em>有用</em>记忆行为的 Reward 信号：</p>

<ul>
<li><p><strong>任务 Reward 回传。</strong> 如果某次记忆检索导致了正确答案，则把功劳归于该检索动作。稀疏但毫不含糊。</p></li>
<li><p><strong>检索精度 Reward。</strong> <span class="arithmatex">\(r_\text{retrieve} = \text{Relevance}(d_\text{retrieved}, \text{task})\)</span>，由一个学习得到的相关性模型估计。</p></li>
<li><p><strong>记忆效率 Reward。</strong> 对不必要的写入施加惩罚：<span class="arithmatex">\(r_\text{write} = -\lambda \cdot \mathbf{1}[\text{write}]\)</span>，从而鼓励选择性存储。</p></li>
<li><p><strong>一致性 Reward。</strong> 奖励内部一致（无矛盾）的记忆状态。</p></li>
</ul>

<p>在第 <span class="arithmatex">\(t\)</span> 步对一次记忆操作 <span class="arithmatex">\(m_t\)</span> 的组合 Reward：</p>



<div class="arithmatex">\[r_t^{\text{mem}} = r_t^{\text{task}} + \alpha \cdot r_t^{\text{retrieve}} + \beta \cdot r_t^{\text{write}} + \gamma \cdot r_t^{\text{consistency}}.\]</div>

## 17.7.2　学习“记什么” {#learning-what-to-remember}

<p>“记什么”是一个元学习挑战：Agent 必须学到一个能最大化未来任务表现的写入 Policy <span class="arithmatex">\(\pi_\text{write}(e)\)</span>。其难点在于：</p>

<ol>
<li><p>一条记忆的价值只有在未来才会显现（延迟 Reward）。</p></li>
<li><p>写入时未来可能查询的空间是未知的。</p></li>
<li><p>记忆之间相互影响：存储 <span class="arithmatex">\(e\)</span> 的价值依赖于 <span class="arithmatex">\(\mathcal{M}\)</span> 中已有的其他内容。</p></li>
</ol>

<p>常见的方法有：</p>

<ul>
<li><p><strong>事后重标记（Hindsight relabeling）</strong> <a class="book-citation" id="cite-andrychowicz2017hindsight-1" href="../../references.md#ref-andrychowicz2017hindsight" aria-label="参考文献 8">[8]</a>。一个 Episode 成功后，回溯性地把被检索到的记忆标记为“重要”，并训练写入 Policy 去存储类似条目。</p></li>
<li><p><strong>元强化学习（Meta-RL）</strong> <a class="book-citation" id="cite-duan2016rl2-1" href="../../references.md#ref-duan2016rl2" aria-label="参考文献 79">[79]</a>。在一组任务分布上训练写入 Policy；Policy 学到存储能在任务间泛化的信息。</p></li>
<li><p><strong>好奇心驱动的存储（Curiosity-driven storage）</strong> <a class="book-citation" id="cite-pathak2017curiosity-1" href="../../references.md#ref-pathak2017curiosity" aria-label="参考文献 286">[286]</a>。存储令人意外（预测误差高）的观察，因为它们更可能蕴含信息。</p></li>
</ul>

## 17.7.3　记忆增强的 Policy 优化 {#memory-augmented-policy-optimization}

<p>联合优化 Policy 与其记忆系统的思想可追溯到可微分记忆网络 <a class="book-citation" id="cite-graves2016hybrid-1" href="../../references.md#ref-graves2016hybrid" aria-label="参考文献 119">[119]</a>，并由 REALM <a class="book-citation" id="cite-guu2020realm-1" href="../../references.md#ref-guu2020realm" aria-label="参考文献 128">[128]</a> 推广到检索增强的 LLM。一个记忆增强 Agent 的完整 Policy Gradient 目标为：</p>



<div class="arithmatex">\[\mathcal{L}(\theta, \phi) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^T \gamma^t r_t\right] - \lambda \cdot \mathcal{L}_\text{mem}(\phi),\]</div>

<p>其中 <span class="arithmatex">\(\theta\)</span> 是 LLM 参数，<span class="arithmatex">\(\phi\)</span> 是记忆系统参数（例如检索模型权重），<span class="arithmatex">\(\mathcal{L}_\text{mem}\)</span> 是对记忆复杂度的正则项。</p>

<div class="admonition note"><p class="admonition-title">关键洞见：把记忆作为可学习的归纳偏置</p>
<p>用 RL 训练记忆操作，让 Agent 得以发展出面向具体任务的记忆策略。一个编码 Agent 学会存储 API 签名；一个研究型 Agent 学会存储引文链；一个客服 Agent 学会存储用户投诉模式。记忆系统由此成为一种为 Agent 领域量身打造的<em>可学习归纳偏置</em>。</p>
</div>
