# 4.2　将文本生成建模为 MDP {#text-generation-as-mdp}

<p>让 RL 得以应用于语言模型的关键洞察，是将自回归生成重述为一个 Markov 决策过程（MDP）：</p>

<div class="admonition tip"><p class="admonition-title">“LLM 即 Agent”的类比</p>
<p>把大语言模型想象成一个 <strong>Agent</strong>，正在一个 token 接一个 token 地撰写回答。在每一步，它查看到目前为止写下的全部内容（<em>state</em>）、选择下一个词（<em>action</em>），页面随之增长一个 token（<em>transition</em>）。当回答完成时，一位评判者对其打分（<em>reward</em>）。目标是：学到一种持续获得高分的写作策略（<em>policy</em>）。</p>
</div>

<p>形式化地，文本生成的 MDP 定义如下：</p>

<ul>
<li><p><strong>State</strong> <span class="arithmatex">\(s_t = (x, y_1, \ldots, y_{t-1})\)</span>：Prompt 与到目前为止已生成的所有 token 的拼接。</p></li>
<li><p><strong>Action</strong> <span class="arithmatex">\(a_t \in \{1, \ldots, |\mathcal{V}|\}\)</span>：从词表（32K–128K 选项）中选出下一个 token。</p></li>
<li><p><strong>Transition</strong> <span class="arithmatex">\(P(s_{t+1}|s_t, a_t)\)</span>：确定性的——只需追加所选 token。环境无随机性。</p></li>
<li><p><strong>Reward</strong> <span class="arithmatex">\(r\)</span>：通常仅在生成结束时给出（稀疏）。对 RLHF 而言是 reward 模型评分；对 RLVR 而言是最终答案的正确性。</p></li>
<li><p><strong>Policy</strong> <span class="arithmatex">\(\pi_\theta(a_t|s_t)\)</span>：大语言模型的下一 token 概率分布——正是 Softmax 输出已经计算出的东西。</p></li>
<li><p><strong>折扣因子</strong> <span class="arithmatex">\(\gamma = 1.0\)</span>：Episode 是有限的（即一条回答），因此无需折扣。</p></li>
</ul>

<p>这种映射之所以强大，是因为大语言模型<em>本身就已经是</em>一个 policy——其 Softmax 输出为每一个 state 定义了 <span class="arithmatex">\(\pi_\theta(a_t|s_t)\)</span>。我们无需另行构建一个 policy 网络；只需调整权重 <span class="arithmatex">\(\theta\)</span>，让模型为能获得更高 reward 的 token 序列赋予更高概率。</p>
