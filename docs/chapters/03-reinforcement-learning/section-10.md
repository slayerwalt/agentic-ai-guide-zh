# 3.10　基于模型 vs 无模型 {#model-based-vs-model-free}

<span id="table-3-inline-3"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>Model-Free</strong></th><th scope="col"><strong>Model-Based</strong></th></tr></thead>
<tbody><tr><td><strong>所学内容</strong></td><td>直接学习策略 <span class="arithmatex">\(\pi\)</span> 和/或价值 <span class="arithmatex">\(V\)</span>/<span class="arithmatex">\(Q\)</span></td><td>环境模型 <span class="arithmatex">\(\hat{P}(s&#x27;|s,a)\)</span></td></tr>
<tr><td><strong>规划能力</strong></td><td>无规划，反应式决策</td><td>可以模拟未来轨迹</td></tr>
<tr><td><strong>样本效率</strong></td><td>低（必须经历所有情况）</td><td>高（可在想象中规划）</td></tr>
<tr><td><strong>准确性</strong></td><td>无模型偏差</td><td>模型误差会累积</td></tr>
<tr><td><strong>何时使用</strong></td><td>复杂/未知的转移</td><td>简单转移，追求效率</td></tr>
<tr><td><strong>示例</strong></td><td>PPO、DQN、SAC <a class="book-citation" id="cite-haarnoja2018soft-1" href="../../references.md#ref-haarnoja2018soft" aria-label="参考文献 129">[129]</a></td><td>MuZero <a class="book-citation" id="cite-schrittwieser2020mastering-1" href="../../references.md#ref-schrittwieser2020mastering" aria-label="参考文献 316">[316]</a>、Dreamer <a class="book-citation" id="cite-hafner2020dream-1" href="../../references.md#ref-hafner2020dream" aria-label="参考文献 130">[130]</a>、AlphaGo <a class="book-citation" id="cite-silver2016mastering-1" href="../../references.md#ref-silver2016mastering" aria-label="参考文献 331">[331]</a></td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 LLM RL 是无模型的</p>
<p>语言生成的状态转移是平凡的（将 token 追加到序列——确定性转移）。环境「模型」并非瓶颈。难的是<strong>奖励</strong>——预测人类的偏好。这使得 model-based 方法对 LLM RL 来说没有必要。</p>

<p>RLHF 中的奖励模型在某种意义上也可以被看作一种「模型」（它预测人类偏好），但它被用作奖励信号，而非用于规划/模拟。LLM RL 本质上是无模型的策略优化。</p>
</div>
