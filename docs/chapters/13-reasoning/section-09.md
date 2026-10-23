# 13.9　推理时算力扩展定律 {#sec:inference-time-scaling}

<p>训练时扩展定律（Chinchilla <a class="book-citation" id="cite-hoffmann2022chinchilla-1" href="../../references.md#ref-hoffmann2022chinchilla" aria-label="参考文献 140">[140]</a>）已被充分理解：模型性能随算力、数据与参数可预测地提升。<strong>推理时算力扩展</strong>则是新的前沿——这个实证发现表明，在<em>测试时</em>投入更多计算（更长的推理链、更多的搜索、迭代式精化）可以解决更大的 base 模型也解决不了的问题。</p>

<div class="admonition note"><p class="admonition-title">核心发现：把测试时算力当作一条扩展轴</p>
<p><strong>Snell 等人（2024）</strong> <a class="book-citation" id="cite-snell2024scaling-1" href="../../references.md#ref-snell2024scaling" aria-label="参考文献 335">[335]</a> 证明：对于困难的推理任务，测试时算力可以比扩展模型参数更有效：</p>

<ul>
<li><p>性能随推理 token 数<strong>对数线性</strong>提升：<span class="arithmatex">\(\text{acc} \approx a \cdot \log(\text{tokens}) + b\)</span>。</p></li>
<li><p>存在<strong>任务特定的最优 token 预算</strong>，超过之后收益急剧递减。</p></li>
<li><p>一个推理算力多 <span class="arithmatex">\(k\times\)</span> 的小模型可以匹敌训练参数多 <span class="arithmatex">\(k\times\)</span> 的大模型——而且对罕见的困难查询，服务成本更低。</p></li>
</ul>

<p>实证里程碑：<strong>o3</strong> 通过扩展推理时搜索，在 ARC-AGI（一个专门设计来抵抗模式匹配的基准）上取得 <strong>45.1%</strong>，远远超出仅靠模型规模所能达到的水平。</p>
</div>

## 13.9.1　过度思考问题 {#subsec:overthinking}

<p>推理时扩展引入了一个关键的失败模式：<strong>过度思考（overthinking）</strong>。被训练生成长思维链的推理模型即使在答案显而易见时也会继续生成 token，浪费算力，有时还会因引入虚假的推理步骤而降低准确率。</p>

<div class="admonition warning"><p class="admonition-title">过度思考：何时更多 token 反而有害</p>
<p>在简单问题上，推理模型的准确率随 token 预算呈现出一条典型的<strong>U 形曲线</strong>：</p>

<ol>
<li><p>短链：推理不足，准确率低。</p></li>
<li><p>最优链：推理正确，准确率峰值。</p></li>
<li><p>过长的链：模型对正确答案反复自我怀疑、引入错误，准确率<em>下降</em>。</p></li>
</ol>

<p>这意味着固定的“想得越久越好”策略是次优的。模型需要学会<strong>何时停止思考</strong>。</p>
</div>

## 13.9.2　高效推理技术 {#subsec:efficient-reasoning}

<p>有几种方法应对推理时扩展中的算力—质量权衡：</p>

<ul>
<li><p><strong>思维草图（Sketch-of-Thought）</strong>：在完整推理链之前先生成压缩的推理草图。通过在推理轨迹中把规划与执行分离，实现<strong>84% 的 token 削减</strong>，准确率损失极小。</p></li>
<li><p><strong>预算感知提示（budget-aware prompting）</strong>：显式地给模型下达 token 预算指令（“用 200 个 token 以内解决”）。在问题难度已知的基准上，推理成本减半。</p></li>
<li><p><strong>更短的链反而提升准确率</strong>：在某些任务上，强制更短的推理链能把准确率提升 <strong>34.5%</strong>——模型被迫更加直接，避开过度思考的螺旋。</p></li>
<li><p><strong>自适应深度</strong>：OpenAI 的 <code>reasoning_effort</code> 参数（low/medium/high）根据估计难度把查询路由到不同的 token 预算，为算力—质量权衡提供了实用 API。</p></li>
</ul>

## 13.9.3　与 RL 的联系：把推理算力视为 MDP 的时域 {#subsec:inference-mdp-connection}

<div class="admonition tip"><p class="admonition-title">MDP 框架中的推理时算力</p>
<p>推理时算力扩展在本书通篇使用的 RL 框架中有一个自然的解释：</p>

<ul>
<li><p>每个推理 token 都是<strong>MDP 中的一步</strong>。</p></li>
<li><p>分配更多推理算力 = 在相同奖励函数下<strong>更长的时域（horizon）</strong>。</p></li>
<li><p>过度思考问题 = 在代理指标（token 数）而非任务成功上的<strong>reward hacking</strong>。</p></li>
<li><p>高效推理 = 在 MDP 内学习<strong>最优停止策略</strong>。</p></li>
</ul>

<p>这一框架表明，推理时扩展定律并非仅仅是经验上的奇闻——它们反映了 RL 优化景观的根本性质。在推理任务上用 RL 训练的模型，隐式地同时在学习<em>想什么</em>与<em>想多久</em>。</p>
</div>

## 13.9.4　潜空间推理：思维链就是计算本身吗？ {#sec:latent-reasoning}

<p>越来越多的证据挑战“思维链<em>就是</em>推理本身”这一假设：</p>

<ul>
<li><p><strong>超越思维链的推理（Reasoning Beyond Chain-of-Thought）</strong> <a class="book-citation" id="cite-reasoning_latent2025-1" href="../../references.md#ref-reasoning_latent2025" aria-label="参考文献 168">[168]</a>：这项工作用稀疏自编码器（Sparse Autoencoder）探针检查内部表征，识别出在相应 token 生成<em>之前</em>就被激活的推理专用特征——暗示模型在写下思维链之前就已经“知道答案”。</p></li>
<li><p><strong>思考以回忆（Thinking to Recall）</strong> <a class="book-citation" id="cite-thinking_recall2026-1" href="../../references.md#ref-thinking_recall2026" aria-label="参考文献 115">[115]</a>（Google，COLM 2026）：证明生成推理轨迹能够解锁那些直接提示几乎无法触及的参数化知识——但轨迹本身可能只是<em>脚手架</em>，而非计算本身。</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">把思维链视为通信协议</p>
<p>新兴的观点是：思维链可能是模型内部推理回路与输出层之间的<em>通信协议</em>——一种“路由”经过包含答案的潜状态的方式——而非算法本身。这具有实际含义：(1) 验证推理质量不能只靠检查思维链，因为模型可能带着一条具有误导性或事后生成的轨迹而正确推理；(2) 未来的架构可能完全在潜空间中完成推理，只输出结论。</p>
</div>
