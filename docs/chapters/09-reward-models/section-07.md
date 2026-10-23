# 9.7　基于列表排序的 Reward {#listwise-rank-based-rewards}

<p>虽然 Bradley-Terry 模型处理的是<em>成对</em>偏好（<span class="arithmatex">\(y_w \succ y_l\)</span>），但许多实际场景需要同时对多个回答进行排序。列表式（listwise）奖励模型从完整排序中学习，提供更丰富的训练信号，并实现更好的校准。</p>

### 动机：超越成对比较 {#motivation-beyond-pairwise}

<div class="admonition tip"><p class="admonition-title">为何要用 listwise?</p>
<ul>
<li><p><strong>更丰富的信号</strong>：对 <span class="arithmatex">\(K\)</span> 个回答的排序包含 <span class="arithmatex">\(\binom{K}{2}\)</span> 个隐式成对比较，同时还能捕捉<em>相对 margin</em>（例如排名第 1 比第 3 好多少）。</p></li>
<li><p><strong>更好的校准</strong>：成对 BT 模型只学习 Reward 的<em>差值</em>；listwise 模型学习的是绝对 Reward 尺度。</p></li>
<li><p><strong>天然契合 GRPO</strong>：GRPO 对每个 prompt 生成 <span class="arithmatex">\(N\)</span> 个回答并排序 —— listwise Reward 与该工作流直接对齐。</p></li>
<li><p><strong>标注效率</strong>：对 5 个回答排序比独立标注所有 10 个可能的成对组合更快。</p></li>
</ul>
</div>

### Plackett-Luce 模型 {#plackett-luce-model}

<p>Plackett-Luce(PL)模型 <a class="book-citation" id="cite-plackett1975analysis-1" href="../../references.md#ref-plackett1975analysis" aria-label="参考文献 291">[291]</a> 是将 Bradley-Terry 推广至完整排序的标准扩展。给定 <span class="arithmatex">\(K\)</span> 个回答 <span class="arithmatex">\(y_1, \ldots, y_K\)</span> 及其排序 <span class="arithmatex">\(\pi\)</span>(其中 <span class="arithmatex">\(\pi(1)\)</span> 为最佳):</p>

<div class="admonition note"><p class="admonition-title">Plackett-Luce 似然</p>


<div class="arithmatex">\[P(\pi \mid q) = \prod_{i=1}^{K} \frac{e^{r_\phi(y_{\pi(i)}, q)}}{\sum_{j=i}^{K} e^{r_\phi(y_{\pi(j)}, q)}}\]</div>

<p><strong>直觉</strong>：顺序地从剩余项中选出最佳。每一步选中 <span class="arithmatex">\(\pi(i)\)</span> 的概率是对剩余项做 Softmax。</p>

<p><strong>Loss 函数</strong>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{PL}}(\phi) = -\frac{1}{|\mathcal{D}|} \sum_{(q, \pi) \in \mathcal{D}} \sum_{i=1}^{K-1} \left[ r_\phi(y_{\pi(i)}, q) - \log \sum_{j=i}^{K} e^{r_\phi(y_{\pi(j)}, q)} \right]\]</div>
</div>

<div class="admonition tip"><p class="admonition-title">Plackett-Luce 退化为 Bradley-Terry</p>
<p>当 <span class="arithmatex">\(K=2\)</span> 时，PL 模型给出：<span class="arithmatex">\(P(y_1 \succ y_2) = \frac{e^{r(y_1)}}{e^{r(y_1)} + e^{r(y_2)}} = \sigma(r(y_1) - r(y_2))\)</span> —— 这正是 Bradley-Terry 模型。PL 是其严格的推广。</p>
</div>

### ListMLE 与基于排序的 Loss {#listmle-and-rank-based-losses}

<div class="admonition note"><p class="admonition-title">列表式 Loss 函数</p>
<ul>
<li><p><strong>ListMLE</strong> <a class="book-citation" id="cite-xia2008listwise-1" href="../../references.md#ref-xia2008listwise" aria-label="参考文献 391">[391]</a>：直接最大化真实排序的 PL 似然。简洁有效。</p></li>
<li><p><strong>ListNet</strong> <a class="book-citation" id="cite-cao2007listnet-1" href="../../references.md#ref-cao2007listnet" aria-label="参考文献 38">[38]</a>：最小化模型的 top-1 概率分布与真实分布之间的 KL 散度：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{ListNet}} = -\sum_{i=1}^{K} P_{\text{true}}(y_i \text{ is best}) \cdot \log P_{\text{model}}(y_i \text{ is best})\]</div>

<p>其中 <span class="arithmatex">\(P_{\text{model}}(y_i \text{ is best}) = \frac{e^{r_\phi(y_i)}}{\sum_j e^{r_\phi(y_j)}}\)</span>。</p></li>
<li><p><strong>LambdaRank</strong> <a class="book-citation" id="cite-burges2006lambdarank-1" href="../../references.md#ref-burges2006lambdarank" aria-label="参考文献 34">[34]</a>：用排序指标（例如 NDCG）的变化对成对 Gradient 加权。当排名靠前的质量更重要时尤为有用。</p></li>
<li><p><strong>RankNet</strong> <a class="book-citation" id="cite-burges2005ranknet-1" href="../../references.md#ref-burges2005ranknet" aria-label="参考文献 33">[33]</a>：对所有成对组合求和的成对交叉熵 —— 等价于在从排序中抽取的所有 <span class="arithmatex">\(\binom{K}{2}\)</span> 个对上应用 BT。</p></li>
</ul>
</div>

### GRPO 与拒绝采样中的 listwise Reward {#listwise-rewards-for-grpo-and-rejection-sampling}

<div class="admonition note"><p class="admonition-title">与 GRPO 的集成</p>
<p>GRPO 天然产生有序的分组：对每个 prompt,<span class="arithmatex">\(N\)</span> 个回答被打分并排序。可以直接在这些排序上训练 listwise 奖励模型：</p>

<ol>
<li><p><strong>生成</strong>：从 Policy 中对每个 prompt 采样 <span class="arithmatex">\(N=8\)</span> 个回答。</p></li>
<li><p><strong>排序</strong>：使用已有的奖励模型（或人工标注者）产生完整排序 <span class="arithmatex">\(\pi\)</span>。</p></li>
<li><p><strong>训练 listwise RM</strong>：在 <span class="arithmatex">\((q, \pi)\)</span> 元组上优化 PL Loss。</p></li>
<li><p><strong>在 GRPO 中使用</strong>：listwise RM 对每个回答分配标量 Reward <span class="arithmatex">\(r(y_i, q)\)</span>；GRPO 将优势计算为 <span class="arithmatex">\(\hat{A}_i = (r_i - \mu) / \sigma\)</span>。</p></li>
</ol>

<p><strong>相对成对方法的优势</strong>：listwise RM 同时看到全部 <span class="arithmatex">\(N\)</span> 个回答，可学到排名第 1 应当比排名第 <span class="arithmatex">\(N\)</span> 拥有高得多的 Reward（而非仅仅“比另一个回答略好”）。</p>
</div>

### 实用考量 {#practical-considerations}

<div class="admonition warning"><p class="admonition-title">Listwise 训练挑战</p>
<ul>
<li><p><strong>标注成本</strong>：完整排序昂贵。部分排序（8 个中取 top-3）能在质量损失很小的前提下降低成本。</p></li>
<li><p><strong>并列</strong>：真实排序常有并列。使用 PL 的并列扩展：对并列项分配相等的概率质量。</p></li>
<li><p><strong>位置偏差</strong>：标注者倾向偏好排在前面的项。随机化展示顺序并训练去偏。</p></li>
<li><p><strong>列表长度</strong>：训练时 <span class="arithmatex">\(K=4\)</span>–8 较为常见。更长的列表（<span class="arithmatex">\(K&gt;16\)</span>）只会增加噪声，收益有限。</p></li>
<li><p><strong>一致性</strong>：不同标注者的排序可能不一致。使用标注者间一致性（<span class="arithmatex">\(\kappa &gt; 0.6\)</span>）作为质量过滤。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">Plackett-Luce 训练代码</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch.nn.functional</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">F</span>

<span class="k">def</span><span class="w"> </span><span class="nf">plackett_luce_loss</span><span class="p">(</span><span class="n">rewards</span><span class="p">,</span> <span class="n">rankings</span><span class="p">):</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    Args:</span>
<span class="sd">        rewards: (batch, K) - 对 K 个回答预测的标量 Reward</span>
<span class="sd">        rankings: (batch, K) - 真实排序索引(0 = 最佳)</span>
<span class="sd">    Returns:</span>
<span class="sd">        标量 Loss</span>
<span class="sd">    """</span>
    <span class="n">batch_size</span><span class="p">,</span> <span class="n">K</span> <span class="o">=</span> <span class="n">rewards</span><span class="o">.</span><span class="n">shape</span>
    <span class="c1"># 按真实排序顺序对 Reward 排序</span>
    <span class="n">sorted_rewards</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">gather</span><span class="p">(</span><span class="n">rewards</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="n">rankings</span><span class="p">)</span>  <span class="c1"># (batch, K)</span>

    <span class="c1"># PL 对数似然:对各位置求和</span>
    <span class="n">loss</span> <span class="o">=</span> <span class="mf">0.0</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">K</span> <span class="o">-</span> <span class="mi">1</span><span class="p">):</span>
        <span class="c1"># 对剩余项(位置 i 到 K)做 log-softmax</span>
        <span class="n">remaining</span> <span class="o">=</span> <span class="n">sorted_rewards</span><span class="p">[:,</span> <span class="n">i</span><span class="p">:]</span>           <span class="c1"># (batch, K-i)</span>
        <span class="n">log_probs</span> <span class="o">=</span> <span class="n">remaining</span><span class="p">[:,</span> <span class="mi">0</span><span class="p">]</span> <span class="o">-</span> <span class="n">torch</span><span class="o">.</span><span class="n">logsumexp</span><span class="p">(</span><span class="n">remaining</span><span class="p">,</span> <span class="n">dim</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
        <span class="n">loss</span> <span class="o">-=</span> <span class="n">log_probs</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>

    <span class="k">return</span> <span class="n">loss</span> <span class="o">/</span> <span class="p">(</span><span class="n">K</span> <span class="o">-</span> <span class="mi">1</span><span class="p">)</span>

<span class="c1"># 示例:每个 prompt 8 个回答,由标注者排序</span>
<span class="n">rewards</span> <span class="o">=</span> <span class="n">reward_model</span><span class="p">(</span><span class="n">responses</span><span class="p">)</span>          <span class="c1"># (batch, 8)</span>
<span class="n">rankings</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">argsort</span><span class="p">(</span><span class="n">human_scores</span><span class="p">,</span> <span class="n">descending</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>  <span class="c1"># 最佳在前</span>
<span class="n">loss</span> <span class="o">=</span> <span class="n">plackett_luce_loss</span><span class="p">(</span><span class="n">rewards</span><span class="p">,</span> <span class="n">rankings</span><span class="p">)</span>
<span class="n">loss</span><span class="o">.</span><span class="n">backward</span><span class="p">()</span>
</code></pre></div>
</div>
