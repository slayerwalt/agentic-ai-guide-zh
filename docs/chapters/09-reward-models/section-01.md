# 9.1　Bradley-Terry 模型 —— 完整推导 {#sec:bradley-terry}

<p>Bradley-Terry 模型 <a class="book-citation" id="cite-bradley1952rank-1" href="../../references.md#ref-bradley1952rank" aria-label="参考文献 29">[29]</a> 是成对偏好学习（pairwise preference learning）的标准概率框架。给定对 prompt <span class="arithmatex">\(q\)</span> 的两个回答 <span class="arithmatex">\(y_1\)</span> 和 <span class="arithmatex">\(y_2\)</span>，该模型假设：</p>



<div class="arithmatex">\[P(y_1 \succ y_2 \mid q) = \sigma(r(y_1, q) - r(y_2, q))
    = \frac{e^{r(y_1,q)}}{e^{r(y_1,q)} + e^{r(y_2,q)}},\]</div>

<p>其中 <span class="arithmatex">\(r: \mathcal{Y} \times \mathcal{Q} \to \mathbb{R}\)</span> 是标量 Reward 函数，<span class="arithmatex">\(\sigma\)</span> 是 sigmoid 函数。</p>

### 最大似然估计 {#maximum-likelihood-estimation}

<p>给定偏好对数据集 <span class="arithmatex">\(\mathcal{D} = \{(q^{(k)}, y_w^{(k)}, y_l^{(k)})\}_{k=1}^N\)</span>，最大似然估计（MLE）目标为：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{BT}}(\phi) =
    -\frac{1}{N}\sum_{k=1}^N \log \sigma\!\bigl(r_\phi(y_w^{(k)}, q^{(k)}) - r_\phi(y_l^{(k)}, q^{(k)})\bigr),\]</div>

<p>其中 <span class="arithmatex">\(r_\phi\)</span> 是由 <span class="arithmatex">\(\phi\)</span> 参数化的神经网络。这是一个二元交叉熵 Loss，其中“正”类别对应偏好的回答。</p>

<div class="admonition note"><p class="admonition-title">Bradley-Terry 假设</p>
<ol>
<li><p>偏好具有<em>传递性</em>：若 <span class="arithmatex">\(y_1 \succ y_2\)</span> 且 <span class="arithmatex">\(y_2 \succ y_3\)</span>，则 <span class="arithmatex">\(y_1 \succ y_3\)</span>。</p></li>
<li><p>偏好由<em>标量</em>Reward 决定（不存在多维偏好）。</p></li>
<li><p>偏好概率仅依赖于 Reward 的<em>差值</em>。</p></li>
<li><p>各偏好对之间相互<em>独立</em>（不存在标注者效应）。</p></li>
</ol>

<p>这些假设在实践中常被违反，这催生了一些扩展模型，例如用于排序的 Plackett-Luce 模型和多维奖励模型。</p>
</div>

### Margin Loss 扩展 {#margin-loss-extension}

<p>一种常见的扩展是引入 margin <span class="arithmatex">\(m\)</span>，以确保胜出与落败 Reward 之间存在最小间隔：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{margin}} =
    -\frac{1}{N}\sum_{k=1}^N \log \sigma\!\bigl(r_\phi(y_w^{(k)}, q^{(k)}) - r_\phi(y_l^{(k)}, q^{(k)}) - m\bigr).\]</div>
