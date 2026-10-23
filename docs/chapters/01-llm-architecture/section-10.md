# 1.10　专家混合模型（Mixture of Experts，MoE） {#mixture-of-experts-moe}

<p>专家混合模型 <a class="book-citation" id="cite-shazeer2017outrageously-1" href="../../references.md#ref-shazeer2017outrageously" aria-label="参考文献 325">[325]</a><a class="book-citation" id="cite-jiang2024mixtral-1" href="../../references.md#ref-jiang2024mixtral" aria-label="参考文献 166">[166]</a> 通过对每个 Token 只激活参数的一个子集，从而在不按比例增加计算成本的情况下扩展模型容量。</p>

## 1.10.1　架构 {#architecture}

<div class="admonition note"><p class="admonition-title">MoE 层</p>
<p>在一个 MoE Transformer 中，每个块中的 FFN 层被替换为 <span class="arithmatex">\(N\)</span> 个并行的“专家” FFN，再加上一个 <strong>路由器（router）</strong> 来选择使用哪些专家：</p>



<div class="arithmatex">\[\text{MoE}(x) = \sum_{i=1}^{N} g_i(x) \cdot E_i(x), \quad g(x) = \text{TopK}(\text{softmax}(W_r x))\]</div>

<ul>
<li><p><span class="arithmatex">\(E_i\)</span> 是专家网络（标准的 FFN 层）</p></li>
<li><p><span class="arithmatex">\(g_i(x)\)</span> 是来自路由器的门控权重（只有 top-<span class="arithmatex">\(K\)</span> 个非零）</p></li>
<li><p>通常每个 Token 在 <span class="arithmatex">\(N=8\)</span>–64 个专家中激活 <span class="arithmatex">\(K=2\)</span> 个</p></li>
<li><p>总参数量随 <span class="arithmatex">\(N\)</span> 增长；<strong>激活参数量</strong> 按 FFN 大小的 <span class="arithmatex">\(K/N\)</span> 比例增长</p></li>
</ul>
</div>

<figure id="figure-1-11"><a href="../../assets/figures/fig_012_fig12.png" target="_blank" rel="noopener" aria-label="查看图 1.11 原图" title="点击查看原图"><img src="../../assets/figures/fig_012_fig12.png" alt="具有 8 个专家和 Top-2 路由的 MoE 层。每个 Token 只计算门控值最高的两个专家；其余专家被完全跳过。" loading="lazy"></a><figcaption><span class="figure-number">图 1.11　</span>具有 8 个专家和 Top-2 路由的 MoE 层。每个 Token 只计算门控值最高的两个专家；其余专家被完全跳过。</figcaption></figure>

## 1.10.2　负载均衡 {#load-balancing}

<div class="admonition warning"><p class="admonition-title">负载均衡问题</p>
<p>在没有约束的情况下，路由器可能会把大多数 Token 都发送到相同的 1–2 个专家上（“专家坍缩（expert collapse）”）。这既浪费了容量，又造成了跨 GPU 的计算不均衡（每个专家通常位于不同的 GPU 上）。</p>

<p><strong>解决方案</strong>：添加一个辅助的负载均衡损失：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{bal}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot p_i\]</div>

<p>其中 <span class="arithmatex">\(f_i\)</span> 表示路由到专家 <span class="arithmatex">\(i\)</span> 的 Token 比例，<span class="arithmatex">\(p_i\)</span> 表示专家 <span class="arithmatex">\(i\)</span> 的平均路由概率。这鼓励了对专家的均匀利用。</p>
</div>

## 1.10.3　噪声 Top-K 门控（Noisy Top-K Gating）：让离散路由变得可训练 {#noisy-top-k-gating-making-discrete-routing-trainable}

<p>MoE 的核心挑战在于 <strong>top-<span class="arithmatex">\(k\)</span> 选择是不可微的</strong>——你无法通过一个硬性的“挑选前 2 名”操作反向传播。该领域发展出了两个关键技巧来解决这个问题：</p>

<div class="admonition tip"><p class="admonition-title">路由可微性问题</p>
<p>路由器为每个专家计算 logits <span class="arithmatex">\(h(x) = W_r \cdot x\)</span>，然后选择 top-<span class="arithmatex">\(k\)</span>。但是：</p>

<ul>
<li><p><em>被选中</em> 的专家会通过其门控权重获得梯度（在被选中者上做 softmax）</p></li>
<li><p><em>选择决策本身</em>（挑选哪 <span class="arithmatex">\(k\)</span> 个）的梯度为零</p></li>
<li><p>如果没有技巧，路由器可能会卡住：一个从未被选中的专家 <span class="arithmatex">\(\rightarrow\)</span> 永远得不到梯度信号 <span class="arithmatex">\(\rightarrow\)</span> 永远不会被选中</p></li>
</ul>
</div>

#### 方法 1：噪声 Top-K 门控（Noisy Top-K Gating） <a class="book-citation" id="cite-shazeer2017outrageously-2" href="../../references.md#ref-shazeer2017outrageously" aria-label="参考文献 325">[325]</a>。 {#approach-1-noisy-top-k-gating-.}

<p>在进行 top-<span class="arithmatex">\(k\)</span> 选择 <em>之前</em>，向路由器的 logits 添加可学习的高斯噪声：</p>



<div class="arithmatex">\[\begin{align*}
h(x) &amp;= W_g \cdot x \tag{clean logits} \\

  H(x) &amp;= h(x) + \epsilon \cdot \text{Softplus}(W_{\text{noise}} \cdot x), \quad \epsilon \sim \mathcal{N}(0, 1) \tag{noisy logits} \\

  \text{TopK}(v, k)_i &amp;= \begin{cases} v_i &amp; \text{if } v_i \text{ is in the top } k \\ -\infty &amp; \text{otherwise} \end{cases} \tag{1.11}\\

  g(x) &amp;= \text{softmax}\big(\text{TopK}(H(x),\, k)\big) \tag{sparse gates}
\end{align*}\]</div>

<ul>
<li><p><span class="arithmatex">\(W_{\text{noise}}\)</span> 是 <em>学习得到的</em> 噪声幅度——模型会学习每个专家需要多少探索量</p></li>
<li><p>在训练过程中，噪声偶尔会将“弱势”专家提升到 top-<span class="arithmatex">\(k\)</span> 中，使其获得梯度信号</p></li>
<li><p>在推理时去除噪声：使用干净的 logits <span class="arithmatex">\(h(x)\)</span> 进行确定性路由</p></li>
<li><p>Softplus 确保噪声尺度始终为正</p></li>
</ul>

#### 方法 2：Gumbel-Softmax 技巧（Gumbel-Softmax Trick，用于可微的离散采样）。 {#approach-2-gumbel-softmax-trick-for-differentiable-discrete-sampling.}

<p>来自变分推断文献的另一种方法 <a class="book-citation" id="cite-jang2017categorical-1" href="../../references.md#ref-jang2017categorical" aria-label="参考文献 161">[161]</a>。<strong>Gumbel-Max 技巧（Gumbel-Max trick）</strong> 提供了从类别分布中精确采样的方式：</p>



<div class="arithmatex">\[z = \arg\max_i \left[ \log \pi_i + G_i \right], \quad G_i \sim \text{Gumbel}(0,1)
\tag{1.12}\]</div>

<p>其中 Gumbel 噪声由 <span class="arithmatex">\(G_i = -\log(-\log(U_i)),\; U_i \sim \text{Uniform}(0,1)\)</span> 生成。</p>

<p>对于 <strong>top-<span class="arithmatex">\(k\)</span> 路由</strong>：取 <span class="arithmatex">\((\log \pi_i + G_i)\)</span> 的 top-<span class="arithmatex">\(k\)</span>，等价于从由 <span class="arithmatex">\(\pi\)</span> 定义的类别分布中 <em>无放回地</em> 抽取 <span class="arithmatex">\(k\)</span> 个样本。</p>

<p>由于 <span class="arithmatex">\(\arg\max\)</span> 是不可微的，<strong>Gumbel-Softmax</strong> 松弛将其替换为一个由温度控制的 softmax：</p>



<div class="arithmatex">\[\hat{g}_i = \frac{\exp\left((\log \pi_i + G_i) / \tau\right)}{\sum_j \exp\left((\log \pi_j + G_j) / \tau\right)}
\tag{1.13}\]</div>

<ul>
<li><p><span class="arithmatex">\(\tau \to 0\)</span>：趋近于硬 one-hot（精确但不可微）</p></li>
<li><p><span class="arithmatex">\(\tau \to \infty\)</span>：趋近于均匀分布（可微但无信息量）</p></li>
<li><p>实践中，在训练过程中将 <span class="arithmatex">\(\tau\)</span> 从 1.0 退火到 0.1–0.5</p></li>
<li><p><strong>直通估计器（Straight-through estimator）</strong>：前向传播中使用硬 top-<span class="arithmatex">\(k\)</span>，反向传播中使用 Gumbel-Softmax 梯度——两全其美</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">实际中使用的是哪种方法？</p>
<ul>
<li><p><strong>Sparsely-Gated MoE <a class="book-citation" id="cite-shazeer2017outrageously-3" href="../../references.md#ref-shazeer2017outrageously" aria-label="参考文献 325">[325]</a>、Mixtral <a class="book-citation" id="cite-jiang2024mixtral-2" href="../../references.md#ref-jiang2024mixtral" aria-label="参考文献 166">[166]</a>、DeepSeek-V2 <a class="book-citation" id="cite-deepseekv2-1" href="../../references.md#ref-deepseekv2" aria-label="参考文献 69">[69]</a></strong>：使用带高斯噪声的噪声 Top-K。简单、有效，在大规模上得到了充分验证。</p></li>
<li><p><strong>Switch Transformer <a class="book-citation" id="cite-fedus2022switch-1" href="../../references.md#ref-fedus2022switch" aria-label="参考文献 89">[89]</a></strong>：简化为不带噪声的 Top-1（仅依赖负载均衡损失）。</p></li>
<li><p><strong>研究型 / 小规模 MoE</strong>：一些工作使用 Gumbel-Softmax 实现完全可微的路由，尤其是当学习路由本身就是研究目标时。</p></li>
<li><p><strong>关键见解</strong>：两种方法都通过噪声注入解决了同一个问题（让离散选择可训练）。高斯噪声更简单；Gumbel 噪声在类别采样上有更强的理论保证。</p></li>
</ul>

<h4 id="heading-1-140">无辅助损失负载均衡。</h4>

<p>标准做法是在训练损失上加一个均衡惩罚项，但 DeepSeek <a class="book-citation" id="cite-wang2024auxfree-1" href="../../references.md#ref-wang2024auxfree" aria-label="参考文献 372">[372]</a> 证明这会与路由信号相互对抗：损失把利用率推向均匀，而路由器想要专业化。他们的解决方案把均衡完全移出梯度，改为一个基于观测利用率更新的逐专家路由<em>偏置（bias）</em>——同时实现了更好的质量<em>和</em>更高的专家专业化。Qwen 的后续洞见进一步揭示：<em>聚合范围</em>比机制本身更重要——按微批次（常见默认）计算均衡统计会悄无声息地摧毁专业化，因为微批次过于同质。像 MAI-Thinking-1 所证实的那样，在整个全局批次上聚合，才能恢复专家分化所需的多样性信号。</p>
</div>

## 1.10.4　值得关注的 MoE 模型 {#notable-moe-models}

<span id="table-1-inline-20"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模型</strong></th><th scope="col"><strong>总参数量</strong></th><th scope="col"><strong>激活参数量</strong></th><th scope="col"><strong>专家</strong></th><th scope="col"><strong>创新点</strong></th></tr></thead>
<tbody><tr><td>Switch Transformer <a class="book-citation" id="cite-fedus2022switch-2" href="../../references.md#ref-fedus2022switch" aria-label="参考文献 89">[89]</a></td><td>1.6T</td><td>100B</td><td>128, Top-1</td><td>首个大规模 MoE；简化的路由</td></tr>
<tr><td>Mixtral 8x7B <a class="book-citation" id="cite-jiang2024mixtral-3" href="../../references.md#ref-jiang2024mixtral" aria-label="参考文献 166">[166]</a></td><td>47B</td><td>13B</td><td>8, Top-2</td><td>开放权重；质量媲美 Llama-2 70B</td></tr>
<tr><td>DeepSeek-V2 <a class="book-citation" id="cite-deepseekv2-2" href="../../references.md#ref-deepseekv2" aria-label="参考文献 69">[69]</a></td><td>236B</td><td>21B</td><td>160, Top-6</td><td>带共享 + 路由专家的 DeepSeekMoE</td></tr>
<tr><td>Qwen-MoE <a class="book-citation" id="cite-qwen2024qwen25-1" href="../../references.md#ref-qwen2024qwen25" aria-label="参考文献 350">[350]</a></td><td>14.3B</td><td>2.7B</td><td>60, Top-4</td><td>为提升效率而设计的细粒度专家</td></tr>
<tr><td>DBRX <a class="book-citation" id="cite-databricks2024dbrx-1" href="../../references.md#ref-databricks2024dbrx" aria-label="参考文献 67">[67]</a></td><td>132B</td><td>36B</td><td>16, Top-4</td><td>每块 4 个专家的细粒度结构</td></tr></tbody>
</table>
</div>
