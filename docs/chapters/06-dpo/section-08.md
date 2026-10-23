# 6.8　DPO 的 Batch Size 配置与扩展 {#dpo-batch-size-configuration-and-scaling}

<p>与在单序列 token 预测上工作的标准 SFT 不同，DPO 使用一个比较“被偏好序列与被拒序列”的<strong>成对 loss</strong>。这从根本上改变了显存利用与优化稳定性。</p>

## 6.8.1　全局 Batch Size 目标 {#global-batch-size-target}

<p>跨多种 DPO 实现的经验证据给出了一个最佳全局 batch size 范围：</p>



<div class="arithmatex">\[\boxed{B_{\text{global}} \in [32, 128]}
\tag{6.12}\]</div>

<ul>
<li><p><span class="arithmatex">\(B_{\text{global}} &lt; 32\)</span>：隐式 reward 估计中的 gradient 噪声严重 <span class="arithmatex">\(\rightarrow\)</span> policy 在多个对齐目标之间（如有用 vs 安全）破坏性振荡。</p></li>
<li><p><span class="arithmatex">\(B_{\text{global}} &gt; 128\)</span>：收敛速度边际收益递减；分布式算力间通信开销巨大。</p></li>
</ul>

## 6.8.2　数学分解 {#mathematical-decomposition}

<p>由于 DPO 同时加载<strong>两份</strong>模型副本（活动 policy <span class="arithmatex">\(\pi_\theta\)</span> + 冻结 reference <span class="arithmatex">\(\pi_{\text{ref}}\)</span>），每条序列的显存翻倍。全局 batch size 可分解为：</p>



<div class="arithmatex">\[\boxed{B_{\text{global}} = B_{\text{micro}} \times N_{\text{GPUs}} \times K_{\text{accum}}}
\tag{6.13}\]</div>

<ul>
<li><p><span class="arithmatex">\(B_{\text{micro}}\)</span>：每设备 micro-batch 大小（每次 forward pass 的偏好对数）。</p></li>
<li><p><span class="arithmatex">\(N_{\text{GPUs}}\)</span>：并行处理数据的设备数量。</p></li>
<li><p><span class="arithmatex">\(K_{\text{accum}}\)</span>：在权重更新前累计 gradient 的步数。</p></li>
</ul>

<p><strong>成对倍数因子</strong>：单个 DPO 数据样本包含 prompt（<span class="arithmatex">\(x\)</span>）、被选（<span class="arithmatex">\(y_w\)</span>）和被拒（<span class="arithmatex">\(y_l\)</span>）。每个 micro-batch 的实际 tensor 负载为：</p>



<div class="arithmatex">\[T_{\text{sequences}} = 2 \times B_{\text{micro}}
\tag{6.14}\]</div>

<p>对于在 80GB GPU 上、上下文长度 4096–8192 token 的 <span class="arithmatex">\(&gt;\)</span>7B 参数模型，物理上限被刚性约束在 <span class="arithmatex">\(B_{\text{micro}} \in [1, 2]\)</span>。</p>

## 6.8.3　分布式扩展配置 {#distributed-scaling-configurations}

<p class="table-caption" id="table-6-1"><span class="table-number">表 6.1　</span>DPO 训练的分布式扩展配置（目标 <span class="arithmatex">\(B_{\text{global}} = 64\)</span>）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>配置</strong></th><th scope="col"><strong>单 GPU</strong></th><th scope="col"><strong>8 卡节点</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(B_{\text{global}}\)</span></td><td>64</td><td>64</td></tr>
<tr><td><span class="arithmatex">\(B_{\text{micro}}\)</span></td><td>2（4 条序列）</td><td>2（4 条序列）</td></tr>
<tr><td><span class="arithmatex">\(N_{\text{GPUs}}\)</span></td><td>1</td><td>8</td></tr>
<tr><td><span class="arithmatex">\(K_{\text{accum}}\)</span></td><td>32 步</td><td>4 步</td></tr>
<tr><td>吞吐</td><td>串行/慢</td><td>高并行吞吐</td></tr></tbody>
</table>
</div>

## 6.8.4　显存优化：预计算 Reference 对数概率 {#vram-optimization-pre-computing-reference-log-probabilities}

<p>DPO loss：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
\tag{6.15}\]</div>

<p>由于 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 在整个训练过程中<strong>完全静态</strong>，其输出可被预先计算：</p>

<div class="admonition note"><p class="admonition-title">Reference 模型驱逐策略</p>
<ol>
<li><p>训练开始前，仅用 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 对数据集 <span class="arithmatex">\(\mathcal{D}\)</span> 执行一次 forward pass。</p></li>
<li><p>将标量 <span class="arithmatex">\(\log \pi_{\text{ref}}(y_w|x)\)</span> 和 <span class="arithmatex">\(\log \pi_{\text{ref}}(y_l|x)\)</span> 缓存到磁盘。</p></li>
<li><p><strong>把 <span class="arithmatex">\(\pi_{\text{ref}}\)</span> 完全从 GPU 显存中驱逐。</strong></p></li>
</ol>

<p><strong>效果</strong>：可用 GPU 显存翻倍 <span class="arithmatex">\(\rightarrow\)</span> <span class="arithmatex">\(B_{\text{micro}}\)</span> 可从 1–2 提升到 4–8，从而最大化硬件利用率与训练吞吐。</p>

<p><em>实现</em>：在 TRL 中，于 <code>DPOConfig</code> 设置 <code>precompute_ref_log_probs=True</code>。对 70B 模型，这能在整个集群上节省约 140GB 的 GPU 显存。</p>
</div>
