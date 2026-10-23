# 5.9　关键超参 {#critical-hyperparameters}

<span id="table-5-inline-2"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>参数</strong></th><th scope="col"><strong>典型值</strong></th><th scope="col"><strong>设错的影响</strong></th></tr></thead>
<tbody><tr><td><code>cliprange</code></td><td>0.2</td><td>过低：学不到东西。过高：不稳定。</td></tr>
<tr><td><code>init_kl_coef</code></td><td>0.01–0.1</td><td>过低：reward hacking。过高：卡在 SFT 状态。</td></tr>
<tr><td><code>target_kl</code></td><td>4–8</td><td>自适应控制器的目标值；越低越保守。</td></tr>
<tr><td><code>ppo_epochs</code></td><td>4</td><td>过多：在单 batch 上过拟合。过少：浪费生成算力。</td></tr>
<tr><td><code>learning_rate</code></td><td><span class="arithmatex">\(1{-}5 \times 10^{-6}\)</span></td><td>过高：灾难性遗忘（Catastrophic Forgetting）。</td></tr>
<tr><td><code>batch_size</code></td><td>64–256</td><td>越大：梯度更平滑，但生成算力更多。</td></tr>
<tr><td><code>temperature</code></td><td>0.7–1.0</td><td>越低：探索越少。越高：advantage 越嘈杂。</td></tr></tbody>
</table>
</div>
