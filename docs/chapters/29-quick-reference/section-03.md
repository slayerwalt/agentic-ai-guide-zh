# 29.3　解码方法 {#decoding-methods}

<span id="table-29-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>公式 / 规则</strong></th><th scope="col"><strong>关键参数</strong></th></tr></thead>
<tbody><tr><td>Greedy</td><td><span class="arithmatex">\(y_t = \arg\max_v P(v|y_{&lt;t})\)</span></td><td>—</td></tr>
<tr><td>Beam search</td><td>按联合概率保留 top-<span class="arithmatex">\(B\)</span> 个部分序列</td><td><span class="arithmatex">\(B=4\)</span>–<span class="arithmatex">\(8\)</span></td></tr>
<tr><td>Temperature</td><td><span class="arithmatex">\(P&#x27;(v) = \text{softmax}(\text{logit}_v / T)\)</span></td><td><span class="arithmatex">\(T \in [0.1, 1.5]\)</span></td></tr>
<tr><td>Top-<span class="arithmatex">\(k\)</span></td><td>仅保留 top-<span class="arithmatex">\(k\)</span> 个 logit，其余置零后重新归一化</td><td><span class="arithmatex">\(k=40\)</span>–<span class="arithmatex">\(100\)</span></td></tr>
<tr><td>Top-<span class="arithmatex">\(p\)</span> (nucleus)</td><td>保留最小集合 <span class="arithmatex">\(V&#x27;\)</span> 使 <span class="arithmatex">\(\sum_{v \in V&#x27;} P(v) \geq p\)</span></td><td><span class="arithmatex">\(p=0.9\)</span>–<span class="arithmatex">\(0.95\)</span></td></tr>
<tr><td>Min-<span class="arithmatex">\(p\)</span></td><td>保留满足 <span class="arithmatex">\(P(v) \geq p_\text{min} \cdot P(v_\text{max})\)</span> 的 token</td><td><span class="arithmatex">\(p_\text{min}=0.05\)</span>–<span class="arithmatex">\(0.1\)</span></td></tr>
<tr><td>Repetition penalty</td><td>若 <span class="arithmatex">\(v\)</span> 已出现，则 <span class="arithmatex">\(\text{logit}_v \leftarrow \text{logit}_v / \theta\)</span></td><td><span class="arithmatex">\(\theta=1.1\)</span>–<span class="arithmatex">\(1.3\)</span></td></tr></tbody>
</table>
</div>
