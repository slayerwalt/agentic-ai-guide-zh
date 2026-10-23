# 7.4　组大小分析 {#group-size-analysis}

<span id="table-7-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><span class="arithmatex">\(G\)</span></th><th scope="col"><strong>信号质量</strong></th><th scope="col"><strong>算力</strong></th><th scope="col"><strong>使用场景</strong></th></tr></thead>
<tbody><tr><td>2</td><td>极嘈杂（掷硬币）</td><td>低</td><td>不推荐 —— 太嘈杂，无法稳定学习</td></tr>
<tr><td>4</td><td>中等</td><td>中等</td><td>快速实验、简单任务（通过率 <span class="arithmatex">\(&gt;\)</span> 50%）</td></tr>
<tr><td>8</td><td>良好（标准）</td><td>高</td><td>默认。适合大多数任务的良好平衡</td></tr>
<tr><td>16</td><td>极佳</td><td>非常高</td><td>困难任务（通过率 <span class="arithmatex">\(&lt;\)</span> 20%），需要多次尝试才能拿到正样本</td></tr>
<tr><td>32</td><td>近乎完美</td><td>极高</td><td>仅在算力充裕且任务非常困难时使用</td></tr></tbody>
</table>
</div>

<div class="admonition note"><p class="admonition-title">关键：组内必须同时包含成功与失败</p>
<p>若 <span class="arithmatex">\(G\)</span> 条响应全对（<span class="arithmatex">\(r_i = 1 \;\forall i\)</span>）：所有 advantage = 0，无学习信号！<br>若全错：同样的问题。Prompt 难度必须与模型能力匹配。<br><strong>Goldilocks 原则</strong>：将 prompt 过滤至当前模型 20–80% 通过率。随着模型改进，每 500 步重新筛选。</p>
</div>
