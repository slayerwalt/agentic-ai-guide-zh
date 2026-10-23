# 6.7　<span class="arithmatex">\(\beta\)</span> 选择指南 {#beta-selection-guide}

<span id="table-6-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><span class="arithmatex">\(\beta\)</span></th><th scope="col"><strong>风格</strong></th><th scope="col"><strong>使用场景</strong></th></tr></thead>
<tbody><tr><td>0.01</td><td>非常激进</td><td>仅当数据极其干净、且你需要大幅度分布迁移时使用</td></tr>
<tr><td>0.05</td><td>激进</td><td>数据较好，希望相对 SFT 取得明显改进</td></tr>
<tr><td>0.1</td><td>标准</td><td>默认起点。质量与稳定性的良好平衡</td></tr>
<tr><td>0.2</td><td>保守</td><td>噪声较多，或模型已接近期望行为</td></tr>
<tr><td>0.5</td><td>非常保守</td><td>安全微调场景，必须保住已有能力</td></tr></tbody>
</table>
</div>
