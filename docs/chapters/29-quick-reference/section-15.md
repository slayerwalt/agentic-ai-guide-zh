# 29.15　推理与测试时扩展 {#reasoning-test-time-scaling}

<span id="table-29-inline-10"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>算力开销</strong></th><th scope="col"><strong>机制</strong></th></tr></thead>
<tbody><tr><td>思维链（Chain-of-Thought，CoT）</td><td>1.5–3<span class="arithmatex">\(\times\)</span> token</td><td>在 prompt 中加入 “step by step”</td></tr>
<tr><td>自洽性（Self-Consistency）</td><td><span class="arithmatex">\(N \times\)</span> 生成</td><td>采样 <span class="arithmatex">\(N\)</span> 条 CoT 路径，对最终答案多数表决</td></tr>
<tr><td>思维树（Tree-of-Thought, ToT）</td><td><span class="arithmatex">\(B \times D \times\)</span> 生成</td><td>在推理树上 BFS/DFS；对分支评估</td></tr>
<tr><td>Best-of-<span class="arithmatex">\(N\)</span></td><td><span class="arithmatex">\(N \times\)</span> 生成</td><td>采样 <span class="arithmatex">\(N\)</span>，用 RM 打分，取最高</td></tr>
<tr><td>Beam search（推理上）</td><td><span class="arithmatex">\(B \times\)</span> 生成</td><td>维持 top-<span class="arithmatex">\(B\)</span> 个部分推理链</td></tr>
<tr><td>预算强制（Budget forcing）</td><td>可变</td><td>动态为更难问题分配更多 token</td></tr>
<tr><td>验证（ORM/PRM）</td><td><span class="arithmatex">\(N \times\)</span> 生成 + 打分</td><td>生成 <span class="arithmatex">\(N\)</span> 个解，按结果/过程 RM 排序</td></tr></tbody>
</table>
</div>
