# 29.22　智能体评估指标 {#agent-evaluation-metrics}

<span id="table-29-inline-16"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>公式 / 定义</strong></th><th scope="col"><strong>目标</strong></th></tr></thead>
<tbody><tr><td>任务成功率（TSR）</td><td>正确完成数 / 总任务数</td><td><span class="arithmatex">\(&gt;85\%\)</span>（生产）</td></tr>
<tr><td>完成步数</td><td>单次成功任务的平均动作数</td><td>越低越高效</td></tr>
<tr><td>单任务成本</td><td>总 token 数 <span class="arithmatex">\(\times\)</span> 单价</td><td>视预算而定</td></tr>
<tr><td>延迟（TTFC）</td><td>从请求到首条有效输出</td><td>交互场景 <span class="arithmatex">\(&lt;5\)</span> 秒</td></tr>
<tr><td>工具调用准确率</td><td>正确工具选择 / 总调用</td><td><span class="arithmatex">\(&gt;90\%\)</span></td></tr>
<tr><td>恢复率</td><td>重试成功 / 初次失败</td><td><span class="arithmatex">\(&gt;60\%\)</span></td></tr>
<tr><td>人工升级率</td><td>需要人工的任务 / 总任务</td><td><span class="arithmatex">\(&lt;15\%\)</span></td></tr></tbody>
</table>
</div>
