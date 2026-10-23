# 12.4　范式对比 {#paradigm-comparison}

<p class="table-caption" id="table-12-1"><span class="table-number">表 12.1　</span>传统 RL 缓冲区与 LLM Agent 缓冲区对比</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>特性</strong></th><th scope="col"><strong>传统 RL 缓冲区</strong></th><th scope="col"><strong>LLM Agent 缓冲区</strong></th></tr></thead>
<tbody><tr><td><strong>数据格式</strong></td><td>连续向量 / Tensor</td><td>Token 化文本、JSON、代码块、工具输出</td></tr>
<tr><td><strong>数据量</strong></td><td>海量（<span class="arithmatex">\(10^5\)</span>–<span class="arithmatex">\(10^7\)</span> 条）</td><td>中小规模（<span class="arithmatex">\(10^3\)</span>–<span class="arithmatex">\(10^5\)</span> 条轨迹）</td></tr>
<tr><td><strong>主要目标</strong></td><td>打破数据相关性</td><td>提供推理示范</td></tr>
<tr><td><strong>采样</strong></td><td>随机均匀 / PER</td><td>语义检索 / 成功优先 / 多样性</td></tr>
<tr><td><strong>状态大小</strong></td><td>固定（如 84<span class="arithmatex">\(\times\)</span>84 像素）</td><td>可变（每个状态 1K–128K Token）</td></tr>
<tr><td><strong>动作空间</strong></td><td>离散/连续向量</td><td>结构化文本（推理 + 工具调用）</td></tr>
<tr><td><strong>Reward 来源</strong></td><td>环境模拟器</td><td>外部执行 / LLM judge / 单元测试</td></tr></tbody>
</table>
</div>
