# 23.6　Skill vs. 微调 {#skills-vs.-fine-tuning}

<p>一个自然的问题：为什么要使用运行时的 skill 注入，而不是对模型进行微调？</p>

<p class="table-caption" id="table-23-2"><span class="table-number">表 23.2　</span>用于添加能力的 Skill（上下文内）与微调对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>Skill（上下文内）</strong></th><th scope="col"><strong>微调</strong></th></tr></thead>
<tbody><tr><td>部署速度</td><td>即时</td><td>数小时–数天</td></tr>
<tr><td>灵活性</td><td>运行时可替换/组合</td><td>训练时固定</td></tr>
<tr><td>上下文成本</td><td>占用上下文窗口</td><td>运行时零成本</td></tr>
<tr><td>深层行为改变</td><td>受上下文长度限制</td><td>深层参数级变化</td></tr>
<tr><td>多租户</td><td>不同用户可用不同 skill</td><td>所有用户共用同一模型</td></tr>
<tr><td>维护</td><td>更新文本文件</td><td>在新数据上重新训练</td></tr></tbody>
</table>
</div>

<p>在实践中，两种方法是互补的：微调提供<em>基础能力</em>（指令跟随、工具使用格式、推理），而 skill 在运行时叠加提供<em>任务特定的专长</em>。</p>
