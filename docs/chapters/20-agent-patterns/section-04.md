# 20.4　模式选择指南 {#pattern-selection-guide}

<p>选择正确的模式取决于三个因素：(1) 任务结构的可预测性如何，(2) 延迟和成本上能承担多少次 LLM 调用，(3) 质量是否需要迭代。可将下表作为决策矩阵——从顶部（最简模式）开始，仅当较简单的模式确实失败时才向下移动。</p>

<p class="table-caption" id="table-20-3"><span class="table-number">表 20.3　</span>各 Agent 设计模式的适用场景</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>复杂度</strong></th><th scope="col"><strong>LLM 调用次数</strong></th><th scope="col"><strong>最适用于</strong></th></tr></thead>
<tbody><tr><td>Prompt 链</td><td>低</td><td><span class="arithmatex">\(N\)</span>（固定）</td><td>顺序任务、内容流水线</td></tr>
<tr><td>路由</td><td>低</td><td>1 + 1</td><td>多类型输入、分诊</td></tr>
<tr><td>并行化</td><td>低</td><td><span class="arithmatex">\(N\)</span>（并行）</td><td>独立子任务、投票</td></tr>
<tr><td>Orchestrator-workers</td><td>中</td><td>可变</td><td>未知的分解结构</td></tr>
<tr><td>Evaluator-optimizer</td><td>中</td><td>2–10（循环）</td><td>对质量要求高的输出</td></tr>
<tr><td>ReAct</td><td>中</td><td>3–25（循环）</td><td>通用工具使用、探索</td></tr>
<tr><td>规划型 Agent</td><td>高</td><td>5–50+</td><td>长时程、多步任务</td></tr>
<tr><td>反思</td><td>高</td><td>+50% 开销</td><td>首次尝试常失败的任务</td></tr>
<tr><td>多 Agent</td><td>高</td><td>多次</td><td>复杂领域、专业化</td></tr></tbody>
</table>
</div>

<p>这些模式是可组合的：一个规划型 Agent 可以在各步骤内使用 Prompt 链，在评审阶段使用 Evaluator-optimizer，并使用路由将子任务分派给专家。艺术在于知道何时停止增加层次。</p>
