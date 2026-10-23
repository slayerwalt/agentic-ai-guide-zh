# 25.10　架构对比 {#subsec:mas-comparison}

<p class="table-caption" id="table-25-4"><span class="table-number">表 25.4　</span>多 Agent 架构模式在关键维度上的对比。评级：<strong>H</strong>（高）/ <strong>M</strong>（中）/ <strong>L</strong>（低）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>架构</strong></th><th scope="col"><strong>可扩展性</strong></th><th scope="col"><strong>可调试性</strong></th><th scope="col"><strong>协调成本</strong></th><th scope="col"><strong>容错性</strong></th><th scope="col"><strong>最适用场景</strong></th></tr></thead>
<tbody><tr><td>集中式（Supervisor）</td><td>M</td><td>H</td><td>L</td><td>L</td><td>简单流水线；任务分解清晰；小团队</td></tr>
<tr><td>去中心化（P2P）</td><td>H</td><td>L</td><td>H</td><td>H</td><td>动态环境；对韧性要求高；大规模</td></tr>
<tr><td>层级式</td><td>H</td><td>M</td><td>M</td><td>M</td><td>企业级工作流；跨多个领域的复杂任务</td></tr>
<tr><td>Swarm</td><td>H</td><td>L</td><td>L</td><td>H</td><td>客户服务路由；简单的 handoff 链</td></tr>
<tr><td>Pipeline</td><td>M</td><td>H</td><td>L</td><td>L</td><td>顺序处理；阶段依赖清晰</td></tr>
<tr><td>Ensemble</td><td>L</td><td>H</td><td>H</td><td>H</td><td>高风险决策；可靠性优先于效率</td></tr></tbody>
</table>
</div>

<div class="admonition note"><p class="admonition-title">如何选择架构</p>
<ul>
<li><p><strong>独立子任务</strong> <span class="arithmatex">\(\rightarrow\)</span> 并行架构（分工、ensemble）。</p></li>
<li><p><strong>具有清晰依赖的顺序任务</strong> <span class="arithmatex">\(\rightarrow\)</span> pipeline。</p></li>
<li><p><strong>需要容错</strong> <span class="arithmatex">\(\rightarrow\)</span> 避免集中式；优先选择层级式或去中心化。</p></li>
<li><p><strong>可调试性关键</strong> <span class="arithmatex">\(\rightarrow\)</span> 集中式或 pipeline（所有决策可追溯）。</p></li>
<li><p><strong><span class="arithmatex">\(&lt;\)</span>5 个 Agent</strong> <span class="arithmatex">\(\rightarrow\)</span> 集中式最简单。<strong><span class="arithmatex">\(&gt;\)</span>20 个 Agent</strong> <span class="arithmatex">\(\rightarrow\)</span> 层级式或 swarm。</p></li>
</ul>

<p>在实践中，大多数生产系统采用<strong>层级式</strong>架构：顶层 Supervisor 委派给特定领域的子 Supervisor，后者管理由专门化 Worker 组成的小团队。</p>
</div>
