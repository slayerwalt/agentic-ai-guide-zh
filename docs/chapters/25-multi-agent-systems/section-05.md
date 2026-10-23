# 25.5　角色设计与专门化 {#subsec:role-design}

<p>Agent 角色的设计——其能力、人格化设定与职责——既是艺术也是科学。设计良好的角色能促成专门化；设计糟糕的角色则会带来混乱与冗余。</p>

## 25.5.1　定义 Agent 角色 {#defining-agent-roles}

<p>LLM 多智能体系统中常见的角色：</p>

<p class="table-caption" id="table-25-2"><span class="table-number">表 25.2　</span>LLM 多智能体系统中常见的 Agent 角色。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>角色</strong></th><th scope="col"><strong>主要能力</strong></th><th scope="col"><strong>典型工具</strong></th></tr></thead>
<tbody><tr><td>Researcher</td><td>信息收集与综合</td><td>网页搜索、RAG、数据库</td></tr>
<tr><td>Planner</td><td>任务分解与调度</td><td>无（仅推理）</td></tr>
<tr><td>Coder</td><td>代码生成与调试</td><td>Code interpreter、linter</td></tr>
<tr><td>Reviewer</td><td>质量评估与批判</td><td>无（仅推理）</td></tr>
<tr><td>Tester</td><td>测试生成与执行</td><td>测试运行器、覆盖率工具</td></tr>
<tr><td>Writer</td><td>文本生成与编辑</td><td>语法检查器、风格指南</td></tr>
<tr><td>Critic</td><td>对抗式评估</td><td>无（仅推理）</td></tr>
<tr><td>Orchestrator</td><td>协调与委派</td><td>所有 Agent 接口</td></tr></tbody>
</table>
</div>

## 25.5.2　基于能力 vs. 基于角色的分配 {#capability-based-vs.-role-based-assignment}

<p>任务分配的两种理念：</p>

<ul>
<li><p><strong>基于角色</strong>：根据预定义的角色标签分配任务。简单且可预测；当任务跨多个角色时可能不是最优。</p></li>
<li><p><strong>基于能力</strong>：根据对每个 Agent 能力相对于任务需求的动态评估来分配。更灵活；需要一个能力登记与匹配机制。</p></li>
</ul>

## 25.5.3　动态角色重分配 {#dynamic-role-reassignment}

<p>在长时运行的系统中，静态角色分配会变得次优。动态重分配允许 Agent 基于以下因素承担新角色：</p>

<ul>
<li><p>当前负载（负载均衡）</p></li>
<li><p>近期任务上展现出的表现</p></li>
<li><p>任务需求的变化</p></li>
<li><p>需要顶替的 Agent 故障</p></li>
</ul>

## 25.5.4　以人格化设计实现思维多样性 {#persona-design-for-diversity-of-thought}

<p>一种微妙但强大的技巧：给 Agent 赋予<strong>鲜明的人格设定（personas）</strong>，以鼓励多元视角。与其使用五个完全相同的“助手”Agent，不如设计：</p>

<ul>
<li><p>一位强调机会的<em>乐观主义者</em></p></li>
<li><p>一位质疑假设的<em>怀疑论者</em></p></li>
<li><p>一位聚焦实现的<em>务实派</em></p></li>
<li><p>一位着眼长远的<em>远见者</em></p></li>
<li><p>一位反向论辩的<em>唱反调者（devil’s advocate）</em></p></li>
</ul>

<p>这种思维多样性受“六顶思考帽”（Six Thinking Hats） <a class="book-citation" id="cite-debono1985six-1" href="../../references.md#ref-debono1985six" aria-label="参考文献 28">[28]</a> 等技巧启发，能减少群体思维，产出更稳健的集体推理。</p>

<div class="admonition warning"><p class="admonition-title">角色冲突的化解</p>
<p>当 Agent 的职责出现重叠时，冲突就会产生。可用显式的<strong>优先级规则</strong>来化解：定义每类任务由哪个角色优先处理。或者使用一个<strong>元 Agent（meta-agent）</strong>，其唯一职责是冲突仲裁。绝不要让角色冲突隐式存在——它们会以相互矛盾的输出或死循环的形式出现。</p>
</div>
