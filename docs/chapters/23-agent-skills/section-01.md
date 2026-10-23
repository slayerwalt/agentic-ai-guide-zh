# 23.1　什么是 Skill？ {#what-is-a-skill}

<p>一个 <strong>skill</strong> 是一个自包含的能力模块，赋予 Agent 在某个特定领域或任务上的专长。与仅暴露单个函数的原始 tool 不同，一个 skill 涵盖：</p>

<ul>
<li><p><strong>系统 prompt 增强</strong>：注入到 Agent 上下文中的领域特定指令、约束与人设要素。</p></li>
<li><p><strong>Tool 绑定</strong>：该 skill 所需的一个或多个 tool（API、MCP server、本地命令）。</p></li>
<li><p><strong>知识</strong>：Agent 正确执行该 skill 所需的参考资料、示例或 few-shot 演示。</p></li>
<li><p><strong>工作流逻辑</strong>：引导 Agent 完成复杂任务的多步骤流程、决策树或条件流。</p></li>
<li><p><strong>安全护栏</strong>：该 skill 特有的安全约束、输出格式要求与校验规则。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">Skill vs. Tool vs. Agent</p>
<span id="table-23-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>概念</strong></th><th scope="col"><strong>范围</strong></th><th scope="col"><strong>示例</strong></th></tr></thead>
<tbody><tr><td><strong>Tool</strong></td><td>单次函数调用</td><td><code>web_search(query)</code></td></tr>
<tr><td><strong>Skill</strong></td><td>完整的能力（prompts + tools + 知识）</td><td>“Research Analyst” skill</td></tr>
<tr><td><strong>Agent</strong></td><td>拥有多个 skill 的自主实体</td><td>一个编程助手</td></tr></tbody>
</table>
</div>

<p>Tool 是一把锤子。Skill 是知道<em>如何搭起房屋的框架</em>。Agent 则是决定使用哪些 skill 的木匠。</p>
</div>
