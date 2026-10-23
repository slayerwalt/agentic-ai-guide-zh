# 29.21　智能体安全检查清单 {#agent-security-checklist}

<span id="table-29-inline-15"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>威胁</strong></th><th scope="col"><strong>层级</strong></th><th scope="col"><strong>缓解措施</strong></th></tr></thead>
<tbody><tr><td>Prompt 注入（直接）</td><td>输入</td><td>输入校验、指令层级、分隔符</td></tr>
<tr><td>Prompt 注入（间接）</td><td>工具输出</td><td>将工具输出视为不可信；勿遵循检索文档中的指令</td></tr>
<tr><td>工具滥用</td><td>执行</td><td>最小权限；<code>destructiveHint</code> 关卡；沙箱化</td></tr>
<tr><td>数据外泄</td><td>输出</td><td>输出过滤；限制工具仅访问白名单域</td></tr>
<tr><td>过度自主</td><td>架构</td><td>最大迭代次数；成本预算；人工审批关卡</td></tr>
<tr><td>混淆代理（Confused Deputy）</td><td>多 Agent</td><td>校验任务来源；基于能力的访问控制</td></tr></tbody>
</table>
</div>
