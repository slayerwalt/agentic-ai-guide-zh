# 23.4　Skill 生命周期 {#skill-lifecycle}

<ol>
<li><p><strong>发现</strong>：系统识别有哪些可用的 skill（注册表、市场、本地定义）。</p></li>
<li><p><strong>选择</strong>：根据用户请求匹配并加载相关 skill。</p></li>
<li><p><strong>激活</strong>：将 skill 的 prompt、tool 与知识注入到 Agent 的上下文中。</p></li>
<li><p><strong>执行</strong>：Agent 利用该 skill 的能力完成任务。</p></li>
<li><p><strong>失活</strong>：移除 skill 上下文，为后续任务释放上下文窗口空间。</p></li>
<li><p><strong>学习</strong>：执行结果可能更新该 skill 的 few-shot 示例或微调路由。</p></li>
</ol>
