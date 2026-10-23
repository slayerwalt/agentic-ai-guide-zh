# 17.5　多轮对话中的记忆 {#sec:memory-multiturn}

## 17.5.1　用户建模与偏好追踪 {#user-modeling-and-preference-tracking}

<p>持久化的用户模型 <span class="arithmatex">\(\mathcal{U}\)</span> 存储：</p>

<ul>
<li><p><strong>显式偏好：</strong> 用户明确表达的喜好/厌恶、沟通风格偏好。</p></li>
<li><p><strong>隐式偏好：</strong> 从行为中推断（例如用户总是要求 Python 代码、偏好简洁回答）。</p></li>
<li><p><strong>专业水平：</strong> 从词汇与问题复杂度中推断出的领域知识水平。</p></li>
<li><p><strong>目标与上下文：</strong> 进行中的项目、当前任务、组织角色。</p></li>
</ul>

<p>用户模型在每次交互后更新：</p>



<div class="arithmatex">\[\mathcal{U}_{t+1} = \text{Update}(\mathcal{U}_t,\, (u_t, a_t, \text{feedback}_t)).\]</div>

## 17.5.2　会话连续性 {#session-continuity}

<p>没有记忆，每次对话都从“冷启动”开始。有了会话记忆后：</p>

<ol>
<li><p>在会话开始时，检索用户模型 <span class="arithmatex">\(\mathcal{U}\)</span> 与近期会话摘要。</p></li>
<li><p>注入个性化的系统 Prompt：“你正在帮助 Alice，一位从事分布式训练项目的高级 ML 工程师。上次会话你帮她调试了一个 Gradient 同步问题。”</p></li>
<li><p>在会话结束时，对会话进行摘要并更新 <span class="arithmatex">\(\mathcal{U}\)</span>。</p></li>
</ol>

## 17.5.3　通过记忆实现个性化 {#personalization-through-memory}

<p>个性化同时提升<em>效率</em>（更少的澄清问题）与<em>质量</em>（针对用户专业度校准的回答）。核心技术包括：</p>

<ul>
<li><p><strong>自适应详尽度：</strong> 根据用户的历史互动调整回答长度。</p></li>
<li><p><strong>领域引导（Domain priming）：</strong> 从语义记忆中预置相关领域上下文。</p></li>
<li><p><strong>主动回忆：</strong> 无需用户提示即主动浮现相关的历史交互（“你上个月问过这个话题，当时我们的结论是……”）。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">隐私与记忆</p>
<p>持久化的用户记忆带来了显著的隐私问题。Agent 必须：（1）在存储个人信息前获取明确同意；（2）提供检视与删除已存储记忆的机制；（3）在多用户部署中强制执行访问控制；（4）遵守数据留存法规（GDPR、CCPA）。记忆系统应秉持“默认隐私（privacy-by-default）”原则进行设计。</p>
</div>
