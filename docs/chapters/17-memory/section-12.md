# 17.12　主动记忆架构 {#sec:proactive-memory}

<p>长时程智能体任务中的一个根本性挑战是<strong>行为状态衰减（behavioral state decay）</strong>：在数十次工具调用之后，执行 Agent 会丢失对原始目标、约束与已积累上下文的追踪。标准记忆系统是反应式地解决这一问题——Agent 在需要信息时才查询记忆。<strong>主动记忆架构</strong> <a class="book-citation" id="cite-meta2026remember-1" href="../../references.md#ref-meta2026remember" aria-label="参考文献 253">[253]</a>（Meta AI，2026 年 7 月，《Remember When It Matters》）采取了不同的方法：一个专门的记忆 Agent <em>监控</em>执行者，并在执行者<em>漂移之前</em>注入约束。</p>

<div class="admonition note"><p class="admonition-title">主动记忆 Agent 架构</p>
<p>系统由两个解耦的 Agent 组成：</p>

<ol>
<li><p><strong>执行 Agent（Executor Agent）</strong>：负责任务执行——工具调用、推理、输出生成。维持一个聚焦于当前子任务的精简上下文。</p></li>
<li><p><strong>记忆 Agent（Memory Agent）</strong>：并行运行，监控执行者的动作流。维护以下内容的压缩表示：</p>

<ul>
<li><p>长期目标与成功标准</p></li>
<li><p>生效中的约束（安全规则、用户偏好、资源限制）</p></li>
<li><p>轨迹早期做出的关键决策</p></li>
</ul></li>
</ol>

<p>记忆 Agent 只在检测到漂移时——即执行者近期的动作表明它忘记了一条约束或正在偏离既定目标时——才把<strong>提醒注入</strong>执行者的上下文。这是与反应式记忆的关键区别：记忆 Agent <em>主动</em>行动，赶在错误累积之前。</p>
</div>

<div class="admonition example"><p class="admonition-title">主动记忆：实证结果</p>
<p>在长时程智能体基准上（每个任务 50+ 次工具调用）：</p>

<ul>
<li><p>在需要持续遵守约束的任务（如“绝不修改 /prod 中的文件”“始终注明来源”）上<strong>成功率显著更高</strong>。</p></li>
<li><p><strong>API 成本增加 <span class="arithmatex">\(\sim\)</span>40%</strong>：记忆 Agent 为监控与注入增加了 LLM 调用。这是主要的权衡代价。</p></li>
<li><p><strong>优雅降级</strong>：没有记忆 Agent 时，执行者在 <span class="arithmatex">\(\sim\)</span>30 次工具调用后成功率急剧下降；有了它，性能在 100+ 次调用中保持稳定。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">RL 联系：把记忆 Agent 视为状态汇总器</p>
<p>主动记忆架构在 RL 框架中有一个自然的解释：</p>

<ul>
<li><p><strong>记忆 Agent</strong> 起到<strong>状态汇总器</strong>的作用——类似于 actor-critic 架构中的注意力机制，它为价值函数选择性加权相关的历史状态。</p></li>
<li><p>在 POMDP（部分可观测 MDP）中，记忆 Agent 维护<strong>信念状态</strong>：与未来决策相关的历史压缩充分统计量。</p></li>
<li><p>注入机制类似于非对称 actor-critic 训练中的<strong>特权信息</strong>：记忆 Agent 可以访问完整轨迹历史，而执行者只操作压缩后的上下文。</p></li>
<li><p>40% 的成本增加反映了<strong>马尔可夫状态表示</strong>（昂贵但充分）与<strong>截断上下文</strong>（廉价但有损）之间的根本权衡。</p></li>
</ul>

<p>这一架构为 RL 训练指明了一个有前景的方向：联合优化执行者策略与记忆 Agent 的注入策略，让记忆 Agent 学习在<em>何时</em>注入<em>什么</em>以最大化任务成功。</p>
</div>
