# 28.16　训练中的多样性问题 {#diversity-in-training-questions}

<section class="quiz-question" id="q-source-67"><h2 class="question-title">Q：如果一个 GRPO 组中 N 个响应全部相同会怎样？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：如果全部 <span class="arithmatex">\(N\)</span> 个响应相同：所有 Reward <span class="arithmatex">\(r_i\)</span> 相等，因此 <span class="arithmatex">\(\sigma_G = 0\)</span>，优势 <span class="arithmatex">\(\hat{A}_i = (r_i - \mu_G)/\sigma_G\)</span> 未定义（除以零）。实践中，实现把所有 <span class="arithmatex">\(\hat{A}_i = 0\)</span>，意味着<strong>零学习信号</strong>——这一步被浪费。</p>

<p><strong>预防</strong>：</p>

<ol>
<li><p><strong>温度</strong>：生成时使用 <span class="arithmatex">\(\tau = 0.7\)</span>–<span class="arithmatex">\(1.0\)</span>（不要贪心解码）。</p></li>
<li><p><strong>大 <span class="arithmatex">\(N\)</span></strong>：<span class="arithmatex">\(N=8\)</span>–<span class="arithmatex">\(16\)</span> 增加多样化响应的概率。</p></li>
<li><p><strong>重复拒绝</strong>：DAPO 的做法——拒绝重复响应并重采样。</p></li>
<li><p><strong>频率惩罚</strong>：生成时对重复的 n-gram 施加惩罚。</p></li>
<li><p><strong>监控</strong>：跟踪每组唯一响应比例。若 <span class="arithmatex">\(&lt;\)</span>50%，调高温度。</p></li>
</ol>

<p><em><strong>复习：</strong>第 <a href="../07-grpo/index.md">7</a> 章（GRPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-source-68"><h2 class="question-title">Q：解释 RLHF 中的多样性—质量权衡。如何检测模式坍缩？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>权衡</strong>：高多样性（高熵/高温度）= 多样但可能随机/低质量的响应。低多样性 = 一致但重复、被 Reward hack 的响应。</p>

<p><strong>检测模式坍缩</strong>（训练期间均应监控）：</p>

<ol>
<li><p><strong>响应熵</strong>：计算每个 token 的熵 <span class="arithmatex">\(H = -\sum p_i \log p_i\)</span>。若快速下降 <span class="arithmatex">\(\to\)</span> 坍缩。</p></li>
<li><p><strong>唯一 n-gram 比例</strong>：同一 Prompt 不同响应间唯一 4-gram 的比例。健康：<span class="arithmatex">\(&gt;\)</span>0.6。</p></li>
<li><p><strong>Reward 分布宽度</strong>：若 <span class="arithmatex">\(\sigma(\text{rewards})\)</span> 收缩到接近零 <span class="arithmatex">\(\to\)</span> 所有响应质量相同 <span class="arithmatex">\(\to\)</span> 可能完全一致。</p></li>
<li><p><strong>KL 散度</strong>：若 <span class="arithmatex">\(D_\text{KL}[\pi_\theta \| \pi_\text{ref}]\)</span> 快速增长，Policy 正远离参考 <span class="arithmatex">\(\to\)</span> 通常朝向某个狭窄模式。</p></li>
<li><p><strong>长度直方图</strong>：若所有响应收敛到相同长度 <span class="arithmatex">\(\to\)</span> 模板化行为。</p></li>
</ol>

<p><strong>修复</strong>：提高 KL 系数 <span class="arithmatex">\(\beta\)</span>、增大熵奖励、提高采样温度，或回滚到更早的 checkpoint。</p>

<p><em><strong>复习：</strong>第 <a href="../07-grpo/index.md">7</a> 章与第 <a href="../09-reward-models/index.md">9</a> 章（GRPO；Reward 模型训练）。</em></p>
</details></section>
