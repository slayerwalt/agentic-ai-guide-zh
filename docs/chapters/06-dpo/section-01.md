# 6.1　动机 {#motivation}

<p>PPO 需要在显存中维持 4 个模型（policy、reference、reward 模型、value head）、复杂的 RL 基础设施，并且以不稳定著称。DPO <a class="book-citation" id="cite-rafailov2023direct-1" href="../../references.md#ref-rafailov2023direct" aria-label="参考文献 302">[302]</a> 提出的问题是：<em>我们能否跳过 RL，直接从偏好中学习？</em></p>

<p><strong>关键洞察</strong>：在 RLHF 目标（reward 最大化 + KL 惩罚）下的最优 policy 拥有<strong>解析解</strong>。由此我们可以推导出一个有监督 loss，隐式优化同一个目标。</p>
