# 28.6　DPO 扩展题 {#dpo-extensions-questions}

<section class="quiz-question" id="q-26"><h2 class="question-title">Q26：比较 f-DPO 的散度选择。前向 KL、JS 与反向 KL 各自何时使用？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：标准 DPO 隐式使用反向 KL（<span class="arithmatex">\(D_\text{KL}[\pi_\theta \| \pi_\text{ref}]\)</span>）：</p>

<ul>
<li><p><strong>反向 KL</strong>（默认）：寻峰（mode-seeking）。policy 把概率集中在参考概率高的位置。避免生成参考不会生成的文本。利于安全（保守）。</p></li>
<li><p><strong>前向 KL</strong>：覆盖（mass-covering）。policy 试图覆盖参考的所有峰，甚至低概率的峰。利于多样性但可能生成低质量输出。</p></li>
<li><p><strong>Jensen-Shannon</strong>：前向与反向之间的对称折中。在覆盖与寻峰间取得平衡。对通用对齐通常最佳。</p></li>
<li><p><strong>Alpha 散度</strong>（<span class="arithmatex">\(\alpha=0.5\)</span>）：在前向（<span class="arithmatex">\(\alpha=0\)</span>）与反向（<span class="arithmatex">\(\alpha=1\)</span>）之间插值。可调。</p></li>
</ul>

<p><strong>实践建议</strong>：先用反向 KL（标准 DPO）。若模型过于保守（不愿尝试创造性方案），切换到 JS 散度。若多样性至关重要（创意写作、头脑风暴），试试前向 KL。</p>

<p><em><strong>复习：</strong> 第 <a href="../06-dpo/index.md">6</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（DPO；偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-27"><h2 class="question-title">Q27：你的 DPO 偏好数据有 15% 的标签噪声。怎么办？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：按复杂度递增的三个方案：</p>

<p><strong>1. Robust DPO</strong>（已知噪声率时最佳）：解析地去偏 loss：<span class="arithmatex">\(\mathcal{L}_\text{robust} = \frac{(1-\varepsilon)\mathcal{L}_\text{DPO}(y_w, y_l) - \varepsilon \mathcal{L}_\text{DPO}(y_l, y_w)}{1 - 2\varepsilon}\)</span>。设 <span class="arithmatex">\(\varepsilon = 0.15\)</span>。在期望意义上可证明恢复干净的 DPO 目标。TRL：<code>loss_type=&quot;robust&quot;, label_smoothing=0.15</code>。</p>

<p><strong>2. IPO（Identity Preference Optimization, IPO）</strong>（噪声率未知时最佳）：带目标 margin 的平方 loss。被误标的对影响有界（平方 loss 不发散）。对任意噪声模式更鲁棒，且无需知道 <span class="arithmatex">\(\varepsilon\)</span>。TRL：<code>loss_type=&quot;ipo&quot;</code>。</p>

<p><strong>3. TR-DPO</strong>（分布漂移时最佳）：训练中通过 EMA 更新参考模型。即便早期数据有噪声，演化的参考也能帮助模型自我修正。TRL：<code>sync_ref_model=True, ref_model_mixup_alpha=0.6</code>。</p>

<p><strong>数据侧修复</strong>：(1) 过滤标注者一致率 <span class="arithmatex">\(&lt;\)</span>70% 的对。(2) 用 RM 给对打分；丢弃 RM 与标签不一致的对。(3) 主动学习：重新标注最不确定的对。</p>

<p><em><strong>复习：</strong> 第 <a href="../06-dpo/index.md">6</a> 与 <a href="../08-preference-optimization/index.md">8</a> 章（DPO；偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-28"><h2 class="question-title">Q28：什么是 SimPO？为什么“无参考模型”是优势？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：SimPO 用回答的平均对数概率作为隐式 reward 信号：<span class="arithmatex">\(r(x,y) = \frac{1}{|y|}\sum_t \log \pi_\theta(y_t|x, y_{&lt;t})\)</span>——无需参考模型。</p>

<p>loss 中加入目标 margin <span class="arithmatex">\(\gamma\)</span>：chosen 回答的平均 log-prob 应至少比 rejected 高 <span class="arithmatex">\(\gamma\)</span>。</p>

<p><strong>为什么“无参考”重要</strong>：</p>

<ol>
<li><p><strong>显存</strong>：无参考模型 = 70B 节省 70–140GB。可在同样硬件上训练更大的模型。</p></li>
<li><p><strong>简洁性</strong>：无需管理/加载/服务第二份模型副本。</p></li>
<li><p><strong>无陈旧参考</strong>：DPO 的参考随着训练推进越来越不相关。SimPO 没有这个问题。</p></li>
<li><p><strong>内置长度归一化</strong>：<span class="arithmatex">\(1/|y|\)</span> 天然防止长度偏差（DPO 需显式处理）。</p></li>
</ol>

<p><strong>权衡</strong>：没有参考锚点，模型有更多自由去塌缩或漂移。<span class="arithmatex">\(\gamma\)</span> margin 和长度归一化部分缓解了这一点，但在激进训练时 SimPO 可能不如 DPO 稳定。</p>

<p><em><strong>复习：</strong> 第 <a href="../08-preference-optimization/index.md">8</a> 章（偏好优化变体）。</em></p>
</details></section>

<section class="quiz-question" id="q-29"><h2 class="question-title">Q29：解释 Iterative RPO。为何在推理任务中把 DPO 与 NLL loss 结合？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：用于推理的标准 DPO 有一种微妙的失效模式：它学会<em>判别</em>（给正确轨迹更高的隐式 reward），但不一定学会<em>生成</em>它们。</p>

<p><strong>为什么</strong>：DPO 的梯度推高 chosen 的概率、压低 rejected 的概率。但 chosen 回答可能与模型自身会生成的东西差异巨大，以至于提升其概率并不能教会模型产出类似的推理模式。</p>

<p><strong>RPO 的修复</strong>：在 chosen 回答上添加负对数似然（NLL/SFT）loss：<span class="arithmatex">\(\mathcal{L} = \mathcal{L}_\text{DPO} + \alpha \cdot \mathcal{L}_\text{NLL}(y_w)\)</span>。</p>

<p>NLL 项显式训练模型逐步生成获胜回答。DPO 项确保模型同时学会避免落败回答。结合起来：模型同时学到“如何正确推理”（NLL）和“该避免什么”（DPO）。</p>

<p><strong>迭代</strong>：生成回答 <span class="arithmatex">\(\rightarrow\)</span> 检查正确性 <span class="arithmatex">\(\rightarrow\)</span> 构造对 <span class="arithmatex">\(\rightarrow\)</span> 用 RPO 训练 <span class="arithmatex">\(\rightarrow\)</span> 重复。每次迭代，模型更擅长生成正确推理，为下一轮提供更高质量的训练数据。</p>

<p>TRL：<code>loss_type=[&quot;sigmoid&quot;, &quot;sft&quot;], loss_weights=[1.0, 1.0]</code></p>

<p><em><strong>复习：</strong> 第 <a href="../08-preference-optimization/index.md">8</a> 与 <a href="../13-reasoning/index.md">13</a> 章（偏好优化变体；大型推理模型的 RL）。</em></p>
</details></section>
