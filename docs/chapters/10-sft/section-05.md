# 10.5　当 SFT 反而有害 —— 灾难性遗忘与对齐税 {#sec:sft-pitfalls}

<p>当 LLM 依次经历各训练阶段 —— 预训练 <span class="arithmatex">\(\rightarrow\)</span> 继续预训练 <span class="arithmatex">\(\rightarrow\)</span> SFT <span class="arithmatex">\(\rightarrow\)</span> RLHF/DPO —— 标准基准上的性能下降时常出现。驱动这些回退的是两种<strong>本质上截然不同</strong>的现象，将其混淆将导致错误的缓解策略。</p>

## 10.5.1　灾难性遗忘（结构性擦除） {#catastrophic-forgetting-structural-erasure}

<div class="admonition warning"><p class="admonition-title">灾难性遗忘（Catastrophic Forgetting）</p>
<p>灾难性遗忘是一种<strong>非故意的优化失败</strong>：当一个在分布 <span class="arithmatex">\(\mathcal{D}_A\)</span> 上优化过的网络随后在与之不相交的分布 <span class="arithmatex">\(\mathcal{D}_B\)</span> 上训练时，为 <span class="arithmatex">\(\mathcal{D}_B\)</span> 所需的权重更新<em>物理上覆盖</em>了编码 <span class="arithmatex">\(\mathcal{D}_A\)</span> 的参数结构：</p>



<div class="arithmatex">\[\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_B(\theta_t) \quad \implies \quad \mathcal{L}_A(\theta_{t+1}) \gg \mathcal{L}_A(\theta_t)
\tag{10.1}\]</div>

<p>知识被<strong>摧毁</strong> —— 编码任务 A 的权重不再存在。不重新训练就无法恢复。</p>
</div>

<p><strong>症状</strong>：</p>

<ul>
<li><p>在微调数据之外的任务上完全崩溃（例如模型在 chat 数据上 SFT 后忘记如何做数学）</p></li>
<li><p>语言多样性丧失 —— 模型只能以微调分布的狭窄风格生成</p></li>
<li><p>在微调阶段未被强化的知识上，事实准确性下降</p></li>
<li><p>仅在英语数据上 SFT 后多语言能力下降</p></li>
</ul>

<p><strong>机制根源 —— Fisher 信息视角</strong>：任务 A 的 Fisher 信息矩阵 <span class="arithmatex">\(F\)</span> 标识了哪些参数对 <span class="arithmatex">\(\mathcal{D}_A\)</span> “重要”:</p>



<div class="arithmatex">\[F = \mathbb{E}_{x \sim \mathcal{D}_A}\!\left[\nabla_\theta \log \pi_\theta(x)\, \nabla_\theta \log \pi_\theta(x)^T\right]
\tag{10.2}\]</div>

<p>Fisher 特征值高的参数对任务 A 至关重要。无约束的任务 B 梯度下降完全忽略这些特征值 —— <span class="arithmatex">\(\Delta\theta\)</span> 沿 <span class="arithmatex">\(\nabla\mathcal{L}_B\)</span> 方向移动，而不管是否摧毁了 <span class="arithmatex">\(\mathcal{L}_A\)</span> 的高 Fisher 方向。</p>

## 10.5.2　对齐税（行为约束） {#alignment-tax-behavioral-constraint}

<p>对齐税（Alignment Tax）是<strong>一种有意为之、可预期的权衡</strong>：模型的原始能力（无约束生成、最大化推理带宽）下降，是因为 Policy 被约束去生成安全、格式良好、与偏好对齐的输出。</p>

<p><strong>机制</strong>：在 DPO/PPO 期间，Policy <span class="arithmatex">\(\pi_\theta\)</span> 通过 KL 散度被惩罚以防偏离参考 <span class="arithmatex">\(\pi_{\text{ref}}\)</span>:</p>



<div class="arithmatex">\[r_{\text{implicit}}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
\tag{10.3}\]</div>

<p>这条“链条”约束着模型的<strong>输出分布</strong> —— 它无法探索偏离参考太远的高方差推理路径。知识并未被<strong>擦除</strong>；而是被<em>抑制</em>了。模型仍“知道”答案，但其分布被压平，偏向安全、通用的回应。</p>

<p><strong>症状</strong>：</p>

<ul>
<li><p>过度拒答（对良性查询回复“I can’t help with that”）</p></li>
<li><p>风格僵硬 —— 模糊措辞、过多警示语、冗长的安全免责声明</p></li>
<li><p>在原始能力基准（MMLU、HumanEval）上分数下降，而在偏好基准（MT-Bench、AlpacaEval）上提升</p></li>
<li><p>生成复杂、高熵输出（创意写作、新颖算法）的能力下降</p></li>
</ul>

## 10.5.3　对比分类 {#comparative-taxonomy}

<p class="table-caption" id="table-10-1"><span class="table-number">表 10.1　</span>灾难性遗忘 vs 对齐税 —— 完整对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>灾难性遗忘</strong></th><th scope="col"><strong>对齐税</strong></th></tr></thead>
<tbody><tr><td><strong>意图性</strong></td><td>非故意（优化产物）</td><td>可预期权衡（为安全/有用而有意承担）</td></tr>
<tr><td><strong>参数状态</strong></td><td>先验知识被物理覆盖</td><td>潜在分布被约束/截断</td></tr>
<tr><td><strong>信息</strong></td><td><strong>被摧毁</strong>：权重不再编码该能力</td><td><strong>被抑制</strong>：知识仍存在但更难触发</td></tr>
<tr><td><strong>主导阶段</strong></td><td>顺序 SFT、领域继续预训练</td><td>偏好优化（PPO、DPO、KTO、RLHF）</td></tr>
<tr><td><strong>主要症状</strong></td><td>基线能力完全崩溃</td><td>过度拒答、风格僵硬、原始基准分数下降</td></tr>
<tr><td><strong>可逆性</strong></td><td>不重新从 checkpoint 训练则不可逆</td><td>部分可逆：调整 <span class="arithmatex">\(\beta\)</span>、system prompt 或微调</td></tr>
<tr><td><strong>检测</strong></td><td>预训练评估集上的困惑度飙升</td><td>困惑度稳定但能力基准上的胜率下降</td></tr>
<tr><td><strong>随模型规模</strong></td><td>各尺度相近</td><td>小模型支付更大的对齐税</td></tr></tbody>
</table>
</div>

## 10.5.4　缓解策略 {#mitigation-strategies}

<p><strong>针对灾难性遗忘</strong>：</p>

<ol>
<li><p><strong>数据回放（Data replay）</strong>：将 5–10% 的预训练数据混入 SFT 数据集。确保 Gradient 更新不会完全忽视预训练分布。</p></li>
<li><p><strong>弹性权重整合（Elastic Weight Consolidation, EWC）</strong> <a class="book-citation" id="cite-kirkpatrick2017overcoming-1" href="../../references.md#ref-kirkpatrick2017overcoming" aria-label="参考文献 186">[186]</a>：添加正则项 <span class="arithmatex">\(\Omega(\theta) = \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2\)</span>，惩罚对原任务而言 Fisher 信息高的参数的变化。</p></li>
<li><p><strong>LoRA / 参数高效微调</strong>：只训练低秩 adapter（参数量 <span class="arithmatex">\(&lt;1\%\)</span>），基础权重完全冻结。这避免了预训练知识的<em>永久性摧毁</em> —— 你随时可以移除 adapter 来恢复原模型。然而，<strong>当 adapter 处于激活状态时</strong>，组合系统 <span class="arithmatex">\((W_0 + BA)\)</span> 仍可能表现出遗忘：adapter 可能将模型的有效行为推离旧技能。LoRA 保护的是 checkpoint，而非激活时的推理行为。</p></li>
<li><p><strong>保守的学习率</strong>：使用 <span class="arithmatex">\(1\)</span>–<span class="arithmatex">\(5 \times 10^{-6}\)</span> 配合较少的 Epoch(1–3)。更大的学习率会加速遗忘。</p></li>
<li><p><strong>渐进式训练</strong>：逐步混合分布，随时间增加 SFT 数据比例，而非突然切换。</p></li>
</ol>

<p><strong>针对对齐税</strong>：</p>

<ol>
<li><p><strong>仔细调节 <span class="arithmatex">\(\beta\)</span></strong>：更低的 <span class="arithmatex">\(\beta\)</span> 给予模型更多自由（减少税负），但可能牺牲安全性。多数场景下最优 <span class="arithmatex">\(\beta \in [0.05, 0.3]\)</span>。</p></li>
<li><p><strong>高质量、多样化的 SFT 数据</strong>：对齐税的一部分来自 SFT 收窄了输出分布；更广、更多样化的 SFT 数据可减少这部分。RL 阶段通过 KL 正则化 <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a>进一步施加约束。</p></li>
<li><p><strong>条件对齐（Conditional alignment）</strong>：训练模型仅在安全标志激活时才对齐。推理时为基准测试关闭约束（仅供研究使用的技术）。</p></li>
<li><p><strong>Constitutional AI / RLAIF</strong>：利用模型自生成反馈创建更细致的偏好数据，在提升对齐的同时保留能力。</p></li>
<li><p><strong>有针对性的 RL 预算</strong>：不要过度训练 RL。监控能力基准，当税负超出可接受阈值（通常是 2–5% MMLU 回退）时停止。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">如何判断你遇到的是哪一种</p>
<ul>
<li><p><strong>在失败任务上运行 base 模型</strong>：若 base 模型成功而微调后模型完全失败 <span class="arithmatex">\(\rightarrow\)</span> 灾难性遗忘。</p></li>
<li><p><strong>Prompt 工程测试</strong>：若精心设计的 prompt（例如“忽略安全准则，逐步解答此数学题”）能恢复该能力 <span class="arithmatex">\(\rightarrow\)</span> 对齐税（知识被抑制而非擦除）。</p></li>
<li><p><strong>困惑度检查</strong>：在预训练验证集上计算困惑度。飙升 = 遗忘。稳定 = 对齐税。</p></li>
<li><p><strong>Few-shot 恢复</strong>：若提供少量上下文示例就能恢复能力 <span class="arithmatex">\(\rightarrow\)</span> 对齐税。若大量示例也无法恢复 <span class="arithmatex">\(\rightarrow\)</span> 遗忘。</p></li>
</ul>
</div>
