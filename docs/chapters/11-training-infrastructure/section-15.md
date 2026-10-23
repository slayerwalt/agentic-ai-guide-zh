# 11.15　RL 训练的优化器配置 {#sec:rl-optimizer-config}

<p>RL 训练（PPO、GRPO、DPO）相比预训练或 SFT 对优化器有独特要求。Loss 地形非平稳（policy 变化会改变生成的数据）、梯度更嘈杂（reward 信号方差大）、且更容易出现灾难性遗忘（Catastrophic Forgetting）或 reward hacking。本节以 AdamW <a class="book-citation" id="cite-loshchilov2019adamw-1" href="../../references.md#ref-loshchilov2019adamw" aria-label="参考文献 238">[238]</a> 作为默认优化器，整理 RL 特有的优化器实践指南。</p>

## 11.15.1　为什么 RL 需要不同的优化器设置 {#why-rl-requires-different-optimizer-settings}

<div class="admonition note"><p class="admonition-title">RL vs. SFT 优化——关键差异</p>
<ul>
<li><p><strong>数据分布非平稳</strong>：SFT 数据集固定，而 RL 每轮迭代都会生成新的 rollout——数据分布随 policy 一起漂移。</p></li>
<li><p><strong>高梯度方差</strong>：reward 信号稀疏且嘈杂；梯度方差远高于在精心策划数据上的交叉熵。</p></li>
<li><p><strong>需要更小的更新</strong>：policy 必须贴近 reference 模型（KL 约束），所以学习率比 SFT 小 10–100<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p><strong>不用 weight decay</strong>：正则化由 KL 惩罚提供，而不是 weight decay。再叠加 WD 可能与 KL 约束相互对抗。</p></li>
<li><p><strong>更短的 warmup</strong>：RL 从已收敛的 SFT checkpoint 起步——优化器状态只需极短的 warmup。</p></li>
</ul>
</div>

## 11.15.2　各 RL 方法的推荐超参 {#recommended-hyperparameters-by-rl-method}

<p class="table-caption" id="table-11-12"><span class="table-number">表 11.12　</span>RL 训练各阶段的优化器设置。均使用 <span class="arithmatex">\(\beta_1=0.9\)</span>、<span class="arithmatex">\(\beta_2=0.95\)</span>、<span class="arithmatex">\(\epsilon=10^{-8}\)</span>、<code>max_grad_norm</code>=1.0、BF16。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>优化器</strong></th><th scope="col"><strong>LR</strong></th><th scope="col"><strong>WD</strong></th><th scope="col"><strong>Warmup</strong></th><th scope="col"><strong>调度</strong></th></tr></thead>
<tbody><tr><td>DPO</td><td>AdamW</td><td><span class="arithmatex">\(5\text{e-}7\)</span></td><td>0.0</td><td>50 步</td><td>Constant 或 Linear</td></tr>
<tr><td>PPO（policy）</td><td>AdamW</td><td><span class="arithmatex">\(1\text{e-}6\)</span></td><td>0.0</td><td>20 步</td><td>Constant</td></tr>
<tr><td>PPO（critic）</td><td>AdamW</td><td><span class="arithmatex">\(1\text{e-}6\)</span></td><td>0.0</td><td>20 步</td><td>Constant</td></tr>
<tr><td>GRPO</td><td>AdamW</td><td><span class="arithmatex">\(1\text{e-}6\)</span></td><td>0.0</td><td>20 步</td><td>Constant</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 RL 用恒定（Constant）调度？</p>
<p>Cosine 与线性衰减调度都假设训练 horizon 固定、loss 单调下降。RL 训练两者都不具备：reward 可能停滞、尖峰，或不可预测地震荡。短暂 warmup 之后保持恒定 LR，可让优化器在整个训练中始终保持响应能力。若必须衰减，请使用非常温和的线性调度，最小 LR 比例保持较高（<span class="arithmatex">\(\geq 0.5\)</span>）。</p>
</div>

## 11.15.3　RL 用 Beta-2 = 0.95：更快的自适应 {#beta-2-0.95-for-rl-faster-adaptation}

<p>Adam 默认的 <span class="arithmatex">\(\beta_2 = 0.999\)</span> 给二阶矩留下了非常长的记忆（有效窗口 <span class="arithmatex">\(\sim\)</span>1000 步）。在 RL 训练中，loss 地形随 policy 演化迅速变化——1000 步前的梯度方差已无关紧要。使用 <span class="arithmatex">\(\beta_2 = 0.95\)</span> 把窗口缩短到 <span class="arithmatex">\(\sim\)</span>20 步，让自适应学习率能够迅速响应变化中的梯度统计。</p>

<div class="admonition warning"><p class="admonition-title">beta2 = 0.95 反受其害的场景</p>
<p>对于非常小的 batch（例如在线 RL 中 batch=1），<span class="arithmatex">\(\beta_2 = 0.95\)</span> 会让二阶矩估计噪声过大。这种情况下，使用 <span class="arithmatex">\(\beta_2 = 0.99\)</span> 作为折中，或通过梯度累积加大有效 batch。</p>
</div>

## 11.15.4　RL 的混合精度：FP32 Master 权重至关重要 {#mixed-precision-for-rl-fp32-master-weights-are-critical}

<p>RL 训练对数值精度尤其敏感：</p>

<ul>
<li><p>梯度更嘈杂——小幅更新必须在多步中精确累积</p></li>
<li><p>学习率极小（<span class="arithmatex">\(10^{-6}\)</span>–<span class="arithmatex">\(10^{-7}\)</span>），使得 <span class="arithmatex">\(\Delta\theta \ll \theta\)</span></p></li>
<li><p>BF16 尾数（7 位 <span class="arithmatex">\(\approx\)</span> 0.8% 相对精度）无法表示相对于 <span class="arithmatex">\(10^{0}\)</span> 量级权重的 <span class="arithmatex">\(10^{-6}\)</span> 量级更新</p></li>
</ul>

<p><strong>RL 训练务必使用 FP32 master 权重。</strong>纯 BF16 训练（不留 FP32 副本）几乎必然会让 PPO/GRPO 在 100–500 步后出现 reward 崩塌。</p>

## 11.15.5　梯度裁剪对 RL 至关重要 {#gradient-clipping-is-critical-for-rl}

<p>在 PPO 和 GRPO 中，reward 信号变异性可能极高，尤其训练早期。一个坏 batch 就能产生 norm <span class="arithmatex">\(&gt;100\)</span> 的梯度，足以彻底破坏模型权重。<code>max_grad_norm=1.0</code> 是标准设置。SFT 下裁剪不那么关键，但仍建议使用。</p>

<div class="admonition warning"><p class="admonition-title">RL 训练绝不要关闭梯度裁剪</p>
<p>SFT 的梯度 norm 通常稳定在 0.1–1.0；而 RL 的梯度是尖刺型的，原因有三：(1) reward 方差经 policy gradient 传播；(2) 稀有的高 reward 轨迹会产生异常大的更新；(3) policy 漂移时 KL 惩罚项会产生大梯度。哪怕只有一步 <span class="arithmatex">\(\|\nabla\| &gt; 50\)</span> 的未裁剪更新，都可能让数百步训练前功尽弃。</p>
</div>

## 11.15.6　诊断 RL 训练的不稳定性 {#diagnosing-rl-training-instability}

<div class="admonition example"><p class="admonition-title">RL 优化的红旗与修复</p>
<span id="table-11-inline-19"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>症状</strong></th><th scope="col"><strong>可能原因与修复</strong></th></tr></thead>
<tbody><tr><td>Reward 先升后崩</td><td>LR 过高或 KL 系数过低。将 LR 降 2–5<span class="arithmatex">\(\times\)</span>，或加大 <span class="arithmatex">\(\beta_\text{KL}\)</span>。</td></tr>
<tr><td>梯度 norm 一直顶在裁剪阈值</td><td>更新过于激进。降低 LR（持续裁剪意味着每一步都丢失梯度方向信息）。</td></tr>
<tr><td>KL 散度爆炸（<span class="arithmatex">\(&gt;\)</span>15 nat）</td><td>LR 过高。降低 10<span class="arithmatex">\(\times\)</span> 或加上自适应 KL 惩罚。</td></tr>
<tr><td>Reward 卡在基线</td><td>LR 过低，或 reward 模型信号过弱。试着把 LR 提 2–5<span class="arithmatex">\(\times\)</span>。检查 reward 模型校准。</td></tr>
<tr><td>100+ 步后 loss NaN</td><td>缺少 FP32 master 权重，或梯度 norm 溢出。开启 FP32 master 权重；验证 BF16 模式。</td></tr></tbody>
</table>
</div>
</div>

## 11.15.7　HuggingFace TRL 的 RL 配置 {#huggingface-trl-configuration-for-rl}

<p>TRL 库 <a class="book-citation" id="cite-vonwerra2022trl-1" href="../../references.md#ref-vonwerra2022trl" aria-label="参考文献 376">[376]</a> 为 LLM 的 PPO、DPO 等 RL 方法提供了生产级实现。</p>

<p class="code-caption"><span class="code-number">代码 11.1　</span>使用 TRL 的完整 PPO 与 DPO 优化器配置。</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">PPOConfig</span><span class="p">,</span> <span class="n">PPOTrainer</span><span class="p">,</span> <span class="n">DPOConfig</span><span class="p">,</span> <span class="n">DPOTrainer</span>

<span class="c1"># --- PPO 配置 ---</span>
<span class="n">ppo_config</span> <span class="o">=</span> <span class="n">PPOConfig</span><span class="p">(</span>
    <span class="c1"># 优化器（AdamW，RL 特化设置）</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">1e-6</span><span class="p">,</span>           <span class="c1"># 比 SFT 小 10-100x</span>

    <span class="c1"># PPO 特有</span>
    <span class="n">ppo_epochs</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>                 <span class="c1"># 每个 rollout 上做的 mini-batch 更新次数</span>
    <span class="n">mini_batch_size</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span>
    <span class="n">batch_size</span><span class="o">=</span><span class="mi">64</span><span class="p">,</span>                <span class="c1"># rollout 的 batch 大小</span>

    <span class="c1"># 梯度控制</span>
    <span class="n">max_grad_norm</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>

    <span class="c1"># KL 惩罚（取代 weight decay 作为正则化）</span>
    <span class="n">init_kl_coef</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span>            <span class="c1"># 初始 KL 惩罚系数</span>
    <span class="n">adap_kl_ctrl</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>           <span class="c1"># 自适应 KL 目标</span>
    <span class="n">target_kl</span><span class="o">=</span><span class="mf">6.0</span><span class="p">,</span>               <span class="c1"># 目标 KL 散度</span>

    <span class="c1"># 混合精度</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>                   <span class="c1"># BF16 计算，FP32 master 权重</span>
<span class="p">)</span>

<span class="n">ppo_trainer</span> <span class="o">=</span> <span class="n">PPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">config</span><span class="o">=</span><span class="n">ppo_config</span><span class="p">,</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
    <span class="n">dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
<span class="p">)</span>

<span class="c1"># --- DPO 配置 ---</span>
<span class="n">dpo_config</span> <span class="o">=</span> <span class="n">DPOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./dpo_output"</span><span class="p">,</span>

    <span class="c1"># 优化器</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">5e-7</span><span class="p">,</span>           <span class="c1"># 比 PPO 还小</span>
    <span class="n">optim</span><span class="o">=</span><span class="s2">"adamw_torch"</span><span class="p">,</span>
    <span class="n">adam_beta1</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span>
    <span class="n">adam_beta2</span><span class="o">=</span><span class="mf">0.95</span><span class="p">,</span>              <span class="c1"># RL 用更短的记忆</span>
    <span class="n">weight_decay</span><span class="o">=</span><span class="mf">0.0</span><span class="p">,</span>            <span class="c1"># 不用 WD —— KL 提供正则化</span>

    <span class="c1"># 调度</span>
    <span class="n">lr_scheduler_type</span><span class="o">=</span><span class="s2">"constant_with_warmup"</span><span class="p">,</span>
    <span class="n">warmup_steps</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span>

    <span class="c1"># 梯度控制</span>
    <span class="n">max_grad_norm</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>

    <span class="c1"># DPO 特有</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>                    <span class="c1"># KL 约束强度</span>
    <span class="n">loss_type</span><span class="o">=</span><span class="s2">"sigmoid"</span><span class="p">,</span>         <span class="c1"># 标准 DPO loss</span>

    <span class="c1"># 混合精度</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>

    <span class="c1"># 训练</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>          <span class="c1"># DPO 通常 1 个 epoch</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">8</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">dpo_trainer</span> <span class="o">=</span> <span class="n">DPOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">dpo_config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">dpo_trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>

## 11.15.8　MoE 在 RL 训练中的考虑 {#moe-considerations-for-rl-training}

<div class="admonition tip"><p class="admonition-title">MoE 用于 RLHF</p>
<p>混合专家（Mixture-of-Experts, MoE）模型 <a class="book-citation" id="cite-fedus2022switch-1" href="../../references.md#ref-fedus2022switch" aria-label="参考文献 89">[89]</a> 在 RLHF 中越来越常见：</p>

<ul>
<li><p><strong>优势</strong>：相同算力下容量提升 3–4<span class="arithmatex">\(\times\)</span>。对 reward 模型尤其有利（更多容量去打分）。</p></li>
<li><p><strong>挑战</strong>：专家并行需要 all-to-all 通信（token 跨 GPU 路由），与流水线并行存在冲突。</p></li>
<li><p><strong>GRPO + MoE</strong>：效果不错，因为生成成本由激活参数主导（而非总参数）。</p></li>
<li><p><strong>MoE 用 LoRA</strong>：可以只对路由器 + 共享层加 LoRA，或对所有专家都加（昂贵）。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">RL 优化器口诀</p>
<p>做 RL 微调请记住：<strong>小 LR、不用 weight decay、恒定调度、FP32 master 权重、激进裁剪</strong>。把正则化交给 KL 惩罚——优化器只管沿 policy gradient 前进，且不要冲过头。</p>
</div>

#### Miles：PyTorch 原生的 RL 后训练。 {#heading-11-57}

<p>Miles <a class="book-citation" id="cite-miles2026pytorch-1" href="../../references.md#ref-miles2026pytorch" aria-label="参考文献 295">[295]</a> 是 PyTorch 官方为大规模 LLM RL 打造的技术栈，把 SGLang（用于高吞吐 rollout 生成）与 Megatron-LM（用于分布式训练更新）连接起来。它同时支持分离式执行（rollout 与训练使用各自的 GPU 池，最大化利用率）与同址执行（把两者放在同一批节点上，用于小规模实验）。Miles 代表了此前临时拼凑基础设施的标准化——正如十年前 PyTorch 本身标准化了深度学习训练一样。</p>
