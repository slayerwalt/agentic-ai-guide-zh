# 第 28 章 · 硬件、优化与训练系统 {#reading-reading-02}

## 28.7　GPU 架构与硬件题 {#gpu-architecture-and-hardware-questions}

<section class="quiz-question" id="q-30"><h3 class="question-title">Q30：解释 GPU 内存层次结构。它对 LLM 推理为何重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：从最快到最慢：</p>

<ol>
<li><p><strong>寄存器（Registers）</strong>：线程私有，<span class="arithmatex">\(\sim\)</span>256 KB/SM。即时访问（0 周期延迟）。</p></li>
<li><p><strong>SRAM（共享内存）</strong>：每 SM 私有，<span class="arithmatex">\(\sim\)</span>192–228 KB/SM（A100：164 KB 可配置）。聚合带宽：<span class="arithmatex">\(\sim\)</span>19 TB/s。延迟：<span class="arithmatex">\(\sim\)</span>20 周期。</p></li>
<li><p><strong>L2 缓存</strong>：跨整张 GPU 共享，40–60 MB（H100：50 MB）。带宽：<span class="arithmatex">\(\sim\)</span>5 TB/s。延迟：<span class="arithmatex">\(\sim\)</span>200 周期。</p></li>
<li><p><strong>HBM</strong>：GPU 主显存，80 GB（A100）。带宽：2–3.35 TB/s。延迟：<span class="arithmatex">\(\sim\)</span>400 周期。</p></li>
<li><p><strong>CPU DRAM</strong>：通过 PCIe，512 GB+。带宽：32–64 GB/s。延迟：<span class="arithmatex">\(\sim\)</span>10K 周期。</p></li>
</ol>

<p><strong>对 LLM 为何重要</strong>：自回归生成在每个 Token 都要读取完整模型权重（70B 约 <span class="arithmatex">\(\sim\)</span>140 GB）。以 2 TB/s 的 HBM 带宽，仅流式读权重就要 70ms。实际计算（一次矩阵-向量乘法）只要 0.5ms。GPU 有 99% 的时间在等数据。</p>

<p><strong>FlashAttention 利用了这一点</strong>：通过把中间结果（QK 分数、softmax）保留在 SRAM（19 TB/s）而非写到 HBM（2 TB/s），消除了 attention 的 90% 内存流量。计算量未变，但 HBM 的读写降低 10<span class="arithmatex">\(\times\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-31"><h3 class="question-title">Q31：FlashAttention 如何工作？什么是 online softmax 技巧？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：标准 attention 在 HBM 中物化 <span class="arithmatex">\(n \times n\)</span> 的 attention 矩阵。<span class="arithmatex">\(n=8192\)</span> 时：每个 head <span class="arithmatex">\(8192^2 \times 2 = 134\)</span> MB，32 头每层 4.3 GB。必须写到 HBM 再读回做 softmax 与乘法——3 次完整的 HBM 往返。</p>

<p><strong>FlashAttention 方案</strong>：从不存储完整的 <span class="arithmatex">\(n \times n\)</span> 矩阵。以可放进 SRAM 的分块（tile）处理。</p>

<p><strong>算法</strong>：</p>

<ol>
<li><p>将 <span class="arithmatex">\(Q\)</span> 拆为 <span class="arithmatex">\(B_r\)</span> 行的块，<span class="arithmatex">\(K/V\)</span> 拆为 <span class="arithmatex">\(B_c\)</span> 行的块。</p></li>
<li><p>对每个 <span class="arithmatex">\(Q\)</span> 块：遍历所有 <span class="arithmatex">\(K\)</span> 块，计算部分 attention 分数。</p></li>
<li><p><strong>Online softmax 技巧</strong>：维护滑动 max <span class="arithmatex">\(m\)</span> 与滑动 sum <span class="arithmatex">\(\ell\)</span> 以做 softmax 归一。处理新的 <span class="arithmatex">\(K\)</span> 块时更新：<span class="arithmatex">\(m_\text{new} = \max(m_\text{old}, \max(\text{scores}))\)</span>，将上一累加器按 <span class="arithmatex">\(e^{m_\text{old} - m_\text{new}}\)</span> 重新缩放，再加上新的贡献。</p></li>
<li><p>输出以增量方式累加——从不需要完整的 <span class="arithmatex">\(n \times n\)</span> 矩阵。</p></li>
</ol>

<p><strong>关键洞见</strong>：softmax 本来是全局操作（对所有元素取 <span class="arithmatex">\(\max\)</span> 和 <span class="arithmatex">\(\sum\)</span>）。online 技巧将其分解为带修正因子的局部更新。数学上精确——并非近似。</p>

<p><strong>结果</strong>：显存 <span class="arithmatex">\(O(n)\)</span> 而非 <span class="arithmatex">\(O(n^2)\)</span>。速度提升 2–4<span class="arithmatex">\(\times\)</span>（更少 HBM 访问，更多时间在 SRAM）。</p>

<p><strong>FlashAttention 2</strong>：在 warp 之间做更好的工作划分，将非矩阵乘的 FLOPs 减半。</p>

<p><strong>FlashAttention 3</strong>（H100/Hopper）：使用张量内存加速器（Tensor Memory Accelerator, TMA）做异步加载、warp 专门化（生产者/消费者 warp）、支持 FP8。</p>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 与 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 架构；系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-32"><h3 class="question-title">Q32：解释 PagedAttention。它如何解决 KV cache 问题？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：生成期间，每条序列都需要 KV cache（存储所有先前 Token 的 K、V 张量）。对 70B 模型：每个 Token 需要 <span class="arithmatex">\(2 \times n_\text{layers} \times d_\text{model} \times 2\)</span> 字节 = <span class="arithmatex">\(2 \times 80 \times 8192 \times 2 \approx 2.5\)</span> MB。2048 Token 的序列：<span class="arithmatex">\(\sim\)</span>5 GB 的 KV cache。</p>

<p><strong>传统分配</strong>：为每条活跃序列预分配 max_sequence_length。若 max=2048 而平均=500，就浪费 75% 已分配显存。50 条并发序列就浪费数百 GB。</p>

<p><strong>PagedAttention</strong>：受操作系统虚拟内存启发：</p>

<ol>
<li><p>KV cache 被切分为固定大小的<em>块</em>（页），每块保存 16 个 Token 的 KV。</p></li>
<li><p>一张<em>块表</em>（类似页表）将逻辑 Token 位置映射到物理显存块。</p></li>
<li><p>块按需随序列增长分配。无预分配浪费。</p></li>
<li><p>序列结束时，释放的块立即归还到池中。</p></li>
</ol>

<p><strong>额外收益</strong>：</p>

<ul>
<li><p><strong>前缀共享</strong>：拥有相同系统 prompt 的多条序列共享 KV cache 块（写时复制）。聊天应用可省 30–50% 显存。</p></li>
<li><p><strong>抢占</strong>：可将低优先级序列的块“换出”到 CPU，为高优先级请求腾出 GPU 显存。</p></li>
<li><p><strong>接近零碎片</strong>：内部碎片仅限末块（<span class="arithmatex">\(&lt;\)</span>16 Token）。外部碎片被消除（任意空闲块可用于任意位置）。</p></li>
</ul>

<p><strong>结果</strong>：同等显存下并发序列数提升 3–5<span class="arithmatex">\(\times\)</span> <span class="arithmatex">\(\rightarrow\)</span> 服务吞吐提升 3–5<span class="arithmatex">\(\times\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-33"><h3 class="question-title">Q33：比较 NVLink 与 InfiniBand。在 RLHF 训练中各用于何处？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>NVLink</strong>（节点内，GPU 到 GPU）：</p>

<ul>
<li><p>带宽：600 GB/s（A100）、900 GB/s（H100）——双向总和。</p></li>
<li><p>延迟：<span class="arithmatex">\(\sim\)</span>1 <span class="arithmatex">\(\mu\)</span>s。</p></li>
<li><p>范围：单个物理节点内（8 GPU 通过 NVSwitch 互联）。</p></li>
<li><p>用例：<strong>张量并行</strong>（TP=8）。每层矩阵乘法在 GPU 间切分，每层后都要 AllReduce。需要超高带宽 + 低延迟。</p></li>
</ul>

<p><strong>InfiniBand NDR</strong>（跨节点，节点到节点）：</p>

<ul>
<li><p>带宽：每端口 400 Gb/s = 50 GB/s。8 端口（GPUDirect RDMA）：每节点聚合 400 GB/s。</p></li>
<li><p>延迟：<span class="arithmatex">\(\sim\)</span>1–5 <span class="arithmatex">\(\mu\)</span>s（RDMA）。</p></li>
<li><p>范围：集群中跨节点。需要交换机（fat-tree 拓扑）。</p></li>
<li><p>用例：<strong>数据并行 / FSDP</strong> 梯度同步。梯度 AllReduce 每训练步发生一次（而非每层），因此延迟容忍度更高。</p></li>
</ul>

<p><strong>在 RLHF 中具体而言</strong>：</p>

<ul>
<li><p><em>生成</em>：节点内 NVLink 上 TP=8。跨节点的多个 vLLM 实例无需通信（embarrassingly parallel）。</p></li>
<li><p><em>训练</em>：节点内 NVLink 上 TP=8 + 节点间 InfiniBand 上 FSDP。梯度在完整反向传播后同步。</p></li>
<li><p><em>权重同步</em>：训练 <span class="arithmatex">\(\to\)</span> 生成使用 InfiniBand（140 GB 传输，异步，50 GB/s 下需 <span class="arithmatex">\(\sim\)</span>3s）。</p></li>
</ul>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（系统基础；系统架构）。</em></p>
</details></section>

## 28.8　优化与训练题 {#optimization-and-training-questions}

<section class="quiz-question" id="q-34"><h3 class="question-title">Q34：解释 Adam 与 AdamW。这一差异对 LLM 为何重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>Adam 加 L2 正则化</strong>：<span class="arithmatex">\(\theta_{t+1} = \theta_t - \alpha \cdot (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda\theta_t)\)</span>。权重衰减项 <span class="arithmatex">\(\lambda\theta_t\)</span> <em>位于</em>自适应缩放<em>内部</em>。梯度大（<span class="arithmatex">\(v_t\)</span> 大）的参数<em>衰减更少</em>（除以 <span class="arithmatex">\(\sqrt{v_t}\)</span>）。这不是真正的权重衰减——它与尺度相关。</p>

<p><strong>AdamW（解耦权重衰减）</strong>：<span class="arithmatex">\(\theta_{t+1} = (1 - \alpha\lambda)\theta_t - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)\)</span>。权重衰减在自适应更新<em>之外</em>且<em>之前</em>施加。无论梯度历史如何，每个参数都获得相同比例的衰减。</p>

<p><strong>对 LLM 为何重要</strong>：</p>

<ol>
<li><p>LLM 的参数跨越多个数量级（embedding 层 vs attention vs FFN）。Adam 的耦合 L2 实际上对小梯度参数惩罚得比大梯度参数更多——错误行为。</p></li>
<li><p>解耦 WD 在所有层提供统一正则化，防止某些层无界增长而另一些层过度收缩。</p></li>
<li><p>经验：在长预训练上，AdamW 相比相同有效正则强度的 Adam+L2 给出 2–5% 更好的 perplexity。</p></li>
</ol>

<p><strong>对 RL 而言</strong>：通常使用 <span class="arithmatex">\(\lambda = 0\)</span>（无权重衰减）。由 KL 惩罚提供正则化。但对 SFT：AdamW 配合 <span class="arithmatex">\(\lambda = 0.01\)</span>–<span class="arithmatex">\(0.1\)</span> 是标准做法。</p>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 章（LLM 架构与优化方法）。</em></p>
</details></section>

<section class="quiz-question" id="q-35"><h3 class="question-title">Q35：为什么学习率 warmup 是必要的？不做 warmup 会发生什么？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：Adam 的二阶矩估计 <span class="arithmatex">\(v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2\)</span> 从 <span class="arithmatex">\(v_0 = 0\)</span> 起步。偏置修正 <span class="arithmatex">\(\hat{v}_t = v_t/(1-\beta_2^t)\)</span> 在数学上做了补偿，但在实践中：</p>

<ul>
<li><p>头几步：<span class="arithmatex">\(v_t\)</span> 基于 1–5 个梯度样本。对真实方差的估计极不准确。</p></li>
<li><p>若某参数初期恰好得到小梯度，<span class="arithmatex">\(v_t\)</span> 微小 <span class="arithmatex">\(\rightarrow\)</span> 有效 LR 巨大 <span class="arithmatex">\(\rightarrow\)</span> 灾难性更新。</p></li>
<li><p>偏置修正放大早期更新：第 1 步时 <span class="arithmatex">\(\hat{v}_1 = v_1/(1-0.999) = 1000 \cdot v_1\)</span>。</p></li>
</ul>

<p><strong>无 warmup 时</strong>：头 10–100 步经常出现梯度尖峰，永久性损坏模型。在优化器稳定之前，早期表征就被打乱。</p>

<p><strong>Warmup 的修复</strong>：从 LR <span class="arithmatex">\(\approx 0\)</span> 起步，在 <span class="arithmatex">\(W\)</span> 步（通常为训练量的 3–10%）内线性升至目标值。当 LR 到达满值时，<span class="arithmatex">\(v_t\)</span> 已积累了足够样本以变得准确。</p>

<p><strong>典型设置</strong>：</p>

<ul>
<li><p>预训练：2000 步 warmup（占 200K 步的 <span class="arithmatex">\(\sim\)</span>1%）。</p></li>
<li><p>SFT：100 步 warmup（占 2000 步的 <span class="arithmatex">\(\sim\)</span>5%）。</p></li>
<li><p>RL（PPO/GRPO）：20–50 步 warmup（短，模型经 SFT 已稳定）。</p></li>
</ul>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 与 <a href="../10-sft/index.md">10</a> 章（LLM 架构；SFT 最佳实践）。</em></p>
</details></section>

<section class="quiz-question" id="q-36"><h3 class="question-title">Q36：比较学习率调度策略。RL 微调你会选哪种？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>余弦衰减（Cosine decay）</strong>：<span class="arithmatex">\(\eta_t = \eta_\text{min} + \frac{1}{2}(\eta_\text{max} - \eta_\text{min})(1 + \cos(\pi t/T))\)</span>。预训练与 SFT 的标准选择。衰减平滑，大部分时间停留在中等 LR。</p>

<p><strong>线性衰减</strong>：<span class="arithmatex">\(\eta_t = \eta_\text{max}(1 - t/T)\)</span>。更简单，短训练下与余弦衰减效果相近。</p>

<p><strong>预热-稳定-衰减（Warmup-Stable-Decay, WSD）</strong>：warmup <span class="arithmatex">\(\rightarrow\)</span> 80% 时间常数 LR <span class="arithmatex">\(\rightarrow\)</span> 最后 20% 快速衰减。预训练的新标准。“稳定”阶段提供一致学习；最终衰减挤出剩余收益。</p>

<p><strong>常数</strong>：无衰减。warmup 之后 <span class="arithmatex">\(\eta_t = \eta_\text{max}\)</span>。</p>

<p><strong>对 RL 微调（PPO/GRPO），我会选择：短 warmup + 常数</strong>。理由：</p>

<ol>
<li><p>RL 训练长度高度不可预测（按胜率停训，而非 epoch）。</p></li>
<li><p>余弦/线性衰减假设你提前知道总步数。</p></li>
<li><p>LR 已经非常低（<span class="arithmatex">\(10^{-6}\)</span>），再衰减会让更新几乎不可见。</p></li>
<li><p>PPO 的自适应 KL 控制器已在调控有效步长。</p></li>
<li><p>如必须衰减：在宽裕预算上做线性衰减，指标停滞时早停。</p></li>
</ol>

<p><em><strong>复习：</strong> 第 <a href="../01-llm-architecture/index.md">1</a> 与 <a href="../05-ppo/index.md">5</a> 章（LLM 架构；PPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-37"><h3 class="question-title">Q37：为何梯度裁剪（Gradient Clipping）对 RL 训练至关重要、对 SFT 却不那么重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>SFT</strong>：监督 loss 平滑且行为良好。梯度范数在 batch 间一致（通常 0.1–1.0）。在 1.0 处裁剪很少触发——只是安全网。</p>

<p><strong>RL（PPO/GRPO）</strong>：梯度范数高度多变，因为：</p>

<ol>
<li><p><strong>Reward 方差</strong>：一个 batch 可能全是高 reward 回答，下一个全是低的。advantage <span class="arithmatex">\(\hat{A}\)</span> 剧烈摆动。</p></li>
<li><p><strong>比率爆炸</strong>：若某个稀有 Token 的概率变化很大，<span class="arithmatex">\(r_t = \pi_\text{new}/\pi_\text{old}\)</span> 可能极大 <span class="arithmatex">\(\rightarrow\)</span> 在裁剪生效前就产生大梯度。</p></li>
<li><p><strong>稀疏 reward</strong>：在二元 reward 的 GRPO 中，有些 prompt 全部正确（advantage <span class="arithmatex">\(\approx 0\)</span>），突然一道难题给出极端 advantage。</p></li>
<li><p><strong>KL 项</strong>：当 policy 偏离时，KL 惩罚的梯度可能尖峰。</p></li>
</ol>

<p><strong>无裁剪时</strong>：一个坏 batch 就能产生比正常大 100<span class="arithmatex">\(\times\)</span> 的梯度 <span class="arithmatex">\(\rightarrow\)</span> 一步毁掉模型。无法恢复（所有预训练遭灾难性遗忘）。</p>

<p><strong>典型设置</strong>：<code>max_grad_norm=1.0</code>。一些人在 RL 训练早期用 0.5 以更安全。范数在所有参数上全局计算（非逐层）。</p>

<p><strong>监控</strong>：若裁剪在超过 20% 步触发，你的 LR 可能过高，或 batch size 过小。</p>

<p><em><strong>复习：</strong> 第 <a href="../05-ppo/index.md">5</a> 与 <a href="../07-grpo/index.md">7</a> 章（PPO；GRPO）。</em></p>
</details></section>

<section class="quiz-question" id="q-38"><h3 class="question-title">Q38：训练中 BF16 vs FP16。这个选择何时重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>FP16</strong>：1 符号 + 5 指数 + 10 尾数位。范围：<span class="arithmatex">\(\pm 65504\)</span>。精度：<span class="arithmatex">\(\sim\)</span>3.3 十进制位。</p>

<p><strong>BF16</strong>：1 符号 + 8 指数 + 7 尾数位。范围：<span class="arithmatex">\(\pm 3.4 \times 10^{38}\)</span>（与 FP32 相同！）。精度：<span class="arithmatex">\(\sim\)</span>2.4 十进制位。</p>

<p><strong>为什么 BF16 在 LLM 上胜出</strong>：</p>

<ol>
<li><p><strong>无需 loss scaling</strong>：FP16 范围窄（<span class="arithmatex">\(\pm\)</span>65K）导致梯度和激活频繁溢出/下溢。需要动态 loss scaling（loss 乘 1024，梯度再除回去）。BF16 拥有 FP32 的范围——溢出基本不可能。</p></li>
<li><p><strong>代码更简单</strong>：无 loss scaler，无 inf/nan 检查，无动态 scaling 调整。</p></li>
<li><p><strong>对 RL 关键</strong>：RL 梯度比 SFT 更嘈杂、更易尖峰。FP16 的 loss scaling 经常失败（选错尺度，产生 NaN）。BF16 “开箱即用”。</p></li>
</ol>

<p><strong>FP16 可能更好的场景</strong>：若你需要极致精度（某些科学计算任务）并能管理好 loss scaling。FP16 多 3 个尾数位 = 结果略更精确。</p>

<p><strong>FP32 主权重</strong>：即便前向/反向使用 BF16，也要在 FP32 中累加梯度更新，以防舍入误差在数千次小步中累积。所有 LLM 训练的标准做法。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

## 28.9　Reward 模型与 SFT 题 {#reward-model-and-sft-questions}

<section class="quiz-question" id="q-39"><h3 class="question-title">Q39：推导 Bradley-Terry reward 模型的 loss。它有哪些局限？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>Bradley-Terry 模型（Bradley-Terry Model）</strong>：给定两个回答，更好的那个（<span class="arithmatex">\(y_w\)</span>）被偏好的概率：<span class="arithmatex">\(P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))\)</span>，其中 <span class="arithmatex">\(\sigma\)</span> 是 sigmoid。</p>

<p><strong>MLE 推导</strong>：给定 <span class="arithmatex">\(N\)</span> 个偏好对，最大化似然：<span class="arithmatex">\(\prod_i P(y_w^i \succ y_l^i)\)</span>。取负对数：<span class="arithmatex">\(\mathcal{L} = -\sum_i \log\sigma(r(x_i, y_w^i) - r(x_i, y_l^i))\)</span>。</p>

<p><strong>局限</strong>：</p>

<ol>
<li><p><strong>无平局</strong>：BT 无法建模“同样好”——强制严格偏好。</p></li>
<li><p><strong>可传递性</strong>：假设若 A<span class="arithmatex">\(&gt;\)</span>B 且 B<span class="arithmatex">\(&gt;\)</span>C 则 A<span class="arithmatex">\(&gt;\)</span>C。人类并不可传递。</p></li>
<li><p><strong>上下文无关</strong>：无论备选项是什么，reward 都相同。</p></li>
<li><p><strong>标量塌缩</strong>：把所有质量维度压成一个数。一个回答可能既安全又无用——RM 必须权衡。</p></li>
<li><p><strong>长度偏差</strong>：更长的回答得到更高分（信息更多 = 更可能包含标注者想要的内容）。必须显式去相关。</p></li>
</ol>

<p><strong>缓解措施</strong>：margin loss（要求最小间隔 <span class="arithmatex">\(\delta\)</span>）、reward 中心化（减去滑动均值）、训练时长度惩罚、多头 RM（为有用性/安全性/准确性提供独立分数）。</p>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 章（Reward 模型训练）。</em></p>
</details></section>

<section class="quiz-question" id="q-40"><h3 class="question-title">Q40：SFT 中的序列打包（Sequence Packing）是什么？为何重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：训练样本长度可变。标准批处理把所有样本填充到 max_length。若 max=4096 而平均=500，就有 88% 的算力浪费在填充 Token 上（梯度贡献为零）。</p>

<p><strong>打包方案</strong>：把多个短样本拼接成一条 max_length 序列，用 EOS Token 分隔。同时对所有样本训练。</p>

<p>例子：不再用 4 条填充到 4096 的序列（16K Token，14K 填充），而是打包成 1 条 4096 的序列，首尾相连地容纳 4 个样本（4096 真实 Token，0 填充）。<strong>效率提升 4<span class="arithmatex">\(\times\)</span>。</strong></p>

<p><strong>关键细节——块对角 attention 掩码</strong>：若不做特殊处理，样本 2 会关注样本 1 的 Token（交叉污染）。必须使用块对角 attention 掩码，限制每个样本只能关注自身的 Token。</p>

<p><strong>在 TRL 中</strong>：<code>SFTConfig(packing=True, max_seq_length=4096)</code>。自动处理掩码。</p>

<p><strong>注意</strong>：(1) 更长的样本仍需自己的 batch 条目（不能在序列中间切分）。(2) 位置 Embedding 实现略复杂（每个样本重置）。(3) 有人认为打包改变了有效 batch size（每步样本更多）——相应调整 LR。</p>

<p><em><strong>复习：</strong> 第 <a href="../10-sft/index.md">10</a> 章（SFT 最佳实践与技巧）。</em></p>
</details></section>

<section class="quiz-question" id="q-41"><h3 class="question-title">Q41：解释 SFT 的 completion-only masking。不使用它会怎样？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：在 chat 格式的 SFT 数据中：<code>[system] + [user message] + [assistant response]</code>。标准 NLL loss 会在所有 Token 上计算 loss，包括 system prompt 和用户消息。</p>

<p><strong>不掩码的问题</strong>：模型浪费容量去预测用户消息（推理时永远不需要生成）。更糟：若训练数据中用户消息多样，模型会困惑“现在轮到谁说话？”</p>

<p><strong>Completion-only masking</strong>：将 prompt（system + user）中所有 Token 的 loss 权重设为 0。仅在 assistant 回答 Token 上计算 loss。</p>

<p>TRL：<code>DataCollatorForCompletionOnlyLM(response_template=&quot;&lt;|assistant|&gt;&quot;)</code></p>

<p><strong>影响</strong>：在指令跟随基准上通常提升 5–15%。收敛更快（梯度信号集中于有用 Token）。算力成本不变。</p>

<p><strong>细节</strong>：response 模板 Token 必须包含在 loss 内（教模型开始回答），但之前的所有内容都要排除。</p>

<p><em><strong>复习：</strong> 第 <a href="../10-sft/index.md">10</a> 章（SFT 最佳实践与技巧）。</em></p>
</details></section>

<section class="quiz-question" id="q-42"><h3 class="question-title">Q42：SFT 质量如何影响 RL 的天花板？pass@k 诊断是什么？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>天花板定理（非正式表述）</strong>：RL 只能强化模型已能以非可忽略概率产出的行为。若 SFT 模型生成正确解的概率为 0%，RL 永远找不到它。</p>

<p><strong>为什么</strong>：GRPO/PPO 从当前 policy 采样并强化好样本。若好样本不在分布内，就没什么可强化。RL 的探索受 base policy 支撑集（support）的限制。</p>

<p><strong>pass@k 诊断</strong>：对每个 prompt 生成 <span class="arithmatex">\(k\)</span> 个回答，检查<em>是否有任一个</em>正确：</p>

<ul>
<li><p>pass@1：模型的典型表现（greedy/低温）。</p></li>
<li><p>pass@8：<span class="arithmatex">\(G=8\)</span> 的 GRPO 所能达到的上界。</p></li>
<li><p>pass@64：激进 Best-of-N 采样（Rejection Sampling）的上界。</p></li>
<li><p>pass@256：RL 改进的近似天花板。</p></li>
</ul>

<p><strong>解读</strong>：</p>

<ul>
<li><p>pass@1=20%、pass@64=80%：好！RL 有 4<span class="arithmatex">\(\times\)</span> 上行空间。预期强劲收益。</p></li>
<li><p>pass@1=20%、pass@64=25%：几乎没有上行空间。RL 帮不上忙。需先做更好的 SFT。</p></li>
<li><p>pass@1=5%、pass@64=60%：模型<em>能</em>解但很少解出来。RL 的完美场景（强化稀有的成功）。</p></li>
</ul>

<p><strong>规则</strong>：若 pass@64 <span class="arithmatex">\(&lt;\)</span> 1.5<span class="arithmatex">\(\times\)</span> pass@1，在开始 RL 之前先投入更好的 SFT 数据。</p>

<p><em><strong>复习：</strong> 第 <a href="../07-grpo/index.md">7</a> 与 <a href="../10-sft/index.md">10</a> 章（GRPO；SFT 最佳实践）。</em></p>
</details></section>

<section class="quiz-question" id="q-43"><h3 class="question-title">Q43：为聊天模型设计一个多目标 reward 系统。如何平衡有用性与安全性？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>架构</strong>：每个目标使用独立 reward 模型：</p>

<ul>
<li><p><span class="arithmatex">\(r_\text{helpful}\)</span>：在有用性偏好上训练（质量、准确性、完整性）。</p></li>
<li><p><span class="arithmatex">\(r_\text{safe}\)</span>：在安全偏好上训练（拒答、无害、无幻觉）。</p></li>
<li><p><span class="arithmatex">\(r_\text{format}\)</span>：基于规则（遵循指令、合适格式、合理长度）。</p></li>
</ul>

<p><strong>组合策略</strong>：</p>

<ol>
<li><p><strong>加权求和</strong>（最简）：<span class="arithmatex">\(r = w_1 r_\text{helpful} + w_2 r_\text{safe} + w_3 r_\text{format}\)</span>。问题：安全性可能被有用性压过。</p></li>
<li><p><strong>约束式</strong>（更安全）：在 <span class="arithmatex">\(r_\text{safe} &gt; \tau\)</span> 约束下最大化 <span class="arithmatex">\(r_\text{helpful}\)</span>。通过 <span class="arithmatex">\(r = r_\text{helpful} - \lambda \cdot \max(0, \tau - r_\text{safe})\)</span> 实现，<span class="arithmatex">\(\lambda\)</span> 取较大值。</p></li>
<li><p><strong>GDPO 归一化</strong>（对 GRPO 最佳）：在 group 内独立归一化每个 reward，再组合：<span class="arithmatex">\(\hat{A} = w_1 \hat{A}_\text{helpful} + w_2 \hat{A}_\text{safe}\)</span>。防止某个 reward 因尺度差异而主导。</p></li>
<li><p><strong>字典序（Lexicographic）</strong>：安全是硬约束（必须通过），再优化有用性。分阶段训练：先做安全对齐，再做有用性。</p></li>
</ol>

<p><strong>实用权重</strong>：从 <span class="arithmatex">\(w_\text{safe}=2.0, w_\text{helpful}=1.0, w_\text{format}=0.5\)</span> 起步。安全权重 2<span class="arithmatex">\(\times\)</span>，因为它的失败模式（有害内容）远比有用性失败（平庸答案）更糟。</p>

<p><em><strong>复习：</strong> 第 <a href="../09-reward-models/index.md">9</a> 与 <a href="../12-agentic-training/index.md">12</a> 章（Reward 模型训练；LLM 智能体训练）。</em></p>
</details></section>

## 28.10　系统架构扩展题 {#system-architecture-extension-questions}

<section class="quiz-question" id="q-44"><h3 class="question-title">Q44：投机解码（Speculative Decoding）如何工作？它在 RLHF 中何时有帮助？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：大模型每次前向只产出一个 Token（70B 约 <span class="arithmatex">\(\sim\)</span>70ms）。慢。</p>

<p><strong>投机解码</strong>：</p>

<ol>
<li><p><strong>草稿</strong>：小模型（1–7B）快速生成 <span class="arithmatex">\(k\)</span> 个候选 Token（全部 <span class="arithmatex">\(k\)</span> 个约 <span class="arithmatex">\(\sim\)</span>5ms）。</p></li>
<li><p><strong>验证</strong>：大模型做一次前向，并行给所有 <span class="arithmatex">\(k\)</span> 个 Token 打分。<span class="arithmatex">\(p_\text{large}(t_i) \geq p_\text{draft}(t_i)\)</span> 的 Token 总是接受。其他按概率接受。</p></li>
<li><p><strong>结果</strong>：每次验证步平均接受 3–4 个 Token。加速：2–3<span class="arithmatex">\(\times\)</span>。</p></li>
</ol>

<p><strong>关键性质</strong>：输出分布与单独从大模型采样<em>完全相同</em>。无质量损失。草稿模型只影响速度，不影响输出。</p>

<p><strong>对 RLHF 而言</strong>：生成占算力 60%。生成加速 2–3<span class="arithmatex">\(\times\)</span> = 端到端加速 1.5–2<span class="arithmatex">\(\times\)</span>。结合 vLLM + INT8：生成从瓶颈变为与训练持平。</p>

<p><strong>局限</strong>：(1) 草稿模型必须共享 tokenizer。(2) 高温下效果较差（草稿模型准确率下降）。(3) 草稿模型需要额外 GPU 显存。(4) 超过 <span class="arithmatex">\(k=5\)</span> 后边际收益递减（接受率下降）。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（系统基础；系统架构）。</em></p>
</details></section>

<section class="quiz-question" id="q-45"><h3 class="question-title">Q45：解释 roofline 模型。如何判断一个 kernel 是计算受限还是内存受限？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：roofline 模型将可达性能（FLOPS）刻画为算术强度（每字节内存流量的 FLOPS）的函数。</p>

<p><strong>两个区域</strong>：</p>

<ul>
<li><p><strong>内存受限</strong>（拐点左侧）：性能受限于向计算单元喂数据的速度。实际 FLOPS = 带宽 <span class="arithmatex">\(\times\)</span> 算术强度。GPU 利用率 <span class="arithmatex">\(&lt;\)</span> 100%。</p></li>
<li><p><strong>计算受限</strong>（拐点右侧）：性能受限于峰值 FLOPS。内存够快。GPU 达到最大利用率。</p></li>
</ul>

<p><strong>拐点</strong>：峰值 FLOPS / 峰值带宽。A100：312 TF / 2 TB/s = 156 FLOP/byte。</p>

<p><strong>LLM 操作</strong>：</p>

<ul>
<li><p><strong>自回归生成</strong>（batch=1）：读取 140GB 权重，做 140G FLOPs = 1 FLOP/byte。<em>极度</em>内存受限（比拐点低 156<span class="arithmatex">\(\times\)</span>）。GPU 利用率仅 0.6%。</p></li>
<li><p><strong>训练前向</strong>（batch=128，seq=2048）：算术强度 <span class="arithmatex">\(\approx 200\)</span>+ FLOP/byte。计算受限。接近峰值利用率。</p></li>
<li><p><strong>Attention</strong>（长序列）：<span class="arithmatex">\(O(n^2 d)\)</span> FLOPs / <span class="arithmatex">\(O(n^2 + nd)\)</span> 字节。<span class="arithmatex">\(n\)</span> 长：计算受限；<span class="arithmatex">\(n\)</span> 短：内存受限。FlashAttention 无论如何都让其留在 SRAM。</p></li>
</ul>

<p><strong>实用法</strong>：若 kernel 内存受限，减少内存流量（量化、缓存、分块）。若计算受限，减少 FLOPs（剪枝、蒸馏、降低精度）。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-46"><h3 class="question-title">Q46：连续批处理（Continuous Batching）如何工作？为何它对 RLHF 生成至关重要？</h3>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>静态批处理</strong>：启动 <span class="arithmatex">\(B\)</span> 条序列。等<em>所有</em>序列结束。若一条生成 500 Token、另一条只生成 50 Token，那 50 Token 序列的 GPU 槽位会闲置 450 Token 的时间。</p>

<p><strong>连续批处理</strong>（迭代级调度）：每个生成步后检查哪些序列已完成。立即在腾出的槽位插入新序列。GPU 槽位从不闲置。</p>

<p><strong>为何对 RLHF 至关重要</strong>：</p>

<ol>
<li><p>RLHF 生成多样输出（高温度）。长度方差巨大——有些回答 50 Token，有些 2000+。</p></li>
<li><p>无连续批处理：平均利用率 <span class="arithmatex">\(\sim\)</span>40–50%（等最慢的序列）。</p></li>
<li><p>有连续批处理：利用率 <span class="arithmatex">\(&gt;\)</span>90%。吞吐高 2–3<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p>RLHF 需要大 batch（每步 128+ 回答）。用静态批处理生成 128 个回答需要 max_tokens <span class="arithmatex">\(\times\)</span> 128 个串行步。连续批处理摊销了这一开销。</p></li>
</ol>

<p><strong>实现</strong>：vLLM 的调度器在每个解码步后检查。抢占：若有新的高优先级请求到达且显存已满，可将低优先级序列的 KV cache 换出到 CPU，稍后恢复。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（系统基础；系统架构）。</em></p>
</details></section>
