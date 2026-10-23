# 28.3　系统设计题 {#system-design-questions}

<section class="quiz-question" id="q-6"><h2 class="question-title">Q6：为训练 70B 模型设计一个 RLHF 系统。逐组件讲解。</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：基于 72 张 A100-80GB GPU 的三集群解耦架构：</p>

<p><strong>集群 1 ——生成（32 GPU）</strong>：</p>

<ul>
<li><p>8 个 vLLM 实例，每个 TP=4。PagedAttention + 投机解码（speculative decoding，1B 草稿模型）。</p></li>
<li><p>连续批处理，最多 256 条在飞序列。INT8 权重以省带宽。</p></li>
<li><p>输出：(prompt, response, 逐 Token log-probs)。吞吐量：<span class="arithmatex">\(\sim\)</span>500 回答/分钟。</p></li>
<li><p>无状态：仅从共享存储加载最新权重。恢复极简。</p></li>
</ul>

<p><strong>集群 2 ——打分（8 GPU）</strong>：</p>

<ul>
<li><p>Reward 模型（70B，INT8 = 70GB），4 GPU（TP=4）。</p></li>
<li><p>参考模型（70B，INT8），4 GPU（TP=4）。计算逐 Token 的 log-probs 以算 KL。</p></li>
<li><p>输出：reward 分数 + 逐 Token 的 KL。轻量级，批量推理。</p></li>
</ul>

<p><strong>集群 3 ——训练（32 GPU）</strong>：</p>

<ul>
<li><p>Policy 模型用 FSDP（ZeRO-3）。FlashAttention 2。梯度检查点。</p></li>
<li><p>从 buffer 消费已打分的 experience。PPO 更新：在 mini-batch=16 上跑 4 个 epoch。</p></li>
<li><p>每 50 步将更新后的权重推送到共享存储（异步、后台传输）。</p></li>
<li><p>每 100 步做一次异步 checkpoint（非阻塞写入 NVMe + S3 备份）。</p></li>
</ul>

<p><strong>连接结构</strong>：</p>

<ul>
<li><p>生成 <span class="arithmatex">\(\rightarrow\)</span> 打分：Ray/Redis 队列（每 batch <span class="arithmatex">\(\sim\)</span>10 MB：Token ID + log-probs）。</p></li>
<li><p>打分 <span class="arithmatex">\(\rightarrow\)</span> 训练：经验缓冲（循环式，保存最近 500 步）。</p></li>
<li><p>训练 <span class="arithmatex">\(\rightarrow\)</span> 生成：共享并行文件系统（Lustre/GPFS）上的权重存储。每 50 步异步推送 140GB。</p></li>
</ul>

<p><strong>重叠</strong>：当训练处理第 <span class="arithmatex">\(N\)</span> 步时，生成已经在为第 <span class="arithmatex">\(N+1\)</span> 步生产数据。这隐藏了生成延迟，相对单体架构吞吐提升 1.3–1.5<span class="arithmatex">\(\times\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>

<section class="quiz-question" id="q-7"><h2 class="question-title">Q7：如何处理生成瓶颈？量化各种方案的收益。</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：生成占 RLHF 总墙钟时间的 60–70%。根本原因：自回归解码受显存带宽约束（算术强度 <span class="arithmatex">\(\approx 1\)</span> FLOP/byte，而 A100 的 roofline 是 156 FLOP/byte）。</p>

<p><strong>按影响力排序的方案</strong>：</p>

<p><strong>1. 解耦生成与训练</strong>（端到端 1.3–1.5<span class="arithmatex">\(\times\)</span>）：在独立硬件上运行生成，与训练重叠。最大的单一架构性胜利。</p>

<p><strong>2. vLLM + PagedAttention</strong>（2–4<span class="arithmatex">\(\times\)</span>）：消除内部碎片造成的 60–80% KV cache 显存浪费。可承载 3–4<span class="arithmatex">\(\times\)</span> 更大的 batch = 更好的带宽利用。</p>

<p><strong>3. 连续批处理</strong>（1.5–2<span class="arithmatex">\(\times\)</span>）：不等最长序列结束。立即在腾出的槽位开新序列。保持 GPU 繁忙。</p>

<p><strong>4. 投机解码（Speculative Decoding）</strong>（2–3<span class="arithmatex">\(\times\)</span>）：1B 草稿模型提议 5 个 Token。70B 模型在一次前向中并行验证全部 5 个！平均接受 3–4 个 <span class="arithmatex">\(\rightarrow\)</span> 每次前向产出 3–4 个 Token 而非 1 个。</p>

<p><strong>5. 生成权重用 INT8/FP8</strong>（2<span class="arithmatex">\(\times\)</span>）：将每 Token 的 140GB 权重读取减半。质量损失极小，因为：(a) 我们本就在用温度采样，(b) 仅生成使用 INT8，训练仍是 BF16。</p>

<p><strong>6. CUDA graphs + kernel fusion</strong>（1.1–1.3<span class="arithmatex">\(\times\)</span>）：消除 Python/CUDA 启动开销。把 layernorm+attention+MLP 融合成更少的 kernel。</p>

<p><strong>组合后</strong>：<span class="arithmatex">\(1.5 \times 3 \times 1.5 \times 2.5 \times 2 \times 1.2 = 40\times\)</span> 优于朴素实现。实际中由于边际收益递减，整体大约为朴素实现的 10–20<span class="arithmatex">\(\times\)</span>。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（系统基础；系统架构）。</em></p>
</details></section>

<section class="quiz-question" id="q-8"><h2 class="question-title">Q8：解释解耦系统中的权重同步。能容忍多少陈旧度（staleness）？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>问题</strong>：生成集群使用 policy 权重生成回答，训练集群更新这些权重。它们位于不同硬件上。如何保持同步？</p>

<p><strong>为什么完美同步是浪费</strong>：70B BF16 全量同步 = 140GB。InfiniBand 400Gb/s（50GB/s）下，每次同步 2.8s。若每步（每 50–90s）都同步，权重传输占总时间的 3–5%。可接受，但没必要。</p>

<p><strong>陈旧度容忍分析</strong>：</p>

<ul>
<li><p>每步 policy 变化：<span class="arithmatex">\(\sim\)</span>0.1%（按参数均值差衡量）。</p></li>
<li><p>50 步：<span class="arithmatex">\(\sim\)</span>5% 累计漂移。</p></li>
<li><p>PPO 裁剪范围：可处理高达 20% 的概率比偏离。</p></li>
<li><p>经验：50 步陈旧度 <span class="arithmatex">\(\rightarrow\)</span> 质量下降 <span class="arithmatex">\(&lt;\)</span>2%（按胜率衡量）。</p></li>
</ul>

<p><strong>生产策略</strong>：</p>

<ol>
<li><p>每 50 训练步：向共享存储推送一次 BF16 完整 checkpoint（传输 2.8s）。</p></li>
<li><p>生成集群：在 batch 之间做非阻塞的权重重载。</p></li>
<li><p>增量压缩（可选）：仅发送变更参数（INT8 delta <span class="arithmatex">\(\approx\)</span> 5GB），作为偏移应用。带宽减少 10<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p>超大规模（256+ GPU）：流式同步——后台连续发送小块。平均陈旧度：5–10 步。</p></li>
</ol>

<p><strong>关键细节</strong>：生成阶段计算的 log-probs 使用的是陈旧权重。PPO 比率 <span class="arithmatex">\(\pi_\text{new}/\pi_\text{old}\)</span> 用这些陈旧 log-probs 作为 <span class="arithmatex">\(\pi_\text{old}\)</span>。这没问题，因为 PPO 本就被设计为处理 off-policy 修正。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>

<section class="quiz-question" id="q-9"><h2 class="question-title">Q9：在 512 GPU 规模下如何做到容错？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：在 512 GPU 规模下，MTBF 为 4–8 小时。5 天的训练会经历 15–30 次故障。</p>

<p><strong>架构级韧性</strong>：</p>

<ul>
<li><p>生成集群 = 无状态。失败实例在 <span class="arithmatex">\(&lt;\)</span>60s 内重启（只需加载权重，无状态）。</p></li>
<li><p>训练集群 = 有状态。需要基于 checkpoint 的恢复。</p></li>
<li><p>打分集群 = 无状态。与生成相同。</p></li>
</ul>

<p><strong>Checkpoint 策略</strong>：</p>

<ul>
<li><p>频率：每 50–100 步（5–10 分钟训练）。</p></li>
<li><p>方法：异步（非阻塞）。后台线程在下一步进行时写入。使用 FSDP 的分布式保存（每个 rank 并行保存自己的分片）。</p></li>
<li><p>内容：模型权重、优化器状态（Adam m/v）、LR scheduler、RNG 状态、KL 自适应系数、全局 step 计数器、回放缓冲指针。</p></li>
<li><p>存储：本地 NVMe（快，70B 仅 30s）+ 异步拷贝到 S3 / 共享 FS（持久化）。</p></li>
<li><p>保留：保留最近 3 个 checkpoint。自动删除更早的。</p></li>
</ul>

<p><strong>检测与恢复流程</strong>：</p>

<ol>
<li><p>NCCL 集合通信超时（60s）或心跳丢失（10s）<span class="arithmatex">\(\rightarrow\)</span> 检测到故障。</p></li>
<li><p>通过 NVML 健康检查定位故障节点。</p></li>
<li><p>方案 A（快，<span class="arithmatex">\(&lt;\)</span>2 min）：Torch Elastic 缩减 world size、重分配分片，以 <span class="arithmatex">\(N-1\)</span> 节点继续。后台请求替换。</p></li>
<li><p>方案 B（干净，<span class="arithmatex">\(\sim\)</span>5 min）：拉起替换节点、重建进程组、加载最近 checkpoint、恢复。</p></li>
<li><p>经验缓冲已持久化——无需重新生成。</p></li>
</ol>

<p><strong>预防</strong>：上线前的压力测试（GEMM、显存、NVLink）。ECC 错误监控（错误激增时抢占式迁移）。热备节点（环境预先加载）。双轨 InfiniBand 提供网络冗余。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>

<section class="quiz-question" id="q-10"><h2 class="question-title">Q10：如何从 7B 扩展到 70B 再到 405B？每个量级要变什么？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：</p>

<p><strong>7B（单 8-GPU 节点，数小时）</strong>：</p>

<ul>
<li><p>架构：单体（TRL 默认）。所有模型在同一组 GPU 上。</p></li>
<li><p>显存：LoRA + INT8 的 ref/RM 在 8<span class="arithmatex">\(\times\)</span>80GB 内轻松容纳。</p></li>
<li><p>并行：节点内 DP=8 或 FSDP。无网络通信。</p></li>
<li><p>超参：LR=<span class="arithmatex">\(5\times10^{-6}\)</span>，激进的 <span class="arithmatex">\(\beta\)</span>=0.02，50K–100K 步。</p></li>
<li><p>时间：每次运行 4–12 小时。快速迭代。</p></li>
</ul>

<p><strong>70B（32–64 GPU，2–5 天）</strong>：</p>

<ul>
<li><p>架构：半解耦。vLLM 生成 + FSDP 训练。</p></li>
<li><p>显存：ZeRO-3 必不可少。梯度检查点。INT8 ref/RM。</p></li>
<li><p>并行：节点内 TP=8（生成），节点间 FSDP（训练）。</p></li>
<li><p>超参：LR=<span class="arithmatex">\(1.5\times10^{-6}\)</span>，温和的 <span class="arithmatex">\(\beta\)</span>=0.05，10K–30K 步。</p></li>
<li><p>容错：异步 checkpoint、监控，但仍可人工管理。</p></li>
</ul>

<p><strong>405B（256–512 GPU，1–3 周）</strong>：</p>

<ul>
<li><p>架构：完全解耦。独立集群。权重存储 + 队列。</p></li>
<li><p>显存：训练用 ZeRO-3 + TP=8 + PP=2。生成用 INT4。</p></li>
<li><p>并行：3D 并行（TP<span class="arithmatex">\(\times\)</span>PP<span class="arithmatex">\(\times\)</span>DP = 8<span class="arithmatex">\(\times\)</span>2<span class="arithmatex">\(\times\)</span>16 = 256 GPU 训练）。</p></li>
<li><p>超参：LR=<span class="arithmatex">\(5\times10^{-7}\)</span>，非常保守的 <span class="arithmatex">\(\beta\)</span>=0.1，2K–5K 步。</p></li>
<li><p>容错：必备。弹性训练、冗余 checkpoint、热备节点。</p></li>
<li><p>关键变化：所需 RL 训练量大幅减少（模型预训练后已非常强）。但每步贵 50<span class="arithmatex">\(\times\)</span>，因此不稳定性是灾难性的。</p></li>
</ul>

<p><strong>悖论</strong>：更大的模型实际上<em>逐步 RL 训练更容易</em>（更稳定，loss 地形更平滑）。但不稳定性的代价随模型规模放大——405B 上一次失败的训练浪费 $100K+ 的算力。</p>

<p><em><strong>复习：</strong> 第 <a href="../11-training-infrastructure/index.md">11</a> 章（系统架构与大规模基础设施）。</em></p>
</details></section>
