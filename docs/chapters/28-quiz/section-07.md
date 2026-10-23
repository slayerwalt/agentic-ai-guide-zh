# 28.7　GPU 架构与硬件题 {#gpu-architecture-and-hardware-questions}

<section class="quiz-question" id="q-30"><h2 class="question-title">Q30：解释 GPU 内存层次结构。它对 LLM 推理为何重要？</h2>
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

<section class="quiz-question" id="q-31"><h2 class="question-title">Q31：FlashAttention 如何工作？什么是 online softmax 技巧？</h2>
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

<section class="quiz-question" id="q-32"><h2 class="question-title">Q32：解释 PagedAttention。它如何解决 KV cache 问题？</h2>
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

<section class="quiz-question" id="q-33"><h2 class="question-title">Q33：比较 NVLink 与 InfiniBand。在 RLHF 训练中各用于何处？</h2>
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
