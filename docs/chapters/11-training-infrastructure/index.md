# 第 11 章 · 大规模系统架构与基础设施 {#system-architecture-infrastructure-at-scale}

<p>用基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）训练 LLM，既是一道算法题，也是一道系统工程题。与标准 SFT 不同——后者只涉及单一模型、单次前向-反向传播、扩展规律也已被充分理解——RLHF 需要<em>同时</em>加载<em>多个模型</em>（policy、reference、reward model、value head），通过复杂的 rollout-打分-训练循环协调起来，并分布到数十乃至数百块 GPU 上。本章覆盖让大规模 RLHF 训练成为可能的系统级细节：显存预算、并行策略（Data、Tensor、Pipeline、Sequence 及其组合）、生成瓶颈、解耦式架构、权重同步、容错以及生产监控。</p>

## 11.1　4 模型显存挑战 {#the-4-model-memory-challenge}

<figure id="figure-11-1"><a href="../../assets/figures/fig_031_fig31.png" target="_blank" rel="noopener" aria-label="查看图 11.1 原图" title="点击查看原图"><img src="../../assets/figures/fig_031_fig31.png" alt="70B PPO 显存预算：RLHF 所需的四个模型及其显存占用。合计 1470–1560GB。朴素方案最少需要 19–20 块 A100-80GB。使用 ZeRO-3 时：8 个节点即可装下。" loading="lazy"></a><figcaption><span class="figure-number">图 11.1　</span>70B PPO 显存预算：RLHF 所需的四个模型及其显存占用。合计 1470–1560GB。朴素方案最少需要 19–20 块 A100-80GB。使用 ZeRO-3 时：8 个节点即可装下。</figcaption></figure>

<div class="admonition warning"><p class="admonition-title">显存预算现实核对——70B BF16</p>
<span id="table-11-inline-1"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col">Policy 权重（BF16）</th><th scope="col">140 GB</th></tr></thead>
<tbody><tr><td>FP32 master 权重</td><td>280 GB</td></tr>
<tr><td>Adam 优化器（m + v，FP32）</td><td>560 GB</td></tr>
<tr><td>梯度（BF16）</td><td>140 GB</td></tr>
<tr><td>Reference model</td><td>140 GB（INT8 时 70 GB）</td></tr>
<tr><td>Reward model</td><td>140 GB（INT8 时 70 GB）</td></tr>
<tr><td>激活值（batch 128，seq 2048）</td><td>50–100 GB</td></tr>
<tr><td>生成用 KV cache</td><td>20–60 GB</td></tr>
<tr><td><strong>合计</strong></td><td><strong>1470–1560 GB</strong></td></tr></tbody>
</table>
</div>

<p><span class="arithmatex">\(\div\)</span> 80 GB/GPU = <strong>至少 19–20 块 A100</strong>（尚未计入任何并行开销）。</p>
</div>

## 11.2　并行策略详解 {#sec:parallelism}

<p>训练大语言模型必须把计算分布到许多 GPU 上。可以沿<em>多个本质不同的维度</em>并行，每种都有各自的取舍。本节用数学公式、示意图和实践指引详细介绍每种策略。</p>

<figure id="figure-11-2"><a href="../../assets/figures/fig_032_fig32.png" target="_blank" rel="noopener" aria-label="查看图 11.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_032_fig32.png" alt="四种并行策略概览。生产系统通常会同时组合其中 2–3 种。" loading="lazy"></a><figcaption><span class="figure-number">图 11.2　</span>四种并行策略概览。生产系统通常会同时组合其中 2–3 种。</figcaption></figure>

### 11.2.1　数据并行（Data Parallelism, DP）与分布式数据并行（Distributed Data Parallelism, DDP） {#subsec:dp-ddp}

<p>数据并行是最简单、也最常见的分布式训练形式 <a class="book-citation" id="cite-li2020pytorch-1" href="../../references.md#ref-li2020pytorch" aria-label="参考文献 209">[209]</a>。每块 GPU 持有模型的<em>完整副本</em>，处理不同的 mini-batch，并同步梯度。</p>

##### 原始 DP（PyTorch <code>DataParallel</code>）。 {#vanilla-dp-pytorch-dataparallel.}

<p>单进程方案：一块 “主” GPU 负责分发输入、汇集输出、广播梯度。受 GIL 以及到主 GPU 的 PCIe 带宽限制。</p>

##### 分布式数据并行（DDP，<code>DistributedDataParallel</code>）。 {#distributed-data-parallelism-ddp-distributeddataparallel.}

<p>多进程方案：每块 GPU 各自运行一个进程。在反向计算继续进行的同时，通过 ring-AllReduce <a class="book-citation" id="cite-sergeev2018horovod-1" href="../../references.md#ref-sergeev2018horovod" aria-label="参考文献 321">[321]</a> 在后台同步梯度。</p>

<figure id="fig:ddp"><a href="../../assets/figures/fig_033_ddp.png" target="_blank" rel="noopener" aria-label="查看图 11.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_033_ddp.png" alt="DDP：每块 GPU 持有完整的模型副本并处理不同的 batch。梯度通过 ring AllReduce 取平均，与反向计算重叠。" loading="lazy"></a><figcaption><span class="figure-number">图 11.3　</span>DDP：每块 GPU 持有完整的模型副本并处理不同的 batch。梯度通过 ring AllReduce 取平均，与反向计算重叠。</figcaption></figure>

<p><strong>DDP 的关键特性</strong>：</p>

<ul>
<li><p><strong>显存</strong>：每块 GPU 都要存完整的模型 + 优化器 + 梯度。70B BF16 模型约 <span class="arithmatex">\(\sim\)</span>560 GB/GPU——如不做显存优化根本无法承受。</p></li>
<li><p><strong>通信</strong>：每步对梯度张量做一次 AllReduce。大小 = 模型参数量 <span class="arithmatex">\(\times\)</span> 2 字节（BF16）。Ring AllReduce 开销：每 GPU 传输 <span class="arithmatex">\(2 \cdot \frac{N-1}{N} \cdot M\)</span> 字节。</p></li>
<li><p><strong>扩展性</strong>：在 <span class="arithmatex">\(\sim\)</span>64 块 GPU 以内接近线性。超过之后通信开始占主导。</p></li>
<li><p><strong>梯度分桶</strong>：DDP 把参数按桶分组（默认 25 MB），只要某个桶的梯度就绪就启动 AllReduce——让通信与反向计算重叠。</p></li>
</ul>

<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">torch.distributed</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">dist</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">torch.nn.parallel</span><span class="w"> </span><span class="kn">import</span> <span class="n">DistributedDataParallel</span> <span class="k">as</span> <span class="n">DDP</span>

<span class="n">dist</span><span class="o">.</span><span class="n">init_process_group</span><span class="p">(</span><span class="n">backend</span><span class="o">=</span><span class="s2">"nccl"</span><span class="p">)</span>  <span class="c1"># NCCL 用于 GPU 间通信</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">to</span><span class="p">(</span><span class="n">local_rank</span><span class="p">)</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">DDP</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">device_ids</span><span class="o">=</span><span class="p">[</span><span class="n">local_rank</span><span class="p">],</span>
            <span class="n">gradient_as_bucket_view</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>    <span class="c1"># 显存优化</span>
            <span class="n">static_graph</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>               <span class="c1"># 开启通信优化</span>
</code></pre></div>

<div class="admonition warning"><p class="admonition-title">DP vs DDP —— 总是用 DDP</p>
<p>PyTorch 旧版的 <code>DataParallel</code>（DP）<strong>绝不应</strong>用于 LLM 训练：</p>

<ul>
<li><p>单进程，受 Python GIL 限制</p></li>
<li><p>所有梯度都汇集到 GPU 0（瓶颈）</p></li>
<li><p>即便在单节点上也比 DDP 慢 2–3<span class="arithmatex">\(\times\)</span></p></li>
<li><p>无法跨机扩展</p></li>
</ul>

<p>DDP 是<em>底线</em>并行策略。对于 <span class="arithmatex">\(&gt;\)</span>7B 的 LLM，应优先使用 FSDP/ZeRO。</p>
</div>

### 11.2.2　张量并行（Tensor Parallelism, TP） {#subsec:tensor-parallel}

<p>张量并行（Megatron-LM 风格 <a class="book-citation" id="cite-shoeybi2019megatron-1" href="../../references.md#ref-shoeybi2019megatron" aria-label="参考文献 327">[327]</a>）把<em>单个权重矩阵</em>切到多块 GPU 上。每块 GPU 计算部分结果，再通过 AllReduce 合并。</p>

##### 列并行（Column-Parallel）线性层。 {#column-parallel-linear-layer.}

<p>权重矩阵 <span class="arithmatex">\(W \in \mathbb{R}^{d \times h}\)</span> 按列切到 <span class="arithmatex">\(T\)</span> 块 GPU 上：</p>



<div class="arithmatex">\[W = [W_0 \;|\; W_1 \;|\; \cdots \;|\; W_{T-1}], \quad W_i \in \mathbb{R}^{d \times h/T}
\tag{11.1}\]</div>

<p>每块 GPU <span class="arithmatex">\(i\)</span> 独立计算 <span class="arithmatex">\(Y_i = XW_i\)</span>（无需通信）。输出沿 hidden 维度被切开。</p>

##### 行并行（Row-Parallel）线性层。 {#row-parallel-linear-layer.}

<p>权重矩阵按行切：<span class="arithmatex">\(W = [W_0; W_1; \ldots; W_{T-1}]\)</span>，其中 <span class="arithmatex">\(W_i \in \mathbb{R}^{d/T \times h}\)</span>。输入 <span class="arithmatex">\(X\)</span> 也必须切开。每块 GPU 计算一个部分和，再通过 <strong>AllReduce</strong> 得到最终输出。</p>

<figure id="figure-11-4"><a href="../../assets/figures/fig_034_tp-column.png" target="_blank" rel="noopener" aria-label="查看图 11.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_034_tp-column.png" alt="列并行线性层（TP=2）。权重按列切开；每块 GPU 独立计算 \(XW_i\)。MLP 将其与行并行层配对，从而避免多余的 AllReduce。" loading="lazy"></a><figcaption><span class="figure-number">图 11.4　</span>列并行线性层（TP=2）。权重按列切开；每块 GPU 独立计算 <span class="arithmatex">\(XW_i\)</span>。MLP 将其与行并行层配对，从而避免多余的 AllReduce。</figcaption></figure>

##### TP 下的 Transformer Block。 {#transformer-block-with-tp.}

<p>在 Transformer 层中，Megatron-LM 按以下方式应用 TP：</p>

<ol>
<li><p><strong>MLP</strong>：第一层线性（<span class="arithmatex">\(h \to 4h\)</span>）列并行，第二层线性（<span class="arithmatex">\(4h \to h\)</span>）行并行。行并行层之后做一次 AllReduce。</p></li>
<li><p><strong>Attention</strong>：<span class="arithmatex">\(Q\)</span>、<span class="arithmatex">\(K\)</span>、<span class="arithmatex">\(V\)</span> 投影做列并行（按 head 切到多块 GPU）。输出投影做行并行。输出投影之后做一次 AllReduce。</p></li>
<li><p><strong>合计</strong>：每个 Transformer 层 2 次 AllReduce（attention 一次，MLP 一次）。</p></li>
</ol>

<figure id="fig:tp-transformer"><a href="../../assets/figures/fig_035_tp-transformer.png" target="_blank" rel="noopener" aria-label="查看图 11.5 原图" title="点击查看原图"><img src="../../assets/figures/fig_035_tp-transformer.png" alt="单个 Transformer block 中的张量并行通信模式。每层需要两次 AllReduce 操作（红色标注）——attention 之后一次，MLP 之后一次。" loading="lazy"></a><figcaption><span class="figure-number">图 11.5　</span>单个 Transformer block 中的张量并行通信模式。每层需要两次 AllReduce 操作（红色标注）——attention 之后一次，MLP 之后一次。</figcaption></figure>

<div class="admonition tip"><p class="admonition-title">为什么 TP 必须限制在节点内</p>
<p>每个 Transformer 层需要 2 次 AllReduce 操作（上文标注为 <span class="arithmatex">\(f\)</span> 和 <span class="arithmatex">\(g\)</span>）。对于 80 层的 70B 模型，每次前向就是 160 次 AllReduce（算上反向共 320 次）。在 NVLink 速率（600 GB/s）下，每次 AllReduce 耗时 <span class="arithmatex">\(&lt;\)</span>0.5 ms。但在 InfiniBand（50 GB/s）上，同一操作要 <span class="arithmatex">\(\sim\)</span>4 ms，总开销 160 <span class="arithmatex">\(\times\)</span> 4 = 640 ms——<em>比计算本身还长</em>。</p>

<p><strong>规则</strong>：TP 度 <span class="arithmatex">\(\leq\)</span> 节点内 GPU 数（通常 TP <span class="arithmatex">\(\leq\)</span> 8）。跨节点扩展请用 DP/FSDP。</p>
</div>

<div class="admonition note"><p class="admonition-title">TP 度的选取</p>
<ul>
<li><p><strong>TP=1</strong>：不做张量并行。模型可放进单块 GPU（BF16 下通常 <span class="arithmatex">\(\leq\)</span> 13B）。</p></li>
<li><p><strong>TP=2</strong>：最小切分。适合在 2 块 GPU 上做 13–34B 推理。开销低（<span class="arithmatex">\(&lt;\)</span>5%）。</p></li>
<li><p><strong>TP=4</strong>：34–70B 推理的标准配置。开销 8–12%。</p></li>
<li><p><strong>TP=8</strong>：占满整节点。70B+ 训练的必备配置。开销 12–18%。</p></li>
<li><p><strong>TP<span class="arithmatex">\(&gt;\)</span>8</strong>：跨节点 TP。极少使用——仅在 PP 不足以承载的 200B+ 模型上才会用。开销 30–50%。</p></li>
</ul>

<p><strong>重要</strong>：attention head 数必须能被 TP 度整除。例如 LLaMA-70B（64 个 head），合法的 TP 取值为 1、2、4、8、16、32、64。</p>
</div>

### 11.2.3　序列并行（Sequence Parallelism, SP） {#subsec:sequence-parallel}

<p>序列并行（Sequence Parallelism, SP） <a class="book-citation" id="cite-korthikanti2023reducing-1" href="../../references.md#ref-korthikanti2023reducing" aria-label="参考文献 191">[191]</a> 解决的是张量并行单独无法消除的一个显存瓶颈：LayerNorm 和 Dropout 层中的<strong>激活值显存</strong>。</p>

##### 问题。 {#the-problem.}

<p>TP 把权重显存切到了多块 GPU 上。但 LayerNorm 和 Dropout 作用在<em>完整</em>的 hidden 维上，并在每块 GPU 上都重复存在。它们的激活值（反向传播需要）占用与 <span class="arithmatex">\(b \times s \times d\)</span> 成正比的显存——每块 GPU 上一样多，TP 并不会把它降下来。</p>

##### 解法。 {#the-solution.}

<p>对那些不需要跨 GPU 通信的操作（LayerNorm、Dropout、残差连接），按<em>序列维度</em>切分。每块 GPU 对这些操作只处理 <span class="arithmatex">\(s/T\)</span> 长的序列切片，仅在真正需要的地方（attention、线性层）再 gather 出完整序列。</p>

<figure id="fig:seq-parallel"><a href="../../assets/figures/fig_036_seq-parallel.png" target="_blank" rel="noopener" aria-label="查看图 11.6 原图" title="点击查看原图"><img src="../../assets/figures/fig_036_seq-parallel.png" alt="序列并行通过沿序列维度切分，降低 LayerNorm/Dropout 的激活显存。通信（AllGather/ReduceScatter）取代了标准 TP 中的 AllReduce——总传输字节数不变，但显存得以节省。" loading="lazy"></a><figcaption><span class="figure-number">图 11.6　</span>序列并行通过沿序列维度切分，降低 LayerNorm/Dropout 的激活显存。通信（AllGather/ReduceScatter）取代了标准 TP 中的 AllReduce——总传输字节数不变，但显存得以节省。</figcaption></figure>

<div class="admonition tip"><p class="admonition-title">SP 通信是 “免费” 的</p>
<p>标准 TP 在每个子层后做 AllReduce，等价于 ReduceScatter + AllGather。SP 只是<em>重排</em>了这些原语：</p>

<ul>
<li><p>不带 SP 的 TP：AllReduce（= ReduceScatter + AllGather）<span class="arithmatex">\(\rightarrow\)</span> 所有 GPU 上数据相同 <span class="arithmatex">\(\rightarrow\)</span> 在完整张量上算 LayerNorm（浪费）。</p></li>
<li><p>带 SP 的 TP：ReduceScatter <span class="arithmatex">\(\rightarrow\)</span> 每块 GPU 拿到 <span class="arithmatex">\(1/T\)</span> 的序列 <span class="arithmatex">\(\rightarrow\)</span> 在部分张量上算 LayerNorm <span class="arithmatex">\(\rightarrow\)</span> 进入下一个 TP 层前再 AllGather。</p></li>
</ul>

<p>总通信量完全相同！SP 是<strong>零额外通信开销</strong>的纯显存优化，使用 TP 时应当始终开启。</p>
</div>

<p><strong>SP 的显存节省（70B 模型，TP=8，batch=4，seq=2048）</strong>：</p>



<div class="arithmatex">\[\text{激活值节省} = (T-1) \times b \times s \times d \times n_\text{layers} \times 2\text{ bytes} = 7 \times 4 \times 2048 \times 8192 \times 80 \times 2 \approx \textbf{59 GB/GPU}
\tag{11.2}\]</div>

### 11.2.4　流水线并行（Pipeline Parallelism, PP） {#subsec:pipeline-parallel}

<p>流水线并行把模型按层<em>纵向</em>切分，把连续的层组指派给不同的设备（stage）。激活值在 stage 间向前流动，梯度向后流动。</p>

##### 气泡（Bubble）问题。 {#the-bubble-problem.}

<p>朴素流水线执行会产生 “气泡”——某个 stage 等待前一 stage 的输入或后一 stage 的梯度时的空闲时间：</p>

<figure id="fig:pipeline-bubble"><a href="../../assets/figures/fig_037_pipeline-bubble.png" target="_blank" rel="noopener" aria-label="查看图 11.7 原图" title="点击查看原图"><img src="../../assets/figures/fig_037_pipeline-bubble.png" alt="流水线气泡对比。左：朴素流水线只有一个 micro-batch，75% 时间空闲。右：\(M=4\) 个 micro-batch 的 GPipe 大幅缩小气泡。当 \(M \gg P\) 时，气泡占比趋近于 0。" loading="lazy"></a><figcaption><span class="figure-number">图 11.7　</span>流水线气泡对比。左：朴素流水线只有一个 micro-batch，75% 时间空闲。右：<span class="arithmatex">\(M=4\)</span> 个 micro-batch 的 GPipe 大幅缩小气泡。当 <span class="arithmatex">\(M \gg P\)</span> 时，气泡占比趋近于 0。</figcaption></figure>

##### 气泡占比公式。 {#bubble-fraction-formula.}

<p>对于 <span class="arithmatex">\(P\)</span> 个流水线 stage、每步 <span class="arithmatex">\(M\)</span> 个 micro-batch：</p>



<div class="arithmatex">\[\text{气泡占比} = \frac{P - 1}{P + M - 1} \approx \frac{P-1}{M} \quad \text{（当 } M \gg P\text{ 时）}
\tag{11.3}\]</div>

<p>要让气泡开销 <span class="arithmatex">\(&lt;\)</span>10%，需要 <span class="arithmatex">\(M \geq 10 \cdot (P-1)\)</span>。PP=4 时：至少需要 30 个 micro-batch。</p>

##### 流水线调度。 {#pipeline-schedules.}

<p class="table-caption" id="table-11-1"><span class="table-number">表 11.1　</span>流水线调度策略</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>调度</strong></th><th scope="col"><strong>气泡</strong></th><th scope="col"><strong>显存</strong></th><th scope="col"><strong>特性</strong></th></tr></thead>
<tbody><tr><td>GPipe</td><td><span class="arithmatex">\(\frac{P-1}{M+P-1}\)</span></td><td><span class="arithmatex">\(M \times\)</span> 激活值</td><td>简单；先全部前向再全部反向 <a class="book-citation" id="cite-huang2019gpipe-1" href="../../references.md#ref-huang2019gpipe" aria-label="参考文献 152">[152]</a></td></tr>
<tr><td>1F1B</td><td><span class="arithmatex">\(\frac{P-1}{M+P-1}\)</span></td><td><span class="arithmatex">\(P \times\)</span> 激活值</td><td>交错；稳态显存有界 <a class="book-citation" id="cite-narayanan2019pipedream-1" href="../../references.md#ref-narayanan2019pipedream" aria-label="参考文献 263">[263]</a></td></tr>
<tr><td>Interleaved 1F1B</td><td><span class="arithmatex">\(\frac{P-1}{M \cdot V + P - 1}\)</span></td><td><span class="arithmatex">\(P \times\)</span> 激活值</td><td>虚拟 stage（<span class="arithmatex">\(V\)</span>）；进一步减小气泡 <a class="book-citation" id="cite-narayanan2021efficient-1" href="../../references.md#ref-narayanan2021efficient" aria-label="参考文献 264">[264]</a></td></tr>
<tr><td>Zero-Bubble（ZB-H1）</td><td><span class="arithmatex">\(\approx 0\)</span></td><td><span class="arithmatex">\(P \times\)</span> 激活值</td><td>把反向拆成 B 和 W 两个阶段 <a class="book-citation" id="cite-qi2023zerobubble-1" href="../../references.md#ref-qi2023zerobubble" aria-label="参考文献 296">[296]</a></td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">1F1B：生产标准</p>
<p><strong>1F1B</strong>（one-forward-one-backward）调度 <a class="book-citation" id="cite-narayanan2019pipedream-2" href="../../references.md#ref-narayanan2019pipedream" aria-label="参考文献 263">[263]</a> 被大多数生产系统采用（Megatron-LM <a class="book-citation" id="cite-narayanan2021efficient-2" href="../../references.md#ref-narayanan2021efficient" aria-label="参考文献 264">[264]</a>、DeepSpeed <a class="book-citation" id="cite-rajbhandari2020zero-1" href="../../references.md#ref-rajbhandari2020zero" aria-label="参考文献 304">[304]</a>）：</p>

<p><strong>预热</strong>：前向先填满流水线（P-1 个 micro-batch）。</p>

<p><strong>稳态</strong>：每个时间槽里交替执行一次前向和一次反向。这把激活值峰值显存限制在 <span class="arithmatex">\(P\)</span> 个 micro-batch（GPipe 是 <span class="arithmatex">\(M\)</span> 个）。</p>

<p><strong>收尾</strong>：剩余的反向把流水线排空。</p>

<p><strong>显存优势</strong>：GPipe 必须同时保存<em>全部</em> <span class="arithmatex">\(M\)</span> 个 micro-batch 的激活值。1F1B 在稳态下只保存 <span class="arithmatex">\(P\)</span> 组激活值——当 <span class="arithmatex">\(M = 32\)</span> 而 <span class="arithmatex">\(P = 4\)</span> 时至关重要。</p>
</div>

##### PP 中的通信。 {#communication-in-pp.}

<p>与 TP（AllReduce）不同，PP 只需要在相邻 stage 之间做激活值的<strong>点对点（point-to-point）</strong>通信：</p>



<div class="arithmatex">\[\text{每次传输的数据} = b_\text{micro} \times s \times d \times 2\text{ bytes (BF16)}
\tag{11.4}\]</div>

<p>若 micro-batch=4、seq=2048、<span class="arithmatex">\(d\)</span>=8192：每次传输 <span class="arithmatex">\(4 \times 2048 \times 8192 \times 2 = 128\)</span> MB。InfiniBand 50 GB/s 时：每次 2.6 ms——相对每个 stage 的计算量来说很小。</p>

##### 负载均衡。 {#load-balancing.}

<p>并非所有层的计算量都相等：</p>

<ul>
<li><p><strong>Embedding 层</strong>：非常便宜（查表）。</p></li>
<li><p><strong>Transformer block</strong>：计算量均匀。</p></li>
<li><p><strong>最末的 LM head</strong>：中等（词表投影是一次大型矩阵乘法）。</p></li>
</ul>

<p>把更多 Transformer 层分配给中间 stage，第一/最后 stage 少放一些，以平衡计算。</p>

### 11.2.5　完全分片数据并行（Fully Sharded Data Parallelism, FSDP / ZeRO-3） {#fully-sharded-data-parallelism-fsdp-zero-3}

<p>FSDP <a class="book-citation" id="cite-zhao2023pytorch-1" href="../../references.md#ref-zhao2023pytorch" aria-label="参考文献 431">[431]</a>（PyTorch）和 ZeRO-3 <a class="book-citation" id="cite-rajbhandari2020zero-2" href="../../references.md#ref-rajbhandari2020zero" aria-label="参考文献 304">[304]</a>（DeepSpeed）解决了 DDP 固有的显存重复问题：不再让每块 GPU 都保存完整的参数、梯度和优化器状态，而是每块 GPU 只拥有 <span class="arithmatex">\(1/N\)</span> 的切片，需要时即时重建完整张量。</p>

<figure id="fig:fsdp"><a href="../../assets/figures/fig_038_fsdp.png" target="_blank" rel="noopener" aria-label="查看图 11.8 原图" title="点击查看原图"><img src="../../assets/figures/fig_038_fsdp.png" alt="FSDP 把所有模型状态切片到多块 GPU 上。每块 GPU 拥有 \(1/N\) 的参数、优化器状态和梯度。在每层计算前，通过 AllGather 按需重建完整参数。" loading="lazy"></a><figcaption><span class="figure-number">图 11.8　</span>FSDP 把所有模型状态切片到多块 GPU 上。每块 GPU 拥有 <span class="arithmatex">\(1/N\)</span> 的参数、优化器状态和梯度。在每层计算前，通过 AllGather 按需重建完整参数。</figcaption></figure>

<p><strong>FSDP 每层的执行流程：</strong></p>

<ol>
<li><p><strong>前向</strong>：AllGather 参数 <span class="arithmatex">\(\rightarrow\)</span> 计算 <span class="arithmatex">\(\rightarrow\)</span> 丢弃非自有的分片。</p></li>
<li><p><strong>反向</strong>：再次 AllGather 参数 <span class="arithmatex">\(\rightarrow\)</span> 计算梯度 <span class="arithmatex">\(\rightarrow\)</span> ReduceScatter 梯度（每块 GPU 拿到自己的梯度分片）<span class="arithmatex">\(\rightarrow\)</span> 丢弃非自有的参数分片。</p></li>
<li><p><strong>优化器更新</strong>：每块 GPU 仅用自己持有的梯度分片和优化器状态更新自己拥有的分片。</p></li>
</ol>

<p class="table-caption" id="table-11-2"><span class="table-number">表 11.2　</span>显存对比：DDP vs FSDP/ZeRO 各阶段（70B 模型，8 GPU）。基线：BF16 参数（140 GB）+ BF16 梯度（140 GB）+ FP32 master+m+v（840 GB）= 每 GPU 1120 GB。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>分片对象</strong></th><th scope="col"><strong>每 GPU 显存</strong></th><th scope="col"><strong>通信</strong></th></tr></thead>
<tbody><tr><td>DDP（不分片）</td><td>无</td><td>1120 GB <span class="arithmatex">\(\times\)</span></td><td>AllReduce（仅梯度）</td></tr>
<tr><td>ZeRO-1</td><td>优化器状态</td><td>385 GB <span class="arithmatex">\(\times\)</span></td><td>AllReduce（梯度）</td></tr>
<tr><td>ZeRO-2</td><td>优化器 + 梯度</td><td>368 GB <span class="arithmatex">\(\times\)</span></td><td>AllReduce（梯度）</td></tr>
<tr><td>ZeRO-3 / FSDP</td><td>全部</td><td><strong>140 GB</strong> ✓</td><td>AllGather + ReduceScatter（每层）</td></tr></tbody>
</table>
</div>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">functools</span><span class="w"> </span><span class="kn">import</span> <span class="n">partial</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">torch.distributed.fsdp</span><span class="w"> </span><span class="kn">import</span> <span class="n">FullyShardedDataParallel</span> <span class="k">as</span> <span class="n">FSDP</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">torch.distributed.fsdp</span><span class="w"> </span><span class="kn">import</span> <span class="n">ShardingStrategy</span><span class="p">,</span> <span class="n">MixedPrecision</span><span class="p">,</span> <span class="n">BackwardPrefetch</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">torch.distributed.fsdp.wrap</span><span class="w"> </span><span class="kn">import</span> <span class="n">transformer_auto_wrap_policy</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">transformers.models.llama.modeling_llama</span><span class="w"> </span><span class="kn">import</span> <span class="n">LlamaDecoderLayer</span>

<span class="c1"># 用 FSDP 包装模型</span>
<span class="n">auto_wrap</span> <span class="o">=</span> <span class="n">partial</span><span class="p">(</span><span class="n">transformer_auto_wrap_policy</span><span class="p">,</span>
                    <span class="n">transformer_layer_cls</span><span class="o">=</span><span class="p">{</span><span class="n">LlamaDecoderLayer</span><span class="p">})</span>
<span class="n">mp_policy</span> <span class="o">=</span> <span class="n">MixedPrecision</span><span class="p">(</span>
    <span class="n">param_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
    <span class="n">reduce_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
    <span class="n">buffer_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">model</span> <span class="o">=</span> <span class="n">FSDP</span><span class="p">(</span>
    <span class="n">model</span><span class="p">,</span>
    <span class="n">sharding_strategy</span><span class="o">=</span><span class="n">ShardingStrategy</span><span class="o">.</span><span class="n">FULL_SHARD</span><span class="p">,</span>  <span class="c1"># ZeRO-3</span>
    <span class="n">mixed_precision</span><span class="o">=</span><span class="n">mp_policy</span><span class="p">,</span>
    <span class="n">auto_wrap_policy</span><span class="o">=</span><span class="n">auto_wrap</span><span class="p">,</span>  <span class="c1"># 对每个 transformer 层都做包装</span>
    <span class="n">use_orig_params</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>        <span class="c1"># torch.compile 兼容所需</span>
    <span class="n">limit_all_gathers</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>      <span class="c1"># 限制显存峰值（同一时刻只允许 1 个 AllGather）</span>
    <span class="n">forward_prefetch</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>       <span class="c1"># 当前层计算时预取下一层参数</span>
    <span class="n">backward_prefetch</span><span class="o">=</span><span class="n">BackwardPrefetch</span><span class="o">.</span><span class="n">BACKWARD_PRE</span><span class="p">,</span>  <span class="c1"># 反向时预取</span>
<span class="p">)</span>
</code></pre></div>

<div class="admonition warning"><p class="admonition-title">FSDP 的通信量</p>
<p>FSDP 每步的通信量是 DDP 的 <strong>3<span class="arithmatex">\(\times\)</span></strong>：</p>

<ul>
<li><p>DDP：1 次梯度 AllReduce = 环上合计 <span class="arithmatex">\(2M\)</span> 字节（其中 <span class="arithmatex">\(M\)</span> = 模型字节数）。</p></li>
<li><p>FSDP：2 次 AllGather（前向 + 反向）+ 1 次 ReduceScatter = <span class="arithmatex">\(3M\)</span> 字节。</p></li>
</ul>

<p>这就是显存与通信的取舍。FSDP 值得使用的场景是：(a) 模型在 DDP 下无法装进 GPU 显存；或 (b) 通信与计算重叠得很好（现代框架可达到 70–90% 重叠）。</p>
</div>

### 11.2.6　3D 并行：组合多种策略 {#subsec:3d-parallel}

<p>大规模生产系统（70B+）会同时组合 TP、PP 和 DP/FSDP：</p>

<figure id="fig:3d-parallel"><a href="../../assets/figures/fig_039_3d-parallel.png" target="_blank" rel="noopener" aria-label="查看图 11.9 原图" title="点击查看原图"><img src="../../assets/figures/fig_039_3d-parallel.png" alt="16 块 GPU 上的 3D 并行布局：TP=4（每个方框内，走 NVLink）、PP=2（橙色箭头，stage 之间）、DP=2（红色箭头，梯度同步）。每个维度利用通信层级中不同的一层。" loading="lazy"></a><figcaption><span class="figure-number">图 11.9　</span>16 块 GPU 上的 3D 并行布局：TP=4（每个方框内，走 NVLink）、PP=2（橙色箭头，stage 之间）、DP=2（红色箭头，梯度同步）。每个维度利用通信层级中不同的一层。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">生产配方：64 块 A100-80GB（8 节点）上的 70B</p>
<p><strong>节点内</strong>（NVLink 600GB/s）：生成阶段用 TP=8；训练阶段在节点内使用 FSDP。<br><strong>节点间</strong>（InfiniBand 400Gb/s）：跨节点 FSDP（8 路数据并行）。<br><strong>结果</strong>：每块 GPU 持有 <span class="arithmatex">\(\sim\)</span>70GB。Policy 权重在前向/反向时按层 gather。<br><strong>流水线并行</strong>：仅当模型超过 100B+ 且 TP+ZeRO 装不下时才用。会引入复杂度（气泡开销 10–20%）和调度困难。</p>

<p><strong>决策流程</strong>：</p>

<ol>
<li><p>模型能装进 1 块 GPU 吗？<span class="arithmatex">\(\rightarrow\)</span> 用 DDP。</p></li>
<li><p>用 FSDP 能装进 1 个节点吗？<span class="arithmatex">\(\rightarrow\)</span> 用 FSDP（ZeRO-3）。</p></li>
<li><p>用 TP+FSDP 能装进 1 个节点吗？<span class="arithmatex">\(\rightarrow\)</span> 节点内用 TP + 节点间用 FSDP。</p></li>
<li><p>仍然装不下？<span class="arithmatex">\(\rightarrow\)</span> 跨节点再加 PP。这是最后的手段。</p></li>
</ol>
</div>

<p class="table-caption" id="table-11-3"><span class="table-number">表 11.3　</span>并行策略对比小结</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>切分对象</strong></th><th scope="col"><strong>通信</strong></th><th scope="col"><strong>扩展上限</strong></th><th scope="col"><strong>开销</strong></th><th scope="col"><strong>使用时机</strong></th></tr></thead>
<tbody><tr><td>DP/DDP</td><td>Batch</td><td>AllReduce（梯度）</td><td><span class="arithmatex">\(\sim\)</span>64 GPU</td><td>5–10%</td><td>模型可装进 1 GPU</td></tr>
<tr><td>FSDP</td><td>参数+优化器+梯度</td><td>AllGather+RS</td><td>数百 GPU</td><td>10–20%</td><td><span class="arithmatex">\(&gt;\)</span>13B 时的默认</td></tr>
<tr><td>TP</td><td>权重矩阵</td><td>AllReduce（每层 2 次）</td><td>8 GPU（1 节点）</td><td>12–18%</td><td>大模型推理+训练</td></tr>
<tr><td>SP</td><td>激活（序列维）</td><td>复用 TP 通信</td><td>同 TP</td><td><span class="arithmatex">\(\approx\)</span>0% 额外</td><td>与 TP 一起常开</td></tr>
<tr><td>PP</td><td>按层切 stage</td><td>点对点</td><td><span class="arithmatex">\(\sim\)</span>16 stage</td><td>15–30%</td><td>仅 100B+ 模型</td></tr></tbody>
</table>
</div>

### 11.2.7　解耦 DiLoCo：跨数据中心训练 {#subsec:decoupled-diloco}

<p>常规分布式训练假设高带宽互联——单个数据中心内的 100+ Gbps InfiniBand。这一要求使得用标准互联网链路跨地理分布式数据中心训练成为不可能。<strong>解耦 DiLoCo（Decoupled DiLoCo）</strong> <a class="book-citation" id="cite-diloco2026google-1" href="../../references.md#ref-diloco2026google" aria-label="参考文献 114">[114]</a>（Google DeepMind，2026 年 4 月）打破了这一假设，让大规模训练可以在普通带宽下跨区域进行。</p>

<div class="admonition note"><p class="admonition-title">DiLoCo：核心思想</p>
<p><strong>DiLoCo</strong>（带压缩的分布式本地 SGD，Distributed Local SGD with Compression）用周期性的外部更新取代连续梯度同步：</p>

<ol>
<li><p>每个 Worker <span class="arithmatex">\(i\)</span> 在本地数据分片上独立训练 <span class="arithmatex">\(H\)</span> 个内部步骤。</p></li>
<li><p><span class="arithmatex">\(H\)</span> 步之后，Worker 们通过计算一个<strong>外部更新（outer update）</strong>来同步：</p>



<div class="arithmatex">\[\Delta_{\text{outer}} = \frac{1}{N}\sum_{i=1}^N \left(\theta_i^{(H)} - \theta_i^{(0)}\right)\]</div>

<p>其中 <span class="arithmatex">\(N\)</span> 是 Worker 数量，<span class="arithmatex">\(\theta_i^{(0)}\)</span> 是共享的起始检查点。</p></li>
<li><p>外部更新通过一个全局优化器（如 Nesterov 动量）应用，所有 Worker 重置到新的共享 <span class="arithmatex">\(\theta\)</span>。</p></li>
</ol>

<p>只有外部更新（每 <span class="arithmatex">\(H\)</span> 步一个模型大小的张量）会穿越网络——而不是逐步骤的梯度。</p>
</div>

<p><strong>解耦 DiLoCo</strong> 进一步解耦了外部同步：Worker 无需同时同步。相反，更新跨区域异步进行，由一个协调者在更新到达时应用。这能够容忍广域网的可变延迟。</p>

<div class="admonition example"><p class="admonition-title">解耦 DiLoCo：实证结果</p>
<p>一个 12B 参数模型在由标准互联网链路（2–5 Gbps）连接的 4 个美国地理区域上完成训练：</p>

<ul>
<li><p><strong>带宽降低</strong>：比同步分布式训练少用 236<span class="arithmatex">\(\times\)</span> 的带宽。</p></li>
<li><p><strong>速度</strong>：比完全异步 SGD（受陈旧梯度困扰）快 20<span class="arithmatex">\(\times\)</span> 的墙钟时间。</p></li>
<li><p><strong>模型质量</strong>：在标准语言建模基准上与单数据中心训练持平。</p></li>
<li><p><strong>容错性</strong>：Worker 故障只会延迟外部更新，而不会拖垮整个训练。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">为什么这很重要：主权 AI 与多云</p>
<p>解耦 DiLoCo 的意义远超效率本身：</p>

<ul>
<li><p><strong>主权 AI（Sovereign AI）</strong>：国家或组织可以从自己的数据中心贡献算力，而无需让敏感数据经过中央设施。</p></li>
<li><p><strong>多云训练</strong>：同时利用 AWS、GCP 与 Azure 上的竞价实例，绕开故障。</p></li>
<li><p><strong>数据中心韧性</strong>：一个数据中心宕机只会丢失该 Worker 在当前 <span class="arithmatex">\(H\)</span> 步窗口内的贡献，而非整个训练。</p></li>
<li><p><strong>异构硬件</strong>：不同 GPU 世代的 Worker 都可以参与，各自按自己的节奏运行。</p></li>
</ul>

<p>关键洞见是：现代优化器对低频、带噪声的外部更新具有鲁棒性——内部训练步骤提供了足够的局部信号，外部同步只需要纠正分布漂移即可。</p>
</div>

## 11.3　生成瓶颈：定量分析 {#the-generation-bottleneck-quantitative-analysis}

<div class="admonition tip"><p class="admonition-title">Roofline 分析：为什么生成是显存带宽瓶颈</p>
<p><strong>A100 参数</strong>：312 TFLOPS（BF16 tensor core），2 TB/s HBM 带宽。</p>

<p><strong>Roofline 拐点</strong>：<span class="arithmatex">\(312\text{T} / 2\text{T} = 156\)</span> FLOP/byte。低于 156 FLOP/byte 的操作受<em>显存带宽</em>约束。</p>

<p><strong>自回归生成</strong>：每生成一个 token，需要读完所有权重（70B 模型为 140GB），并执行 <span class="arithmatex">\(2 \times 70\text{B} = 140\text{G}\)</span> FLOPs（batch=1 时）。</p>

<p><strong>算术强度</strong>：<span class="arithmatex">\(140\text{G FLOP} / 140\text{GB} = 1\)</span> FLOP/byte。比 roofline 低了 <span class="arithmatex">\(156\times\)</span>！</p>

<p><strong>利用率</strong>：<span class="arithmatex">\(1/156 = 0.6\%\)</span> 的峰值 FLOPS。GPU 有 99.4% 的时间在等显存读。</p>

<p><strong>Token 速率</strong>：<span class="arithmatex">\(2\text{TB/s} / 140\text{GB} = 14.3\)</span> tokens/秒（单流，batch=1）。</p>

<p><strong>生成 512 token</strong>：<span class="arithmatex">\(512 / 14.3 = 35.8\)</span> 秒每条响应（batch=1，TP=1）。</p>

<p><strong>Batch 化能改善</strong>：Batch=64、TP=4 <span class="arithmatex">\(\rightarrow\)</span> 只读一次权重，并行生成 64 个 token。算术强度：<span class="arithmatex">\(64 \times 1 = 64\)</span> FLOP/byte。好一些，但仍在 roofline 之下！</p>
</div>

<p class="table-caption" id="table-11-4"><span class="table-number">表 11.4　</span>70B 模型在各种配置下的生成吞吐（512 token）</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>配置</strong></th><th scope="col"><strong>Batch</strong></th><th scope="col"><strong>每 batch 用时</strong></th><th scope="col"><strong>Tok/s/GPU</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>TP=1, batch=1</td><td>1</td><td>36s</td><td>14</td><td>基线，最差情况</td></tr>
<tr><td>TP=4, batch=1</td><td>1</td><td>9s</td><td>57</td><td>生成上 TP 线性扩展</td></tr>
<tr><td>TP=4, batch=32</td><td>32</td><td>15s</td><td>1092</td><td>接近最佳 batch</td></tr>
<tr><td>TP=4, batch=128, vLLM</td><td>128</td><td>45s</td><td>1456</td><td>连续 batching</td></tr>
<tr><td>TP=4, batch=128, INT8</td><td>128</td><td>25s</td><td>2621</td><td>带宽减半</td></tr></tbody>
</table>
</div>

<p><strong>优化栈</strong>（累乘加速比）：</p>

<ol>
<li><p><strong>vLLM + PagedAttention</strong> <a class="book-citation" id="cite-kwon2023efficient-1" href="../../references.md#ref-kwon2023efficient" aria-label="参考文献 195">[195]</a>（2–4<span class="arithmatex">\(\times\)</span>）：消除 KV cache 碎片，使更大 batch 成为可能</p></li>
<li><p><strong>连续 batching（continuous batching）</strong> <a class="book-citation" id="cite-yu2022orca-1" href="../../references.md#ref-yu2022orca" aria-label="参考文献 412">[412]</a>（1.5–2<span class="arithmatex">\(\times\)</span>）：不必等最长序列结束；一旦有序列结束就插入新的</p></li>
<li><p><strong>推测解码（Speculative decoding）</strong> <a class="book-citation" id="cite-leviathan2023fast-1" href="../../references.md#ref-leviathan2023fast" aria-label="参考文献 205">[205]</a>（2–3<span class="arithmatex">\(\times\)</span>）：小 draft 模型一次猜 5 个 token，大模型一次前向就能验证。平均接受 3–4 个。</p></li>
<li><p><strong>生成用 INT8/FP8 权重</strong>（2<span class="arithmatex">\(\times\)</span>）：带宽需求减半。因为我们在做采样（不是为训练算精确 logits），质量损失极小。</p></li>
<li><p><strong>CUDA graphs</strong>（1.1–1.3<span class="arithmatex">\(\times\)</span>）：消除固定形状操作的 kernel 启动开销</p></li>
<li><p><strong>Prefix caching</strong>（共享前缀 prompt 下 1.5<span class="arithmatex">\(\times\)</span>）：不重新计算 system prompt 的 KV cache</p></li>
</ol>

<div class="highlight"><pre><span></span><code><span class="c1"># 生产环境的 vLLM 生成配置</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">vllm</span><span class="w"> </span><span class="kn">import</span> <span class="n">LLM</span><span class="p">,</span> <span class="n">SamplingParams</span>

<span class="n">engine</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"./policy_checkpoint"</span><span class="p">,</span>
    <span class="n">tensor_parallel_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>           <span class="c1"># 每个实例 TP=4</span>
    <span class="n">gpu_memory_utilization</span><span class="o">=</span><span class="mf">0.92</span><span class="p">,</span>      <span class="c1"># 给 KV cache 留余量</span>
    <span class="n">max_num_batched_tokens</span><span class="o">=</span><span class="mi">16384</span><span class="p">,</span>     <span class="c1"># 在飞 token 上限</span>
    <span class="n">max_num_seqs</span><span class="o">=</span><span class="mi">256</span><span class="p">,</span>                 <span class="c1"># 并发序列上限</span>
    <span class="n">dtype</span><span class="o">=</span><span class="s2">"bfloat16"</span><span class="p">,</span>
    <span class="n">enable_prefix_caching</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>       <span class="c1"># 缓存 system prompt 的 KV</span>
    <span class="n">speculative_model</span><span class="o">=</span><span class="s2">"./draft_1B"</span><span class="p">,</span>   <span class="c1"># 推测解码</span>
    <span class="n">num_speculative_tokens</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span>
    <span class="n">block_size</span><span class="o">=</span><span class="mi">16</span><span class="p">,</span>                    <span class="c1"># PagedAttention block 大小</span>
    <span class="n">swap_space</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>                     <span class="c1"># 用于抢占的 swap 空间（GB）</span>
<span class="p">)</span>

<span class="c1"># 为 RLHF batch 生成响应</span>
<span class="n">sampling_params</span> <span class="o">=</span> <span class="n">SamplingParams</span><span class="p">(</span>
    <span class="n">temperature</span><span class="o">=</span><span class="mf">0.7</span><span class="p">,</span> <span class="n">top_p</span><span class="o">=</span><span class="mf">0.9</span><span class="p">,</span> <span class="n">max_tokens</span><span class="o">=</span><span class="mi">512</span><span class="p">,</span>
    <span class="n">logprobs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>  <span class="c1"># PPO 的 ratio 计算需要 log-prob</span>
<span class="p">)</span>
<span class="n">outputs</span> <span class="o">=</span> <span class="n">engine</span><span class="o">.</span><span class="n">generate</span><span class="p">(</span><span class="n">prompts</span><span class="p">,</span> <span class="n">sampling_params</span><span class="p">)</span>
<span class="c1"># 提取：responses 以及每个 token 的 log_probs（PPO/GRPO 所需）</span>
</code></pre></div>

## 11.4　解耦式架构：生产级设计 {#decoupled-architecture-production-design}

<p>诸如 DeepSpeed-Chat <a class="book-citation" id="cite-yao2023deepspeedchat-1" href="../../references.md#ref-yao2023deepspeedchat" aria-label="参考文献 410">[410]</a> 与 OpenRLHF <a class="book-citation" id="cite-hu2024openrlhf-1" href="../../references.md#ref-hu2024openrlhf" aria-label="参考文献 148">[148]</a> 这样的生产级 RLHF 系统采用<strong>解耦式架构</strong>，将生成、打分、训练拆成三个可独立扩展的集群。</p>

<figure id="figure-11-10"><a href="../../assets/figures/fig_040_fig40.png" target="_blank" rel="noopener" aria-label="查看图 11.10 原图" title="点击查看原图"><img src="../../assets/figures/fig_040_fig40.png" alt="解耦式 RLHF 架构。每个集群针对自己的负载做优化。打分后的 rollout 先在经验缓冲（experience buffer）中累积，再被训练消费。" loading="lazy"></a><figcaption><span class="figure-number">图 11.10　</span>解耦式 RLHF 架构。每个集群针对自己的负载做优化。打分后的 rollout 先在经验缓冲（experience buffer）中累积，再被训练消费。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">为什么要解耦？</p>
<p><strong>生成</strong>受显存带宽约束（需要高速 HBM，浪费算力）。<br><strong>训练</strong>受算力约束（需要 tensor core，反向时浪费带宽）。<br><strong>同一硬件无法同时优化两者</strong>：若把所有事情放在一起，要么生成时浪费算力，要么训练时浪费带宽。解耦让每个集群用最优的硬件/配置。</p>

<p><strong>实际好处</strong>：</p>

<ul>
<li><p>生成和训练可以独立扩展</p></li>
<li><p>生成集群无状态 <span class="arithmatex">\(\rightarrow\)</span> 容错极简</p></li>
<li><p>gen(step <span class="arithmatex">\(N+1\)</span>) 与 train(step <span class="arithmatex">\(N\)</span>) 可重叠 <span class="arithmatex">\(\rightarrow\)</span> 30–40% 加速</p></li>
<li><p>不同精度：生成用 INT8（带宽优先），训练用 BF16（精度优先）</p></li>
</ul>
</div>

## 11.5　权重同步策略 {#weight-synchronization-strategies}

<span id="table-11-inline-6"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>滞后</strong></th><th scope="col"><strong>带宽</strong></th><th scope="col"><strong>质量影响</strong></th></tr></thead>
<tbody><tr><td>同步（每步）</td><td>0 步</td><td>140 GB/步</td><td>完美但太慢</td></tr>
<tr><td>周期性（每 50 步）</td><td>平均 25 步</td><td>摊销后 2.8 GB/步</td><td>质量损失 <span class="arithmatex">\(&lt;\)</span>2%</td></tr>
<tr><td>Delta 压缩（INT8）</td><td>平均 25 步</td><td>0.4 GB/步</td><td>质量损失 <span class="arithmatex">\(&lt;\)</span>3%</td></tr>
<tr><td>异步流式</td><td>5–10 步</td><td>14 GB/步（后台）</td><td>质量损失 <span class="arithmatex">\(&lt;\)</span>1%</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 PPO/GRPO 能容忍滞后</p>
<p>PPO 的裁剪目标本来就是<em>为</em>离策略数据设计的！裁剪区间 <span class="arithmatex">\([1-\epsilon, 1+\epsilon]\)</span> 限制了陈旧数据的影响。当滞后在 10–50 步时：</p>

<ul>
<li><p>每步 policy 大约变化 <span class="arithmatex">\(\sim\)</span>0.1–1%（合理学习率下）</p></li>
<li><p>50 步累计：<span class="arithmatex">\(\sim\)</span>5% 的 policy 漂移</p></li>
<li><p>PPO 裁剪本身就能处理高达 20% 的漂移</p></li>
<li><p>经验上：50 步滞后的质量损失 <span class="arithmatex">\(&lt;\)</span>2%</p></li>
</ul>

<p><strong>带宽算账</strong>：70B BF16 = 140GB。InfiniBand 400Gb/s = 50GB/s <span class="arithmatex">\(\rightarrow\)</span> 完整同步 2.8s。配合 delta 压缩：<span class="arithmatex">\(&lt;\)</span>0.5s。异步同步 = 免费（在后台跑）。</p>
</div>

## 11.6　显存优化技术 {#memory-optimization-techniques}

<span id="table-11-inline-7"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>ZeRO 阶段</strong></th><th scope="col"><strong>切分对象</strong></th><th scope="col"><strong>每 GPU 显存（70B，8 GPU）</strong></th></tr></thead>
<tbody><tr><td>无（数据并行）</td><td>无（完整副本）</td><td>每 GPU 560GB（不可行）</td></tr>
<tr><td>ZeRO-1</td><td>仅优化器状态</td><td>175GB</td></tr>
<tr><td>ZeRO-2</td><td>优化器状态 + 梯度</td><td>105GB</td></tr>
<tr><td>ZeRO-3（FSDP）</td><td>优化器 + 梯度 + 参数</td><td><strong>70GB（A100-80GB 装得下！）</strong></td></tr></tbody>
</table>
</div>

<p><strong>其他技术</strong>：</p>

<ul>
<li><p><strong>梯度检查点（Gradient checkpointing）</strong> <a class="book-citation" id="cite-chen2016training-1" href="../../references.md#ref-chen2016training" aria-label="参考文献 49">[49]</a>：不存全部激活，反向时重算。节省约 60% 激活显存，代价是约 33% 的额外计算。可选择性：只对 attention 层（显存重）做 checkpoint，保留 FFN 激活（重算成本高）。</p></li>
<li><p><strong>混合精度（Mixed precision）</strong> <a class="book-citation" id="cite-micikevicius2018mixed-1" href="../../references.md#ref-micikevicius2018mixed" aria-label="参考文献 255">[255]</a>：前向用 BF16（2 字节/参数），优化器状态用 FP32（m、v 各 4 字节）。master 权重用 FP32 累积。</p></li>
<li><p><strong>CPU offloading</strong>（ZeRO-Infinity <a class="book-citation" id="cite-rajbhandari2021zeroinfinity-1" href="../../references.md#ref-rajbhandari2021zeroinfinity" aria-label="参考文献 305">[305]</a>）：把优化器状态放到 CPU RAM。显存省一半，但慢 2–3<span class="arithmatex">\(\times\)</span>（PCIe 64GB/s 瓶颈）。</p></li>
<li><p><strong>激活 offloading</strong>：前向时把激活搬到 CPU，反向时再搬回来。仅在显存确实紧张时使用。</p></li>
<li><p><strong>FlashAttention</strong> <a class="book-citation" id="cite-dao2022flashattention-1" href="../../references.md#ref-dao2022flashattention" aria-label="参考文献 65">[65]</a><a class="book-citation" id="cite-dao2023flashattention2-1" href="../../references.md#ref-dao2023flashattention2" aria-label="参考文献 64">[64]</a>：attention 显存从 O(<span class="arithmatex">\(n^2\)</span>) 降到 O(<span class="arithmatex">\(n\)</span>)。快 2–4<span class="arithmatex">\(\times\)</span>，长序列时显存节省巨大。</p></li>
</ul>

### 11.6.1　FlashAttention 对 RLHF 的影响 {#flash-attentions-impact-on-rlhf}

<div class="admonition tip"><p class="admonition-title">为什么 FlashAttention 对 RLHF 很重要</p>
<p>RLHF 涉及生成长序列（rollout）并对其训练。没有 FlashAttention 时：</p>

<ul>
<li><p>一条 4K token、32 个 head 的序列，仅 attention 矩阵就要 <span class="arithmatex">\(\sim\)</span>4 GB</p></li>
<li><p>这严重限制了 PPO/GRPO 训练时的 batch 大小</p></li>
<li><p>对 attention 激活做梯度检查点代价昂贵</p></li>
</ul>

<p>有了 FlashAttention：</p>

<ul>
<li><p>attention 显存为 <span class="arithmatex">\(O(n)\)</span>——由 <span class="arithmatex">\(Q, K, V, O\)</span> 张量主导</p></li>
<li><p>同等 GPU 显存下可承载更长的 rollout（8K–32K token）</p></li>
<li><p>反向从 <span class="arithmatex">\(Q, K, V\)</span> 重算 attention 分块（不存 <span class="arithmatex">\(n^2\)</span> 矩阵）</p></li>
<li><p>这是长上下文 RLHF（如推理模型）得以成立的关键</p></li>
</ul>
</div>

<div class="admonition warning"><p class="admonition-title">FlashAttention 与梯度检查点</p>
<p>FlashAttention 的反向在原地从 <span class="arithmatex">\(Q, K, V\)</span>（已存）重算 attention 分块。这意味着 FlashAttention 已经<em>隐式实现</em>了 <span class="arithmatex">\(O(n^2)\)</span> attention 矩阵的激活重算。无需再单独对 attention 层做 checkpoint——这样做反而会多余地重算 <span class="arithmatex">\(Q, K, V\)</span>。</p>
</div>

<div class="highlight"><pre><span></span><code><span class="c1"># 70B RLHF 训练的 DeepSpeed ZeRO-3 配置</span>
<span class="n">ds_config</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"bf16"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"enabled"</span><span class="p">:</span> <span class="kc">True</span><span class="p">},</span>
    <span class="s2">"zero_optimization"</span><span class="p">:</span> <span class="p">{</span>
        <span class="s2">"stage"</span><span class="p">:</span> <span class="mi">3</span><span class="p">,</span>
        <span class="s2">"overlap_comm"</span><span class="p">:</span> <span class="kc">True</span><span class="p">,</span>                    <span class="c1"># 通信与计算重叠</span>
        <span class="s2">"contiguous_gradients"</span><span class="p">:</span> <span class="kc">True</span><span class="p">,</span>            <span class="c1"># 更好的显存布局</span>
        <span class="s2">"reduce_scatter"</span><span class="p">:</span> <span class="kc">True</span><span class="p">,</span>                  <span class="c1"># 比 allreduce 更高效</span>
        <span class="s2">"reduce_bucket_size"</span><span class="p">:</span> <span class="mf">5e7</span><span class="p">,</span>               <span class="c1"># 每桶 5000 万参数</span>
        <span class="s2">"prefetch_bucket_size"</span><span class="p">:</span> <span class="mf">5e7</span><span class="p">,</span>             <span class="c1"># 预取下一个桶</span>
        <span class="s2">"param_persistence_threshold"</span><span class="p">:</span> <span class="mf">1e5</span><span class="p">,</span>      <span class="c1"># 小参数在所有 GPU 上常驻</span>
        <span class="s2">"offload_optimizer"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"device"</span><span class="p">:</span> <span class="s2">"cpu"</span><span class="p">,</span> <span class="s2">"pin_memory"</span><span class="p">:</span> <span class="kc">True</span><span class="p">},</span>  <span class="c1"># CPU offload</span>
        <span class="s2">"sub_group_size"</span><span class="p">:</span> <span class="mf">1e9</span><span class="p">,</span>                   <span class="c1"># 减少碎片</span>
    <span class="p">},</span>
    <span class="s2">"gradient_accumulation_steps"</span><span class="p">:</span> <span class="mi">4</span><span class="p">,</span>
    <span class="s2">"gradient_clipping"</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">,</span>
    <span class="s2">"train_micro_batch_size_per_gpu"</span><span class="p">:</span> <span class="mi">2</span><span class="p">,</span>
    <span class="s2">"wall_clock_breakdown"</span><span class="p">:</span> <span class="kc">True</span><span class="p">,</span>
<span class="p">}</span>
</code></pre></div>

## 11.7　大规模容错 {#fault-tolerance-at-scale}

<div class="admonition warning"><p class="admonition-title">硬件故障现实</p>
<p><strong>单卡 MTBF</strong>：约 10,000 小时。<br><strong>512 卡集群 MTBF</strong>：<span class="arithmatex">\(10000/512 \approx 20\)</span> 小时。算上软件/网络故障，<strong>实际 4–8 小时</strong>。<br><strong>多日训练</strong>：会遇到 5–15 次故障。没有容错机制，一次故障就会让整次训练报废。</p>
</div>

<p><strong>生产级容错栈</strong>：</p>

<ol>
<li><p><strong>探测</strong>：NCCL 超时（60s）、GPU 心跳（10s）、NVML 健康监控、ECC 错误计数。</p></li>
<li><p><strong>Checkpoint</strong>：每 50–100 步异步保存。非阻塞（后台线程）。保存内容：模型权重、优化器状态（Adam m/v）、调度器状态、RNG 状态、KL 系数、replay buffer。保留最近 3 个 checkpoint。70B 用时约 30s（并行写入 NVMe）。</p></li>
<li><p><strong>恢复</strong>：(a) 生成集群无状态，直接重启并加载最新权重。(b) 训练集群：加载 checkpoint，剔除故障节点重建 NCCL 进程组，重新分配 FSDP 分片，从最近 checkpoint 续训。</p></li>
<li><p><strong>弹性训练</strong>：Torch Elastic / Kubernetes 自动扩缩。几分钟内替换故障节点。训练以 <span class="arithmatex">\(N-1\)</span> 块 GPU 暂时继续。</p></li>
<li><p><strong>预防</strong>：训练前做 GPU 健康预筛（跑 GEMM 压测）。热备机待命。双轨 InfiniBand 等冗余网络通路。</p></li>
</ol>

## 11.8　端到端延迟拆解 {#end-to-end-latency-breakdown}

<figure id="figure-11-11"><a href="../../assets/figures/fig_041_fig41.png" target="_blank" rel="noopener" aria-label="查看图 11.11 原图" title="点击查看原图"><img src="../../assets/figures/fig_041_fig41.png" alt="不重叠（单体式）。解耦后：生成与训练重叠，实际加速 1.4\(\times\)。" loading="lazy"></a><figcaption><span class="figure-number">图 11.11　</span>不重叠（单体式）。解耦后：生成与训练重叠，实际加速 1.4<span class="arithmatex">\(\times\)</span>。</figcaption></figure>

<span id="table-11-inline-8"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>阶段</strong></th><th scope="col"><strong>用时（70B）</strong></th><th scope="col"><strong>受限于</strong></th><th scope="col"><strong>优化手段</strong></th></tr></thead>
<tbody><tr><td>生成（128<span class="arithmatex">\(\times\)</span>512 tok）</td><td>30–45s</td><td>显存带宽</td><td>vLLM、推测解码、INT8</td></tr>
<tr><td>Reward 打分</td><td>5–8s</td><td>算力（批量前向）</td><td>INT8 RM、batch=128</td></tr>
<tr><td>Reference log-prob</td><td>4–6s</td><td>算力（批量前向）</td><td>INT8 ref，或 LoRA（免费）</td></tr>
<tr><td>PPO 更新（4 epoch）</td><td>8–12s</td><td>算力（反向）</td><td>FSDP、FlashAttention</td></tr>
<tr><td>权重同步</td><td>0–3s</td><td>网络（异步）</td><td>Delta 压缩、异步</td></tr>
<tr><td><strong>合计（单体式）</strong></td><td><strong>50–75s</strong></td><td></td><td></td></tr>
<tr><td><strong>合计（解耦+重叠）</strong></td><td><strong>35–50s</strong></td><td></td><td>生成与上一步训练重叠</td></tr></tbody>
</table>
</div>

## 11.9　监控与可观测性 {#monitoring-and-observability}

<div class="admonition note"><p class="admonition-title">RLHF 训练中要追踪的关键指标</p>
<p><strong>质量指标</strong>（每 10 步记录）：</p>

<ul>
<li><p>平均 reward（应当先上升后趋稳）</p></li>
<li><p>与 reference 的 KL 散度（应当保持在 3–10）</p></li>
<li><p>响应长度分布（警惕 length hacking）</p></li>
<li><p>Entropy（应缓慢下降，不能崩塌）</p></li>
</ul>

<p><strong>系统指标</strong>（每步记录）：</p>

<ul>
<li><p>GPU 利用率（目标：训练 <span class="arithmatex">\(&gt;\)</span>80%，生成 <span class="arithmatex">\(&gt;\)</span>60%）</p></li>
<li><p>每 GPU 的显存水位（在 OOM 发生前发现）</p></li>
<li><p>生成吞吐（tokens/秒，应稳定）</p></li>
<li><p>Gradient norm（尖峰 = 即将出现不稳定）</p></li>
<li><p>NCCL 通信时间（侦测网络劣化）</p></li>
</ul>
</div>

## 11.10　网络拓扑与通信模式 {#sec:network-topology}

<p>高效的分布式训练要求理解连接 GPU 的分层通信网络。现代集群采用两级架构：超高速的节点内链路，以及较慢但可扩展的节点间网络。</p>

### 11.10.1　节点内：NVLink 与 NVSwitch {#intra-node-nvlink-and-nvswitch}

<p class="table-caption" id="table-11-5"><span class="table-number">表 11.5　</span>NVLink 各代及其对 LLM 训练的影响</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>代际</strong></th><th scope="col"><strong>每链路带宽</strong></th><th scope="col"><strong>每 GPU 链路数</strong></th><th scope="col"><strong>总带宽</strong></th><th scope="col"><strong>平台</strong></th></tr></thead>
<tbody><tr><td>NVLink 3.0</td><td>50 GB/s</td><td>12</td><td>600 GB/s</td><td>A100（DGX A100）</td></tr>
<tr><td>NVLink 4.0</td><td>50 GB/s</td><td>18</td><td>900 GB/s</td><td>H100（DGX H100）</td></tr>
<tr><td>NVLink 5.0</td><td>100 GB/s</td><td>18</td><td>1800 GB/s</td><td>B200（DGX B200）</td></tr></tbody>
</table>
</div>

<p>在单个节点内（通常 8 块 GPU），<strong>NVSwitch</strong> 在任意 GPU 对之间提供全二分带宽（full-bisection bandwidth）。这意味着任意 GPU 可以同时以全 NVLink 速率与任意其他 GPU 通信——这对张量并行至关重要，因为每层都需要在 8 块 GPU 间做 AllReduce。</p>

<div class="admonition note"><p class="admonition-title">NVSwitch vs PCIe 拓扑</p>
<p><strong>有 NVSwitch</strong>（DGX/HGX）：8 块 GPU 两两全互联，速率 600–1800 GB/s。TP 的 AllReduce 每层 <span class="arithmatex">\(\sim\)</span>0.2ms。</p>

<p><strong>无 NVSwitch</strong>（仅 PCIe 的服务器）：GPU 经由 CPU 的 PCIe root complex 通信，带宽仅 32–64 GB/s。8 卡 TP 慢上 10–30<span class="arithmatex">\(\times\)</span>。<strong>在仅 PCIe 的系统上绝不要用 TP<span class="arithmatex">\(&gt;\)</span>2。</strong></p>
</div>

### 11.10.2　节点间：InfiniBand 与 RoCE {#inter-node-infiniband-and-roce}

<p>跨节点做 FSDP/ZeRO-3 的 AllGather 与 ReduceScatter 时，节点间网络成为主要瓶颈。</p>

<p class="table-caption" id="table-11-6"><span class="table-number">表 11.6　</span>LLM 训练集群的节点间网络选项</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>技术</strong></th><th scope="col"><strong>带宽</strong></th><th scope="col"><strong>延迟</strong></th><th scope="col"><strong>说明</strong></th></tr></thead>
<tbody><tr><td>InfiniBand NDR</td><td>400 Gb/s（50 GB/s）</td><td>1–2 <span class="arithmatex">\(\mu\)</span>s</td><td>黄金标准，RDMA，无损</td></tr>
<tr><td>InfiniBand NDR（双轨）</td><td>800 Gb/s（100 GB/s）</td><td>1–2 <span class="arithmatex">\(\mu\)</span>s</td><td>用于 H100 集群</td></tr>
<tr><td>RoCE v2</td><td>100–400 Gb/s</td><td>2–5 <span class="arithmatex">\(\mu\)</span>s</td><td>更便宜，需要调优 PFC/ECN</td></tr>
<tr><td>Ethernet（TCP）</td><td>100–400 Gb/s</td><td>10–50 <span class="arithmatex">\(\mu\)</span>s</td><td>不适合 <span class="arithmatex">\(&gt;\)</span>16 GPU 训练</td></tr></tbody>
</table>
</div>

### 11.10.3　通信原语及其代价 {#communication-primitives-and-their-costs}

<p>理解每种集合通信原语的使用场景有助于诊断瓶颈：</p>

<p class="table-caption" id="table-11-7"><span class="table-number">表 11.7　</span>分布式 LLM 训练中的 NCCL 集合通信操作</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>集合通信</strong></th><th scope="col"><strong>搬运数据量</strong></th><th scope="col"><strong>被谁使用</strong></th><th scope="col"><strong>使用时机</strong></th></tr></thead>
<tbody><tr><td>AllReduce</td><td><span class="arithmatex">\(2 \cdot \frac{N-1}{N} \cdot M\)</span></td><td>TP、DP</td><td>跨 GPU 求和梯度或激活</td></tr>
<tr><td>AllGather</td><td><span class="arithmatex">\(\frac{N-1}{N} \cdot M\)</span></td><td>FSDP 前向</td><td>矩阵乘前重建完整参数张量</td></tr>
<tr><td>ReduceScatter</td><td><span class="arithmatex">\(\frac{N-1}{N} \cdot M\)</span></td><td>FSDP 反向</td><td>反向后分发梯度分片</td></tr>
<tr><td>Broadcast</td><td><span class="arithmatex">\(M\)</span></td><td>PP</td><td>把激活发到下一个流水线 stage</td></tr>
<tr><td>Send/Recv</td><td><span class="arithmatex">\(M\)</span></td><td>PP</td><td>相邻 stage 间的点对点</td></tr></tbody>
</table>
</div>

<p>其中 <span class="arithmatex">\(M\)</span> 是消息大小（字节），<span class="arithmatex">\(N\)</span> 是参与者数量。</p>

<div class="admonition tip"><p class="admonition-title">通信-计算重叠</p>
<p>现代框架（FSDP、DeepSpeed）会积极地让通信与计算重叠：</p>

<p><strong>前向传播</strong>：当第 <span class="arithmatex">\(i\)</span> 层在计算时，AllGather 预取第 <span class="arithmatex">\(i+1\)</span> 层的参数。第 <span class="arithmatex">\(i\)</span> 层一结束，其参数立刻被丢弃（“free-after-forward”）。</p>

<p><strong>反向传播</strong>：当第 <span class="arithmatex">\(i\)</span> 层在算梯度时，ReduceScatter 发送第 <span class="arithmatex">\(i+1\)</span> 层的梯度。调优得当时，这种重叠可隐藏 70–90% 的通信延迟。</p>

<p><strong>调优旋钮</strong>：<code>prefetch_factor</code>（提前预取多少层）、<code>reduce_bucket_size</code>（梯度归约的粒度）、<code>backward_prefetch</code>（反向预取的 “pre” 还是 “post” 策略）。</p>
</div>

### 11.10.4　网络拓扑设计 {#network-topology-design}

<p>生产集群采用 <strong>fat-tree</strong> 或 <strong>rail-optimized</strong>（按轨优化）拓扑：</p>

<ul>
<li><p><strong>Fat-tree</strong>：每一级都具备全二分带宽。任一节点都能以满速与任一其他节点通信。代价昂贵（需要大量交换机），但灵活性最高。</p></li>
<li><p><strong>Rail-optimized</strong>：每个节点的 GPU <span class="arithmatex">\(i\)</span> 连接到同一台叶子交换机（“轨 <span class="arithmatex">\(i\)</span>”）。轨内 AllReduce 便宜，跨轨流量昂贵。Meta 的 RSC 和 Google 的 TPU pod 都采用此拓扑。</p></li>
<li><p><strong>3D torus / Dragonfly</strong>：用于 HPC 集群（Frontier、Aurora）。拓扑感知的作业放置至关重要。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">作业放置很重要</p>
<p>在 512 卡集群上，随机分配节点会因网络拥塞导致 2–3<span class="arithmatex">\(\times\)</span> 减速。<strong>始终请求连续的节点块。</strong>生产调度器（Slurm、Kubernetes）应当强制本地性：一个训练作业的所有节点应处于同一叶子交换机下，或彼此距离不超过一跳。</p>
</div>

## 11.11　训练吞吐与模型 FLOPs 利用率 {#sec:mfu}

### 11.11.1　衡量训练效率：MFU {#measuring-training-efficiency-mfu}

<p><strong>模型 FLOPs 利用率（Model FLOPs Utilization, MFU）</strong> <a class="book-citation" id="cite-chowdhery2022palm-1" href="../../references.md#ref-chowdhery2022palm" aria-label="参考文献 55">[55]</a> 是衡量训练效率的标准指标：</p>



<div class="arithmatex">\[\text{MFU} = \frac{\text{实际吞吐（tokens/秒）} \times \text{每 token 的 FLOPs}}{\text{硬件峰值 FLOPS}}
\tag{11.5}\]</div>

<p>对于参数量为 <span class="arithmatex">\(P\)</span>、序列长度为 <span class="arithmatex">\(s\)</span>、batch 大小为 <span class="arithmatex">\(b\)</span> 的 transformer：</p>



<div class="arithmatex">\[\text{每 token 的 FLOPs} \approx 6P + 12 \cdot n_\text{layers} \cdot d_\text{model} \cdot s
\tag{11.6}\]</div>

<p>系数 6 来自：2（乘加）<span class="arithmatex">\(\times\)</span> 3（前向 + 反向，反向约为前向的 <span class="arithmatex">\(2\times\)</span>）。第二项对应 attention 的 <span class="arithmatex">\(O(s^2)\)</span> 代价。</p>

<p class="table-caption" id="table-11-8"><span class="table-number">表 11.8　</span>各规模与硬件下的 MFU 基准</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模型</strong></th><th scope="col"><strong>硬件</strong></th><th scope="col"><strong>MFU</strong></th><th scope="col"><strong>Tokens/秒/GPU</strong></th><th scope="col"><strong>配置</strong></th></tr></thead>
<tbody><tr><td>LLaMA-7B</td><td>8<span class="arithmatex">\(\times\)</span>A100</td><td>57%</td><td>3,200</td><td>FSDP, FlashAttn, BF16</td></tr>
<tr><td>LLaMA-13B</td><td>16<span class="arithmatex">\(\times\)</span>A100</td><td>52%</td><td>1,750</td><td>FSDP, FlashAttn, BF16</td></tr>
<tr><td>LLaMA-70B</td><td>64<span class="arithmatex">\(\times\)</span>A100</td><td>45%</td><td>380</td><td>FSDP+TP=8, FlashAttn</td></tr>
<tr><td>GPT-4（估计）</td><td>10,000+ H100</td><td>40–50%</td><td>—</td><td>3D 并行</td></tr>
<tr><td>PaLM-540B</td><td>6144 TPUv4</td><td>46%</td><td>—</td><td>DP+TP+PP</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">为什么 MFU 会随规模下降</p>
<p>更大的模型需要更多并行，会引入：</p>

<ol>
<li><p><strong>通信开销</strong>：FSDP 的 AllGather/ReduceScatter（64 GPU 时 <span class="arithmatex">\(\sim\)</span>10–15%）</p></li>
<li><p><strong>流水线气泡</strong>：PP 在 micro-batch 的首尾引入空闲（PP=4 时 <span class="arithmatex">\(\sim\)</span>15–25%）</p></li>
<li><p><strong>辅助模型占用显存</strong>：reference/RM 占用了本可用于更大 batch 的显存</p></li>
<li><p><strong>负载不均</strong>：并非所有层计算量相等（embedding vs transformer block）</p></li>
</ol>

<p><strong>经验法则</strong>：训练目标 MFU <span class="arithmatex">\(&gt;\)</span> 40%。低于 30% 时应使用 profiling 诊断。</p>
</div>

### 11.11.2　算力最优的 batch 设计 {#compute-optimal-batch-sizing}

<p>有效 batch 大小与硬件利用率之间存在并非显而易见的相互作用：</p>



<div class="arithmatex">\[\text{有效 batch} = \text{micro\_batch} \times \text{grad\_accum} \times \text{DP 度}
\tag{11.7}\]</div>

<ul>
<li><p><strong>过小</strong>：GPU 利用率低（算术强度低），通信占主导。</p></li>
<li><p><strong>过大</strong>：每 token 的学习收益递减（超过 critical batch size），算力被浪费。</p></li>
<li><p><strong>最佳点</strong>：<em>临界 batch size</em>（critical batch size）<span class="arithmatex">\(B_\text{crit}\)</span>，即梯度噪声等于梯度信号之处。对 LLM，<span class="arithmatex">\(B_\text{crit} \sim 1\)</span>–<span class="arithmatex">\(4\)</span>M token <a class="book-citation" id="cite-mccandlish2018empirical-1" href="../../references.md#ref-mccandlish2018empirical" aria-label="参考文献 250">[250]</a>。</p></li>
</ul>

<p>对 RLHF 而言，batch 包含的是<em>rollout</em>（不仅仅是 token）：</p>



<div class="arithmatex">\[\text{RLHF batch} = N_\text{prompts} \times K_\text{generations} \times L_\text{平均响应长度}
\tag{11.8}\]</div>

<p>生产常见取值：<span class="arithmatex">\(N=128\)</span> 条 prompt，<span class="arithmatex">\(K=1\)</span>–<span class="arithmatex">\(4\)</span> 个生成，<span class="arithmatex">\(L=256\)</span>–<span class="arithmatex">\(512\)</span> token <span class="arithmatex">\(\rightarrow\)</span> 每步 32K–256K token。</p>

### 11.11.3　Profiling 与瓶颈诊断 {#profiling-and-bottleneck-diagnosis}

<p>关键的 profiling 工具及其揭示的内容：</p>

<span id="table-11-inline-13"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>工具</strong></th><th scope="col"><strong>捕获信息</strong></th><th scope="col"><strong>擅长</strong></th></tr></thead>
<tbody><tr><td><code>torch.profiler</code></td><td>Kernel 计时、显存</td><td>找慢算子、显存泄漏</td></tr>
<tr><td>NVIDIA Nsight Systems</td><td>完整的 GPU 时间线</td><td>可视化重叠、kernel 间空隙</td></tr>
<tr><td><code>nccl_debug=INFO</code></td><td>集合通信大小/耗时</td><td>诊断通信瓶颈</td></tr>
<tr><td><code>torch.cuda.memory_stats</code></td><td>分配模式</td><td>找碎片、峰值占用</td></tr>
<tr><td>DeepSpeed Flops Profiler</td><td>每层 FLOPs</td><td>识别负载不均</td></tr>
<tr><td><code>py-spy</code> / <code>scalene</code></td><td>CPU profiling</td><td>数据加载、分词瓶颈</td></tr></tbody>
</table>
</div>

<div class="admonition example"><p class="admonition-title">诊断 MFU 偏低：一份清单</p>
<ol>
<li><p><strong>GPU 利用率 <span class="arithmatex">\(&lt;\)</span> 80%？</strong> <span class="arithmatex">\(\rightarrow\)</span> 数据加载瓶颈（检查 CPU、I/O）。</p></li>
<li><p><strong>Kernel 之间有大空隙？</strong> <span class="arithmatex">\(\rightarrow\)</span> Python 开销、同步点。使用 CUDA graphs。</p></li>
<li><p><strong>通信占步时间 <span class="arithmatex">\(&gt;\)</span> 20%？</strong> <span class="arithmatex">\(\rightarrow\)</span> 降低 TP 度、加大 batch、检查网络健康。</p></li>
<li><p><strong>显存 99%？</strong> <span class="arithmatex">\(\rightarrow\)</span> batch 无法继续增大。尝试梯度检查点、offloading。</p></li>
<li><p><strong>生成时 OOM？</strong> <span class="arithmatex">\(\rightarrow\)</span> KV cache 太大。降低 max_seq_len 或生成 batch。</p></li>
</ol>
</div>

## 11.12　成本分析与云端部署 {#cost-analysis-and-cloud-deployment}

<p>理解 RLHF 训练的经济账，对规划至关重要。</p>

### 11.12.1　硬件成本对比 {#hardware-cost-comparison}

<p class="table-caption" id="table-11-9"><span class="table-number">表 11.9　</span>RLHF 训练大致云端 GPU 成本（2024–2025 价格）</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>GPU</strong></th><th scope="col"><strong>按需/小时</strong></th><th scope="col"><strong>Spot/小时</strong></th><th scope="col"><strong>显存</strong></th><th scope="col"><strong>适用场景</strong></th></tr></thead>
<tbody><tr><td>A100 80GB</td><td>$2.50–3.50</td><td>$1.00–1.50</td><td>80 GB HBM2e</td><td>经济型训练、生成集群</td></tr>
<tr><td>H100 80GB</td><td>$4.00–6.00</td><td>$2.00–3.00</td><td>80 GB HBM3</td><td>生产级训练</td></tr>
<tr><td>H200 141GB</td><td>$6.00–8.00</td><td>—</td><td>141 GB HBM3e</td><td>大上下文、少卡配置</td></tr>
<tr><td>MI300X 192GB</td><td>$3.50–5.00</td><td>$1.50–2.50</td><td>192 GB HBM3</td><td>性价比替代方案</td></tr></tbody>
</table>
</div>

### 11.12.2　RLHF 训练成本估算 {#rlhf-training-cost-estimation}



<div class="arithmatex">\[\text{成本} = \frac{N_\text{steps} \times T_\text{step}}{3600} \times N_\text{GPUs} \times C_\text{GPU/hr}
\tag{11.9}\]</div>

<div class="admonition example"><p class="admonition-title">成本示例：70B 模型 RLHF（10K 步）</p>
<span id="table-11-inline-15"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col">步数</th><th scope="col">10,000</th></tr></thead>
<tbody><tr><td>每步用时（解耦）</td><td>45 秒</td></tr>
<tr><td>总训练时间</td><td><span class="arithmatex">\(10000 \times 45 / 3600 = 125\)</span> 小时</td></tr>
<tr><td>GPU 数（生成 + 训练）</td><td>64 块 A100-80GB</td></tr>
<tr><td>每 GPU-小时成本（spot）</td><td>$1.20</td></tr>
<tr><td><strong>总成本</strong></td><td><span class="arithmatex">\(125 \times 64 \times \$1.20 =\)</span> <strong>$9,600</strong></td></tr></tbody>
</table>
</div>

<p><strong>按阶段拆分</strong>：</p>

<ul>
<li><p>生成集群（32 GPU）：$4,800（占 60% 时间）</p></li>
<li><p>训练集群（32 GPU）：$4,800（可重叠 <span class="arithmatex">\(\rightarrow\)</span> 实际 $3,400）</p></li>
<li><p>打分（与生成共享 GPU）：已含在上文中</p></li>
</ul>

<p><strong>重叠后</strong>：完整对齐 70B 模型的实际成本约为 <strong>$7,500</strong>。</p>
</div>

### 11.12.3　成本优化策略 {#cost-optimization-strategies}

<ul>
<li><p><strong>Spot/可抢占实例</strong>：节省 50–70%。要求 checkpoint 机制健壮（每 5 分钟保存一次）。</p></li>
<li><p><strong>合理配型</strong>：不要用 H100 做生成（显存带宽瓶颈）；推理时 A100 的 tokens/$ 相近。</p></li>
<li><p><strong>量化推理</strong>：生成与打分用 INT8/FP8 可让相应集群的 GPU 数减半。</p></li>
<li><p><strong>渐进式训练</strong>：先用 8B 代理模型做 reward 工程/调试（约 $200），再扩到 70B。</p></li>
<li><p><strong>用 LoRA 取消 reference</strong>：彻底移除 reference 模型（显存减少 50%）。</p></li>
<li><p><strong>先短后长</strong>：按 256<span class="arithmatex">\(\rightarrow\)</span>512<span class="arithmatex">\(\rightarrow\)</span>1024 token 的 curriculum 生成可节省 40% 算力。</p></li>
</ul>

## 11.13　分布式 Checkpointing {#sec:checkpointing}

<p>在大规模下，朴素 checkpointing 会成为瓶颈。一个带优化器状态的 70B 模型，每次 checkpoint 需要保存 <span class="arithmatex">\(\sim\)</span>840 GB（FP32 master 权重 + Adam m + v）。</p>

### 11.13.1　Checkpointing 策略 {#checkpointing-strategies}

<p class="table-caption" id="table-11-10"><span class="table-number">表 11.10　</span>大规模 RLHF 的 checkpointing 方案</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>策略</strong></th><th scope="col"><strong>保存耗时（70B）</strong></th><th scope="col"><strong>每 ckpt 存储</strong></th><th scope="col"><strong>特性</strong></th></tr></thead>
<tbody><tr><td>同步（所有 rank）</td><td>30–60s（阻塞）</td><td>420 GB</td><td>简单，会让训练停顿</td></tr>
<tr><td>异步（后台复制）</td><td><span class="arithmatex">\(&lt;\)</span>1s（非阻塞）</td><td>420 GB</td><td>与下一步重叠</td></tr>
<tr><td>增量（delta）</td><td><span class="arithmatex">\(&lt;\)</span>1s</td><td>5–20 GB</td><td>只保存变化的参数</td></tr>
<tr><td>分片（FSDP 原生）</td><td>5–10s</td><td>420 GB 分片</td><td>每个 rank 各存自己的分片</td></tr></tbody>
</table>
</div>

### 11.13.2　用 torch.distributed.checkpoint 做生产级 Checkpointing {#production-checkpointing-with-torch.distributed.checkpoint}

<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">torch.distributed.checkpoint</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">dcp</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">torch.distributed.checkpoint.state_dict</span><span class="w"> </span><span class="kn">import</span> <span class="n">get_state_dict</span><span class="p">,</span> <span class="n">StateDictOptions</span>

<span class="c1"># 保存：每个 rank 并行写自己的分片</span>
<span class="n">state_dict</span> <span class="o">=</span> <span class="p">{</span><span class="s2">"model"</span><span class="p">:</span> <span class="n">get_state_dict</span><span class="p">(</span><span class="n">model</span><span class="p">,</span> <span class="n">options</span><span class="o">=</span><span class="n">StateDictOptions</span><span class="p">(</span><span class="n">full_state_dict</span><span class="o">=</span><span class="kc">False</span><span class="p">))}</span>
<span class="n">dcp</span><span class="o">.</span><span class="n">save</span><span class="p">(</span>
    <span class="n">state_dict</span><span class="o">=</span><span class="n">state_dict</span><span class="p">,</span>
    <span class="n">storage_writer</span><span class="o">=</span><span class="n">dcp</span><span class="o">.</span><span class="n">FileSystemWriter</span><span class="p">(</span><span class="s2">"/mnt/checkpoints/step_5000"</span><span class="p">),</span>
    <span class="n">planner</span><span class="o">=</span><span class="n">dcp</span><span class="o">.</span><span class="n">DefaultSavePlanner</span><span class="p">(),</span>  <span class="c1"># 自动处理 FSDP 分片</span>
<span class="p">)</span>

<span class="c1"># 异步保存：非阻塞，运行于后台线程</span>
<span class="n">future</span> <span class="o">=</span> <span class="n">dcp</span><span class="o">.</span><span class="n">async_save</span><span class="p">(</span>
    <span class="n">state_dict</span><span class="o">=</span><span class="n">state_dict</span><span class="p">,</span>
    <span class="n">storage_writer</span><span class="o">=</span><span class="n">dcp</span><span class="o">.</span><span class="n">FileSystemWriter</span><span class="p">(</span><span class="s2">"/mnt/checkpoints/step_5000"</span><span class="p">),</span>
<span class="p">)</span>
<span class="c1"># 训练立即继续；只在需要时才用 future.result() 阻塞</span>
</code></pre></div>

<div class="admonition note"><p class="admonition-title">RLHF 的 Checkpoint 卫生</p>
<p>RLHF 的 checkpoint 需要捕获<em>比</em>标准预训练<em>更多</em>的内容：</p>

<ul>
<li><p>Policy 模型权重 + 优化器状态（标准）</p></li>
<li><p>KL 系数（<span class="arithmatex">\(\beta\)</span>）及其调度状态</p></li>
<li><p>Replay buffer 内容（用于离策略修正）</p></li>
<li><p>所有 GPU 的 RNG 状态（可复现性）</p></li>
<li><p>Prompt 迭代器位置（避免重复处理 prompt）</p></li>
<li><p>Reward 模型版本标签（用于审计）</p></li>
<li><p>Wandb/指标 run ID（用于连续日志）</p></li>
</ul>
</div>

## 11.14　硬件选型指南 {#sec:hardware-guide}

<p>合适的硬件选择取决于模型规模、预算和训练阶段。</p>

<p class="table-caption" id="table-11-11"><span class="table-number">表 11.11　</span>按模型规模与训练阶段给出的硬件建议</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模型规模</strong></th><th scope="col"><strong>训练阶段</strong></th><th scope="col"><strong>推荐</strong></th><th scope="col"><strong>配置</strong></th></tr></thead>
<tbody><tr><td><span class="arithmatex">\(\leq\)</span>7B</td><td>SFT + RLHF</td><td>1–2<span class="arithmatex">\(\times\)</span> A100</td><td>单节点，无需并行</td></tr>
<tr><td>7–13B</td><td>SFT + RLHF</td><td>4–8<span class="arithmatex">\(\times\)</span> A100</td><td>FSDP，可选生成 TP=2</td></tr>
<tr><td>13–34B</td><td>SFT + RLHF</td><td>8–16<span class="arithmatex">\(\times\)</span> A100/H100</td><td>FSDP + 生成 TP=4</td></tr>
<tr><td>70B</td><td>RLHF（完整）</td><td>32–64<span class="arithmatex">\(\times\)</span> A100/H100</td><td>解耦，FSDP + TP=8</td></tr>
<tr><td>70B</td><td>RLHF（LoRA）</td><td>8–16<span class="arithmatex">\(\times\)</span> A100/H100</td><td>无 reference，LoRA adapter</td></tr>
<tr><td><span class="arithmatex">\(&gt;\)</span>100B</td><td>RLHF</td><td>128+<span class="arithmatex">\(\times\)</span> H100</td><td>3D 并行（TP+PP+DP）</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">H100 vs A100：何时升级才值得？</p>
<p>H100 提供：</p>

<ul>
<li><p>峰值 FLOPS 约 <span class="arithmatex">\(\sim\)</span>1.6<span class="arithmatex">\(\times\)</span>（带稀疏 BF16：989 vs 624 TFLOPS；不带稀疏：495 vs 312）</p></li>
<li><p>显存带宽约 <span class="arithmatex">\(\sim\)</span>2<span class="arithmatex">\(\times\)</span>（3.35 vs 2.0 TB/s）</p></li>
<li><p>支持 FP8（推理再快 2<span class="arithmatex">\(\times\)</span>）</p></li>
<li><p>NVLink 4.0（900 vs 600 GB/s）</p></li>
</ul>

<p><strong>训练上</strong>：端到端约快 <span class="arithmatex">\(\sim\)</span>1.8–2.2<span class="arithmatex">\(\times\)</span>（FP8 支持与更高带宽放大了原始 FLOPS 优势）。</p>

<p><strong>生成上</strong>：约快 <span class="arithmatex">\(\sim\)</span>1.7<span class="arithmatex">\(\times\)</span>（带宽受限，2<span class="arithmatex">\(\times\)</span> BW 加上开销 <span class="arithmatex">\(\approx\)</span> 1.7<span class="arithmatex">\(\times\)</span> 吞吐）。</p>

<p><strong>性价比</strong>：在 1.5<span class="arithmatex">\(\times\)</span> 价格下，H100 在训练上几乎总是更划算。仅推理（生成集群）时，A100 在 spot 价格下可能更划算。</p>
</div>

## 11.15　RL 训练的优化器配置 {#sec:rl-optimizer-config}

<p>RL 训练（PPO、GRPO、DPO）相比预训练或 SFT 对优化器有独特要求。Loss 地形非平稳（policy 变化会改变生成的数据）、梯度更嘈杂（reward 信号方差大）、且更容易出现灾难性遗忘（Catastrophic Forgetting）或 reward hacking。本节以 AdamW <a class="book-citation" id="cite-loshchilov2019adamw-1" href="../../references.md#ref-loshchilov2019adamw" aria-label="参考文献 238">[238]</a> 作为默认优化器，整理 RL 特有的优化器实践指南。</p>

### 11.15.1　为什么 RL 需要不同的优化器设置 {#why-rl-requires-different-optimizer-settings}

<div class="admonition note"><p class="admonition-title">RL vs. SFT 优化——关键差异</p>
<ul>
<li><p><strong>数据分布非平稳</strong>：SFT 数据集固定，而 RL 每轮迭代都会生成新的 rollout——数据分布随 policy 一起漂移。</p></li>
<li><p><strong>高梯度方差</strong>：reward 信号稀疏且嘈杂；梯度方差远高于在精心策划数据上的交叉熵。</p></li>
<li><p><strong>需要更小的更新</strong>：policy 必须贴近 reference 模型（KL 约束），所以学习率比 SFT 小 10–100<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p><strong>不用 weight decay</strong>：正则化由 KL 惩罚提供，而不是 weight decay。再叠加 WD 可能与 KL 约束相互对抗。</p></li>
<li><p><strong>更短的 warmup</strong>：RL 从已收敛的 SFT checkpoint 起步——优化器状态只需极短的 warmup。</p></li>
</ul>
</div>

### 11.15.2　各 RL 方法的推荐超参 {#recommended-hyperparameters-by-rl-method}

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

### 11.15.3　RL 用 Beta-2 = 0.95：更快的自适应 {#beta-2-0.95-for-rl-faster-adaptation}

<p>Adam 默认的 <span class="arithmatex">\(\beta_2 = 0.999\)</span> 给二阶矩留下了非常长的记忆（有效窗口 <span class="arithmatex">\(\sim\)</span>1000 步）。在 RL 训练中，loss 地形随 policy 演化迅速变化——1000 步前的梯度方差已无关紧要。使用 <span class="arithmatex">\(\beta_2 = 0.95\)</span> 把窗口缩短到 <span class="arithmatex">\(\sim\)</span>20 步，让自适应学习率能够迅速响应变化中的梯度统计。</p>

<div class="admonition warning"><p class="admonition-title">beta2 = 0.95 反受其害的场景</p>
<p>对于非常小的 batch（例如在线 RL 中 batch=1），<span class="arithmatex">\(\beta_2 = 0.95\)</span> 会让二阶矩估计噪声过大。这种情况下，使用 <span class="arithmatex">\(\beta_2 = 0.99\)</span> 作为折中，或通过梯度累积加大有效 batch。</p>
</div>

### 11.15.4　RL 的混合精度：FP32 Master 权重至关重要 {#mixed-precision-for-rl-fp32-master-weights-are-critical}

<p>RL 训练对数值精度尤其敏感：</p>

<ul>
<li><p>梯度更嘈杂——小幅更新必须在多步中精确累积</p></li>
<li><p>学习率极小（<span class="arithmatex">\(10^{-6}\)</span>–<span class="arithmatex">\(10^{-7}\)</span>），使得 <span class="arithmatex">\(\Delta\theta \ll \theta\)</span></p></li>
<li><p>BF16 尾数（7 位 <span class="arithmatex">\(\approx\)</span> 0.8% 相对精度）无法表示相对于 <span class="arithmatex">\(10^{0}\)</span> 量级权重的 <span class="arithmatex">\(10^{-6}\)</span> 量级更新</p></li>
</ul>

<p><strong>RL 训练务必使用 FP32 master 权重。</strong>纯 BF16 训练（不留 FP32 副本）几乎必然会让 PPO/GRPO 在 100–500 步后出现 reward 崩塌。</p>

### 11.15.5　梯度裁剪对 RL 至关重要 {#gradient-clipping-is-critical-for-rl}

<p>在 PPO 和 GRPO 中，reward 信号变异性可能极高，尤其训练早期。一个坏 batch 就能产生 norm <span class="arithmatex">\(&gt;100\)</span> 的梯度，足以彻底破坏模型权重。<code>max_grad_norm=1.0</code> 是标准设置。SFT 下裁剪不那么关键，但仍建议使用。</p>

<div class="admonition warning"><p class="admonition-title">RL 训练绝不要关闭梯度裁剪</p>
<p>SFT 的梯度 norm 通常稳定在 0.1–1.0；而 RL 的梯度是尖刺型的，原因有三：(1) reward 方差经 policy gradient 传播；(2) 稀有的高 reward 轨迹会产生异常大的更新；(3) policy 漂移时 KL 惩罚项会产生大梯度。哪怕只有一步 <span class="arithmatex">\(\|\nabla\| &gt; 50\)</span> 的未裁剪更新，都可能让数百步训练前功尽弃。</p>
</div>

### 11.15.6　诊断 RL 训练的不稳定性 {#diagnosing-rl-training-instability}

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

### 11.15.7　HuggingFace TRL 的 RL 配置 {#huggingface-trl-configuration-for-rl}

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

### 11.15.8　MoE 在 RL 训练中的考虑 {#moe-considerations-for-rl-training}

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

##### Miles：PyTorch 原生的 RL 后训练。 {#heading-11-57}

<p>Miles <a class="book-citation" id="cite-miles2026pytorch-1" href="../../references.md#ref-miles2026pytorch" aria-label="参考文献 295">[295]</a> 是 PyTorch 官方为大规模 LLM RL 打造的技术栈，把 SGLang（用于高吞吐 rollout 生成）与 Megatron-LM（用于分布式训练更新）连接起来。它同时支持分离式执行（rollout 与训练使用各自的 GPU 池，最大化利用率）与同址执行（把两者放在同一批节点上，用于小规模实验）。Miles 代表了此前临时拼凑基础设施的标准化——正如十年前 PyTorch 本身标准化了深度学习训练一样。</p>
