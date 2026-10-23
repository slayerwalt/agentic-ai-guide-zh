# 11.2　并行策略详解 {#sec:parallelism}

<p>训练大语言模型必须把计算分布到许多 GPU 上。可以沿<em>多个本质不同的维度</em>并行，每种都有各自的取舍。本节用数学公式、示意图和实践指引详细介绍每种策略。</p>

<figure id="figure-11-2"><a href="../../assets/figures/fig_032_fig32.png" target="_blank" rel="noopener" aria-label="查看图 11.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_032_fig32.png" alt="四种并行策略概览。生产系统通常会同时组合其中 2–3 种。" loading="lazy"></a><figcaption><span class="figure-number">图 11.2　</span>四种并行策略概览。生产系统通常会同时组合其中 2–3 种。</figcaption></figure>

## 11.2.1　数据并行（Data Parallelism, DP）与分布式数据并行（Distributed Data Parallelism, DDP） {#subsec:dp-ddp}

<p>数据并行是最简单、也最常见的分布式训练形式 <a class="book-citation" id="cite-li2020pytorch-1" href="../../references.md#ref-li2020pytorch" aria-label="参考文献 209">[209]</a>。每块 GPU 持有模型的<em>完整副本</em>，处理不同的 mini-batch，并同步梯度。</p>

#### 原始 DP（PyTorch <code>DataParallel</code>）。 {#vanilla-dp-pytorch-dataparallel.}

<p>单进程方案：一块 “主” GPU 负责分发输入、汇集输出、广播梯度。受 GIL 以及到主 GPU 的 PCIe 带宽限制。</p>

#### 分布式数据并行（DDP，<code>DistributedDataParallel</code>）。 {#distributed-data-parallelism-ddp-distributeddataparallel.}

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

## 11.2.2　张量并行（Tensor Parallelism, TP） {#subsec:tensor-parallel}

<p>张量并行（Megatron-LM 风格 <a class="book-citation" id="cite-shoeybi2019megatron-1" href="../../references.md#ref-shoeybi2019megatron" aria-label="参考文献 327">[327]</a>）把<em>单个权重矩阵</em>切到多块 GPU 上。每块 GPU 计算部分结果，再通过 AllReduce 合并。</p>

#### 列并行（Column-Parallel）线性层。 {#column-parallel-linear-layer.}

<p>权重矩阵 <span class="arithmatex">\(W \in \mathbb{R}^{d \times h}\)</span> 按列切到 <span class="arithmatex">\(T\)</span> 块 GPU 上：</p>



<div class="arithmatex">\[W = [W_0 \;|\; W_1 \;|\; \cdots \;|\; W_{T-1}], \quad W_i \in \mathbb{R}^{d \times h/T}
\tag{11.1}\]</div>

<p>每块 GPU <span class="arithmatex">\(i\)</span> 独立计算 <span class="arithmatex">\(Y_i = XW_i\)</span>（无需通信）。输出沿 hidden 维度被切开。</p>

#### 行并行（Row-Parallel）线性层。 {#row-parallel-linear-layer.}

<p>权重矩阵按行切：<span class="arithmatex">\(W = [W_0; W_1; \ldots; W_{T-1}]\)</span>，其中 <span class="arithmatex">\(W_i \in \mathbb{R}^{d/T \times h}\)</span>。输入 <span class="arithmatex">\(X\)</span> 也必须切开。每块 GPU 计算一个部分和，再通过 <strong>AllReduce</strong> 得到最终输出。</p>

<figure id="figure-11-4"><a href="../../assets/figures/fig_034_tp-column.png" target="_blank" rel="noopener" aria-label="查看图 11.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_034_tp-column.png" alt="列并行线性层（TP=2）。权重按列切开；每块 GPU 独立计算 \(XW_i\)。MLP 将其与行并行层配对，从而避免多余的 AllReduce。" loading="lazy"></a><figcaption><span class="figure-number">图 11.4　</span>列并行线性层（TP=2）。权重按列切开；每块 GPU 独立计算 <span class="arithmatex">\(XW_i\)</span>。MLP 将其与行并行层配对，从而避免多余的 AllReduce。</figcaption></figure>

#### TP 下的 Transformer Block。 {#transformer-block-with-tp.}

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

## 11.2.3　序列并行（Sequence Parallelism, SP） {#subsec:sequence-parallel}

<p>序列并行（Sequence Parallelism, SP） <a class="book-citation" id="cite-korthikanti2023reducing-1" href="../../references.md#ref-korthikanti2023reducing" aria-label="参考文献 191">[191]</a> 解决的是张量并行单独无法消除的一个显存瓶颈：LayerNorm 和 Dropout 层中的<strong>激活值显存</strong>。</p>

#### 问题。 {#the-problem.}

<p>TP 把权重显存切到了多块 GPU 上。但 LayerNorm 和 Dropout 作用在<em>完整</em>的 hidden 维上，并在每块 GPU 上都重复存在。它们的激活值（反向传播需要）占用与 <span class="arithmatex">\(b \times s \times d\)</span> 成正比的显存——每块 GPU 上一样多，TP 并不会把它降下来。</p>

#### 解法。 {#the-solution.}

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

## 11.2.4　流水线并行（Pipeline Parallelism, PP） {#subsec:pipeline-parallel}

<p>流水线并行把模型按层<em>纵向</em>切分，把连续的层组指派给不同的设备（stage）。激活值在 stage 间向前流动，梯度向后流动。</p>

#### 气泡（Bubble）问题。 {#the-bubble-problem.}

<p>朴素流水线执行会产生 “气泡”——某个 stage 等待前一 stage 的输入或后一 stage 的梯度时的空闲时间：</p>

<figure id="fig:pipeline-bubble"><a href="../../assets/figures/fig_037_pipeline-bubble.png" target="_blank" rel="noopener" aria-label="查看图 11.7 原图" title="点击查看原图"><img src="../../assets/figures/fig_037_pipeline-bubble.png" alt="流水线气泡对比。左：朴素流水线只有一个 micro-batch，75% 时间空闲。右：\(M=4\) 个 micro-batch 的 GPipe 大幅缩小气泡。当 \(M \gg P\) 时，气泡占比趋近于 0。" loading="lazy"></a><figcaption><span class="figure-number">图 11.7　</span>流水线气泡对比。左：朴素流水线只有一个 micro-batch，75% 时间空闲。右：<span class="arithmatex">\(M=4\)</span> 个 micro-batch 的 GPipe 大幅缩小气泡。当 <span class="arithmatex">\(M \gg P\)</span> 时，气泡占比趋近于 0。</figcaption></figure>

#### 气泡占比公式。 {#bubble-fraction-formula.}

<p>对于 <span class="arithmatex">\(P\)</span> 个流水线 stage、每步 <span class="arithmatex">\(M\)</span> 个 micro-batch：</p>



<div class="arithmatex">\[\text{气泡占比} = \frac{P - 1}{P + M - 1} \approx \frac{P-1}{M} \quad \text{（当 } M \gg P\text{ 时）}
\tag{11.3}\]</div>

<p>要让气泡开销 <span class="arithmatex">\(&lt;\)</span>10%，需要 <span class="arithmatex">\(M \geq 10 \cdot (P-1)\)</span>。PP=4 时：至少需要 30 个 micro-batch。</p>

#### 流水线调度。 {#pipeline-schedules.}

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

#### PP 中的通信。 {#communication-in-pp.}

<p>与 TP（AllReduce）不同，PP 只需要在相邻 stage 之间做激活值的<strong>点对点（point-to-point）</strong>通信：</p>



<div class="arithmatex">\[\text{每次传输的数据} = b_\text{micro} \times s \times d \times 2\text{ bytes (BF16)}
\tag{11.4}\]</div>

<p>若 micro-batch=4、seq=2048、<span class="arithmatex">\(d\)</span>=8192：每次传输 <span class="arithmatex">\(4 \times 2048 \times 8192 \times 2 = 128\)</span> MB。InfiniBand 50 GB/s 时：每次 2.6 ms——相对每个 stage 的计算量来说很小。</p>

#### 负载均衡。 {#load-balancing.}

<p>并非所有层的计算量都相等：</p>

<ul>
<li><p><strong>Embedding 层</strong>：非常便宜（查表）。</p></li>
<li><p><strong>Transformer block</strong>：计算量均匀。</p></li>
<li><p><strong>最末的 LM head</strong>：中等（词表投影是一次大型矩阵乘法）。</p></li>
</ul>

<p>把更多 Transformer 层分配给中间 stage，第一/最后 stage 少放一些，以平衡计算。</p>

## 11.2.5　完全分片数据并行（Fully Sharded Data Parallelism, FSDP / ZeRO-3） {#fully-sharded-data-parallelism-fsdp-zero-3}

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

## 11.2.6　3D 并行：组合多种策略 {#subsec:3d-parallel}

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

## 11.2.7　解耦 DiLoCo：跨数据中心训练 {#subsec:decoupled-diloco}

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
