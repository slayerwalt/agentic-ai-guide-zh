# 2.1.9　Tensor Cores {#tensor-cores}

<div class="admonition note"><p class="admonition-title">什么是 Tensor Cores？</p>
<p>张量核心（Tensor Core）是 Volta 架构（2017）引入的专用矩阵乘加（MMA）单元。每个 Tensor Core 在单个时钟周期内执行一次 <span class="arithmatex">\(4\times4\times4\)</span> 矩阵乘法：</p>



<div class="arithmatex">\[D = A \times B + C \quad (4\times4 \text{ matrices})\]</div>

<p>A100 在 108 个 SM 中共有 <strong>432 个 Tensor Cores</strong>（每个 SM 4 个，每个子分区 1 个）。在 BF16 精度下提供 312 TFLOP/s 吞吐——约是 FP32 CUDA cores 吞吐的 <span class="arithmatex">\(16\times\)</span>。</p>
</div>

<ul>
<li><p><strong>支持的精度：</strong>FP64、TF32、BF16、FP16、INT8、FP8（H100+）。</p></li>
<li><p><strong>累加：</strong>内部始终使用 FP32，即使输入为 BF16。这能防止点积过程中的灾难性抵消。</p></li>
<li><p><strong>要求：</strong>当矩阵维度是 8（BF16）或 16（FP8）的倍数时，Tensor Cores 效率最高。为此进行填充通常是值得的。</p></li>
<li><p><strong>WGMMA（H100）：</strong>Hopper 引入了 warpgroup 级别的 MMA 指令，作用于更大的分块（64<span class="arithmatex">\(\times\)</span>256<span class="arithmatex">\(\times\)</span>16），并可与 TMA（Tensor Memory Accelerator）数据搬运形成流水线。</p></li>
</ul>

<div class="admonition warning"><p class="admonition-title">Tensor Core 陷阱</p>
<p>只有当你的内核是<em>计算密集（Compute-Bound）</em>时，Tensor Cores 才有用。如果你跑的是小 batch（batch size 1 的推理），GEMM 分块很小，Tensor Core 利用率很低，你又回到内存密集区间。这就是为什么推理引擎要激进地对请求进行 batch。</p>
</div>
