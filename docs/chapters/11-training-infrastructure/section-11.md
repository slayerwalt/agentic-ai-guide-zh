# 11.11　训练吞吐与模型 FLOPs 利用率 {#sec:mfu}

## 11.11.1　衡量训练效率：MFU {#measuring-training-efficiency-mfu}

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

## 11.11.2　算力最优的 batch 设计 {#compute-optimal-batch-sizing}

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

## 11.11.3　Profiling 与瓶颈诊断 {#profiling-and-bottleneck-diagnosis}

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
