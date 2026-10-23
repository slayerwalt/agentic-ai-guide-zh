# 11.6　显存优化技术 {#memory-optimization-techniques}

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

## 11.6.1　FlashAttention 对 RLHF 的影响 {#flash-attentions-impact-on-rlhf}

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
