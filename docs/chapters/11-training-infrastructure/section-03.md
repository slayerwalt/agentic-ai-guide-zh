# 11.3　生成瓶颈：定量分析 {#the-generation-bottleneck-quantitative-analysis}

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
