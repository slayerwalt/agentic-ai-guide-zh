# 10.1　序列打包以提升效率 {#sec:sequence-packing}

<div class="admonition tip"><p class="admonition-title">Padding 问题</p>
<p>标准 SFT 的 Batch 会将所有序列填充（pad）到该 Batch 中最长序列的长度。对长度方差较大的数据集（例如短指令与长文档混合），这会浪费 50–80% 的计算量在 padding Token 上。序列打包（Sequence Packing）消除了这种浪费。</p>
</div>

<p>序列打包将多个短样本拼接为长度为 <code>max_seq_length</code> 的单一序列，以 EOS Token 分隔。Attention mask 确保不同样本的 Token 之间不会相互 attend：</p>

<ol>
<li><p>按长度对样本排序（可选，可提升打包效率）。</p></li>
<li><p>贪心地将样本打包进大小为 <code>max_seq_length</code> 的 bin。</p></li>
<li><p>使用块对角（block-diagonal）Attention mask，防止跨样本 attention。</p></li>
<li><p>只在非 padding Token 上计算 Loss。</p></li>
</ol>

<div class="admonition note"><p class="admonition-title">打包效率</p>
<ul>
<li><p>典型打包效率：85–95%(相比之下，padding 仅 20–50%)。</p></li>
<li><p>加速比：对长度方差较大的数据集可达 2–4<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p>显存：与 padding 相近（每个 Batch 的总 Token 数相同）。</p></li>
<li><p>注意事项：需谨慎设计 attention masking 以避免交叉污染。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">TRL 中的序列打包</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">SFTConfig</span><span class="p">,</span> <span class="n">SFTTrainer</span>

<span class="n">config</span> <span class="o">=</span> <span class="n">SFTConfig</span><span class="p">(</span>
    <span class="n">max_seq_length</span><span class="o">=</span><span class="mi">4096</span><span class="p">,</span>
    <span class="n">packing</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>           <span class="c1"># 启用序列打包</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"sft_model"</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">2e-5</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">SFTTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">args</span><span class="o">=</span><span class="n">config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">dataset</span><span class="p">,</span>
    <span class="c1"># dataset_text_field="text",  # 或使用 formatting_func</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>
</div>
