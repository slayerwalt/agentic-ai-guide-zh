# 11.13　分布式 Checkpointing {#sec:checkpointing}

<p>在大规模下，朴素 checkpointing 会成为瓶颈。一个带优化器状态的 70B 模型，每次 checkpoint 需要保存 <span class="arithmatex">\(\sim\)</span>840 GB（FP32 master 权重 + Adam m + v）。</p>

## 11.13.1　Checkpointing 策略 {#checkpointing-strategies}

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

## 11.13.2　用 torch.distributed.checkpoint 做生产级 Checkpointing {#production-checkpointing-with-torch.distributed.checkpoint}

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
