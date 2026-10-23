# 2.2.3　PagedAttention 的收益 {#benefits-of-pagedattention}

### 近零浪费。 {#near-zero-waste.}

<p>内部碎片至多为每条序列一个部分填充块（最后一个块）。当块大小为 16 时，最差浪费为每条序列 15 个 token——可忽略不计。外部碎片被消除，因为块是固定大小、可互换的。</p>

### 动态分配。 {#dynamic-allocation.}

<p>块在序列增长时按需分配。无需预先知道最终序列长度。这对生成任务至关重要，因为输出长度未知。</p>

### 前缀共享（写时复制）。 {#prefix-sharing-copy-on-write.}

<p>多条共享同一前缀（例如同一系统提示）的序列可以<em>共享相同的物理块</em>来表示该前缀。块表只需让多条序列指向相同的物理块。当某条序列需要写入共享块（与前缀分叉）时，会触发写时复制（Copy-on-Write）。</p>

<div class="admonition tip"><p class="admonition-title">前缀共享带来的节约</p>
<p>在一个拥有 1000-token 系统提示、服务 128 位并发用户的聊天机器人中：</p>

<ul>
<li><p>无前缀共享：仅系统提示 KV cache 就需 <span class="arithmatex">\(128 \times 1000 \times 327{,}680 / 10^9 \approx 42\)</span> GB</p></li>
<li><p>有前缀共享：<span class="arithmatex">\(1 \times 1000 \times 327{,}680 / 10^9 \approx 0.33\)</span> GB</p></li>
<li><p>节约：共享前缀部分约 128<span class="arithmatex">\(\times\)</span></p></li>
</ul>
</div>

### 通过 swap 抢占。 {#preemption-via-swap.}

<p>当 GPU 显存耗尽时，vLLM 可以通过把某条序列的 KV 块换出到 CPU DRAM（或者直接丢弃，之后重新计算）来<em>抢占</em>该序列。这之所以可行，是因为块是自包含且非连续的——若是连续分配则需要复制整个缓冲区。</p>
