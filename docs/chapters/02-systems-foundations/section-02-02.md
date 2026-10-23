# 2.2.2　分页注意力——为 KV cache 引入虚拟内存 {#pagedattention-virtual-memory-for-kv-caches}

<p>分页注意力（PagedAttention，Kwon 等，2023）借鉴了操作系统的<em>分页</em>抽象。它不再为每条序列分配一块连续内存，而是把 KV cache 切分成固定大小的<strong>页（page）</strong>（即块），并通过一个间接表——类似 CPU 页表——将每条序列的逻辑 token 位置翻译为分散的物理 GPU 内存地址。</p>

<div class="admonition note"><p class="admonition-title">PagedAttention 核心概念</p>
<ul>
<li><p><strong>块大小：</strong>通常每块 16 个 token（可调）。每个块存储 <span class="arithmatex">\(16 \times 2 \times L \times H \times d\)</span> 个元素。</p></li>
<li><p><strong>块表（Block table）：</strong>每条序列一份的映射，把逻辑块索引映射到 GPU 内存池中的物理块索引。</p></li>
<li><p><strong>物理块池：</strong>预先分配好的固定大小块池。分配为 <span class="arithmatex">\(O(1)\)</span>——只需从空闲列表弹出。</p></li>
<li><p><strong>注意力内核：</strong>做了修改，在注意力计算时使用块表从不连续的物理位置 gather KV 块。</p></li>
</ul>
</div>

<div class="admonition example"><p class="admonition-title">块表示例</p>
<p>假设块大小 = 4 个 token，我们有两条序列：</p>

<ul>
<li><p>序列 A（7 个 token）：逻辑块 [0,1] <span class="arithmatex">\(\to\)</span> 物理块 [3, 7]</p></li>
<li><p>序列 B（5 个 token）：逻辑块 [0,1] <span class="arithmatex">\(\to\)</span> 物理块 [1, 5]</p></li>
</ul>

<p>物理块 3 持有序列 A 的 token 0–3。物理块 7 持有序列 A 的 token 4–6（部分填充）。序列 A 的注意力内核按顺序从物理块 3 和 7 读取，并以块表作为间接层。</p>
</div>
