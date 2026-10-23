# 2.2.4　连续批处理 {#continuous-batching}

<p>传统的批处理（“静态批处理”）会等到一批中<em>所有</em>序列都完成后才开始新批次。如果一条序列生成 500 个 token、另一条生成 10 个，那么 GPU 在短序列上有 490 步是空闲的。这极其浪费。</p>

<div class="admonition note"><p class="admonition-title">连续批处理（Continuous Batching）</p>
<p>连续批处理（Continuous Batching，也叫迭代级调度）一次处理一个<em>解码步</em>。每一步之后：</p>

<ol>
<li><p>检查哪些序列已完成（生成了 EOS token）</p></li>
<li><p>把已完成的序列从批中移除，释放其 KV 块</p></li>
<li><p>加入新的等待序列以填补空出的槽位</p></li>
<li><p>在更新后的批上运行下一解码步</p></li>
</ol>

<p>批次组成每一步都在变化——序列动态地加入和离开。这让 GPU 利用率接近 100%，并大幅提升吞吐（相对静态批处理提升 1.5–3<span class="arithmatex">\(\times\)</span>）。PagedAttention 在这里至关重要：在批次中动态添加/移除序列需要动态分配/释放 KV 块，只有在分页内存下才高效。</p>
</div>
