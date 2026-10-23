# 2.2.10　请求生命周期（端到端流程） {#request-lifecycle-end-to-end-flow}

<ol>
<li><p><strong>到达</strong>：客户端发送 prompt。API server 将其 tokenize，创建一个 <code>SequenceGroup</code>，放入 <code>waiting</code> 队列。</p></li>
<li><p><strong>调度</strong>：每一步，调度器执行：</p>

<ol>
<li><p>检查是否有 <code>swapped</code> 序列可以恢复（有足够空闲块）。</p></li>
<li><p>检查是否有 <code>waiting</code> 序列可以开始 prefill（有足够块容纳完整 prompt）。</p></li>
<li><p>在 <code>running</code> 序列之间预算剩余块（若当前块已满，每条序列每步需要 1 个新块）。</p></li>
<li><p>如果超出预算：<strong>抢占</strong>优先级最低的运行中序列（把 KV 换出到 CPU 或之后重新计算）。</p></li>
</ol></li>
<li><p><strong>Prefill</strong>（请求的第一次迭代）：整个 prompt 在一次前向传播中被处理。所有 prompt token 的 KV cache 被计算并存入已分配的块。这是计算密集（Compute-Bound）的（大批 token）。</p></li>
<li><p><strong>Decode</strong>（后续迭代）：每条序列每步生成一个新 token。所有运行中的序列被一起 batch（连续批处理）。这是内存密集（Memory-Bound）的（读取完整 KV cache，生成 1 个 token）。</p></li>
<li><p><strong>块分配</strong>：每个 decode 步后，如果某条序列的最后一个块已满，Block Manager 分配一个新的物理块并将其映射到下一逻辑块。</p></li>
<li><p><strong>完成</strong>：当一条序列遇到 EOS 或达到最大长度时，它会从 <code>running</code> 中移除。其物理块立即被释放 <span class="arithmatex">\(\rightarrow\)</span> 可供其他序列使用。响应被流式回传给客户端。</p></li>
</ol>
