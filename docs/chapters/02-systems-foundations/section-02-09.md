# 2.2.9　核心组件 {#core-components}

<ul>
<li><p><strong>API Server</strong>：接收 OpenAI 兼容的请求（completions、chat）。对输入进行 tokenize，并创建“序列组（sequence groups）”（用于 beam search 或多样本）。</p></li>
<li><p><strong>Scheduler</strong>：vLLM 的大脑。维护三个队列：</p>

<ul>
<li><p><code>waiting</code>：尚未开始的新请求（等待 prefill）</p></li>
<li><p><code>running</code>：正在生成 token（decode 阶段）</p></li>
<li><p><code>swapped</code>：被抢占、KV cache 已卸载到 CPU 的请求</p></li>
</ul>

<p>每次迭代，调度器根据可用 GPU 内存块决定运行哪些请求。</p></li>
<li><p><strong>Block Manager</strong>：为 KV cache 实现虚拟内存抽象。将逻辑块（按序列）映射到物理块（GPU 内存池中）。负责处理：</p>

<ul>
<li><p>分配（生成新 token <span class="arithmatex">\(\rightarrow\)</span> 需要新块）</p></li>
<li><p>写时复制（Copy-on-Write）（用于 beam search：多 beam 共享前缀块，仅在分叉时复制）</p></li>
<li><p>Swap（抢占/恢复时在 GPU <span class="arithmatex">\(\leftrightarrow\)</span> CPU 之间迁移）</p></li>
<li><p>前缀缓存（Prefix Caching）（当 prompt 共享相同前缀时复用已缓存块）</p></li>
</ul></li>
<li><p><strong>Model Executor</strong>：执行实际的 LLM 前向传播。管理跨 GPU 的张量并行，调度从分页 KV cache 块读取的注意力内核。</p></li>
<li><p><strong>KV Cache Pool</strong>：预分配的 GPU 内存，划分为固定大小的块（默认每块 16 token <span class="arithmatex">\(\times\)</span> num_heads <span class="arithmatex">\(\times\)</span> head_dim <span class="arithmatex">\(\times\)</span> 2 字节）。运行时无动态分配 <span class="arithmatex">\(\rightarrow\)</span> 零碎片。</p></li>
</ul>
