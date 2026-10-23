# 2.2　vLLM——分页注意力与高吞吐推理 {#vllm-pagedattention-and-high-throughput-inference}

<p>vLLM <a class="book-citation" id="cite-kwon2023efficient-1" href="../../references.md#ref-kwon2023efficient" aria-label="参考文献 195">[195]</a> 引入了分页注意力（PagedAttention），它借鉴操作系统用于管理 RAM 的分页抽象，并将其应用于 GPU 的键值缓存（KV Cache）。在 LLM 推理过程中，<em>KV cache</em>——存储了所有先前 token 的 key 和 value 张量——是最大的内存消耗者。高效管理 KV cache 是高吞吐推理的核心挑战。</p>

## 本节内容

<ul class="chapter-outline">
<li><a href="section-02-01.md">2.2.1　KV cache 碎片化问题</a></li>
<li><a href="section-02-02.md">2.2.2　分页注意力——为 KV cache 引入虚拟内存</a></li>
<li><a href="section-02-03.md">2.2.3　PagedAttention 的收益</a></li>
<li><a href="section-02-04.md">2.2.4　连续批处理</a></li>
<li><a href="section-02-05.md">2.2.5　vLLM 中的投机解码</a></li>
<li><a href="section-02-06.md">2.2.6　具体显存节约——大规模 70B 模型</a></li>
<li><a href="section-02-07.md">2.2.7　vLLM：端到端系统</a></li>
<li><a href="section-02-08.md">2.2.8　架构总览</a></li>
<li><a href="section-02-09.md">2.2.9　核心组件</a></li>
<li><a href="section-02-10.md">2.2.10　请求生命周期（端到端流程）</a></li>
<li><a href="section-02-11.md">2.2.11　前缀缓存（自动 Prompt 缓存）</a></li>
<li><a href="section-02-12.md">2.2.12　vLLM 中的引导（受约束）解码</a></li>
</ul>
