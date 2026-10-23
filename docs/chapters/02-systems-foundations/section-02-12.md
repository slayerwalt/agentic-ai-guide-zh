# 2.2.12　vLLM 中的引导（受约束）解码 {#sec:vllm-guided-decoding}

<p>vLLM 通过可插拔的后端原生支持受约束解码（参见 Section <a href="../01-llm-architecture/section-12.md#sec:constrained-decoding">1.12.11</a>），可以在服务期间以极小的性能开销<em>保证</em>结构化输出。</p>

### 支持的约束类型。 {#supported-constraint-types.}

<p>OpenAI 兼容 API 通过 <code>guided_*</code> 参数或 <code>response_format</code> 字段接受约束：</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">openai</span><span class="w"> </span><span class="kn">import</span> <span class="n">OpenAI</span>
<span class="n">client</span> <span class="o">=</span> <span class="n">OpenAI</span><span class="p">(</span><span class="n">base_url</span><span class="o">=</span><span class="s2">"http://localhost:8000/v1"</span><span class="p">)</span>

<span class="c1"># --- JSON Schema 约束 ---</span>
<span class="n">response</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">chat</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B-Instruct"</span><span class="p">,</span>
    <span class="n">messages</span><span class="o">=</span><span class="p">[{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span>
               <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"Extract: name, age, city from: "</span>
                          <span class="s2">"'John is 30 and lives in NYC'"</span><span class="p">}],</span>
    <span class="n">extra_body</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"guided_json"</span><span class="p">:</span> <span class="p">{</span>
            <span class="s2">"type"</span><span class="p">:</span> <span class="s2">"object"</span><span class="p">,</span>
            <span class="s2">"properties"</span><span class="p">:</span> <span class="p">{</span>
                <span class="s2">"name"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">},</span>
                <span class="s2">"age"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"integer"</span><span class="p">},</span>
                <span class="s2">"city"</span><span class="p">:</span> <span class="p">{</span><span class="s2">"type"</span><span class="p">:</span> <span class="s2">"string"</span><span class="p">}</span>
            <span class="p">},</span>
            <span class="s2">"required"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"name"</span><span class="p">,</span> <span class="s2">"age"</span><span class="p">,</span> <span class="s2">"city"</span><span class="p">]</span>
        <span class="p">}</span>
    <span class="p">}</span>
<span class="p">)</span>
<span class="c1"># 输出保证是符合该 schema 的合法 JSON</span>

<span class="c1"># --- 正则表达式约束 ---</span>
<span class="n">response</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B-Instruct"</span><span class="p">,</span>
    <span class="n">prompt</span><span class="o">=</span><span class="s2">"Generate an IPv4 address: "</span><span class="p">,</span>
    <span class="n">extra_body</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"guided_regex"</span><span class="p">:</span> <span class="sa">r</span><span class="s2">"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"</span>
    <span class="p">}</span>
<span class="p">)</span>

<span class="c1"># --- 选项约束 ---</span>
<span class="n">response</span> <span class="o">=</span> <span class="n">client</span><span class="o">.</span><span class="n">completions</span><span class="o">.</span><span class="n">create</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B-Instruct"</span><span class="p">,</span>
    <span class="n">prompt</span><span class="o">=</span><span class="s2">"Sentiment: "</span><span class="p">,</span>
    <span class="n">extra_body</span><span class="o">=</span><span class="p">{</span><span class="s2">"guided_choice"</span><span class="p">:</span> <span class="p">[</span><span class="s2">"positive"</span><span class="p">,</span> <span class="s2">"negative"</span><span class="p">,</span> <span class="s2">"neutral"</span><span class="p">]}</span>
<span class="p">)</span>
</code></pre></div>

### 后端架构。 {#backend-architecture.}

<p>vLLM 将掩码计算委托给后端引擎：</p>

<ul>
<li><p><strong>XGrammar</strong>（自 v0.7 起为默认）：基于下推自动机的引擎，支持 JSON schema、正则和任意 EBNF 语法。得益于高效的 C++ 核心，对复杂 schema 而言最快。</p></li>
<li><p><strong>Outlines</strong> <a class="book-citation" id="cite-willard2023outlines-1" href="../../references.md#ref-willard2023outlines" aria-label="参考文献 378">[378]</a>：基于 FSM；支持 JSON 和正则。在 XGrammar 不可用时作为回退方案。</p></li>
</ul>

<p>掩码在模型前向传播产生 logits <em>之后</em>、采样<em>之前</em>施加——实际中每步只增加不到 1 ms，因为 FSM/PDA 状态转移和预先计算的索引查找均为 <span class="arithmatex">\(O(1)\)</span>。</p>

### 性能影响。 {#performance-impact.}

<p>由于约束仅对 logits 进行掩码（不重新计算 attention 或 FFN），吞吐损失可忽略（基准测试中 <span class="arithmatex">\(&lt;\)</span>2%）。主要成本来自把 schema <em>编译</em>成 FSM/PDA 索引，耗时 0.5–5 秒，取决于 schema 复杂度。vLLM 会跨请求缓存已编译的 schema，因此每个独特 schema 只需支付一次成本。</p>

<div class="admonition warning"><p class="admonition-title">结构化输出 <span class="arithmatex">\(\neq\)</span> 正确输出</p>
<p>受约束解码保证输出<em>语法上</em>有效（能解析为 JSON、符合 schema 的类型）。但它<em>不</em>保证<em>语义上</em>的正确性——模型仍可能产生能成功解析但事实错误的幻觉值。务必在下游验证业务逻辑。</p>
</div>

<p class="table-caption" id="table-2-4"><span class="table-number">表 2.4　</span>vLLM 与其他方案的性能对比（70B 模型，A100 <span class="arithmatex">\(\times\)</span> 4，TP=4）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>vLLM</strong></th><th scope="col"><strong>HF Generate</strong></th><th scope="col"><strong>原因</strong></th></tr></thead>
<tbody><tr><td>吞吐（tok/s）</td><td>2,500–4,000</td><td>300–600</td><td>连续批处理 + PagedAttention</td></tr>
<tr><td>显存利用率</td><td>90–95%</td><td>50–60%</td><td>零碎片，动态块分配</td></tr>
<tr><td>最大并发序列数</td><td>200–500</td><td>16–32</td><td>分页 KV 消除每序列预留</td></tr>
<tr><td>首 token 延迟</td><td>100–300ms</td><td>500–2000ms</td><td>对重复系统提示启用前缀缓存</td></tr></tbody>
</table>
</div>

### Dynamo：数据中心级推理编排。 {#heading-2-47}

<p>Dynamo <a class="book-citation" id="cite-dynamo2025nvidia-1" href="../../references.md#ref-dynamo2025nvidia" aria-label="参考文献 270">[270]</a> 是 NVIDIA 的开源编排层，位于各推理引擎（SGLang、TensorRT-LLM、vLLM）之<em>上</em>，把它们组织成协同的多节点服务系统。关键能力包括：分离式服务（把 prefill 与 decode 阶段拆到专用节点池）、基于 KV 缓存局部性的智能请求路由、多级 KV 缓存（GPU <span class="arithmatex">\(\to\)</span> CPU <span class="arithmatex">\(\to\)</span> SSD），以及自动扩缩容。它填补了单 GPU 推理优化（由 vLLM 覆盖）与数据中心级生产服务之间的空白。</p>
