# 1.15　投机解码（Speculative Decoding）方法 {#speculative-decoding-methods}

<p>投机解码 <a class="book-citation" id="cite-leviathan2023fast-1" href="../../references.md#ref-leviathan2023fast" aria-label="参考文献 205">[205]</a> 通过同时预测多个 token，然后在目标模型的一次前向传递中对其进行验证，来加速自回归生成。它产生与标准解码<strong>完全相同的输出分布</strong>（无质量损失），同时实现 2–3<span class="arithmatex">\(\times\)</span> 加速。</p>

## 1.15.1　核心原理 {#core-principle}

<div class="admonition note"><p class="admonition-title">投机解码框架</p>
<ol>
<li><p>一个快速的<strong>草稿（draft）机制</strong>提议 <span class="arithmatex">\(k\)</span> 个候选 token：<span class="arithmatex">\(\hat{x}_1, \ldots, \hat{x}_k\)</span></p></li>
<li><p>大的<strong>目标模型</strong>对所有 <span class="arithmatex">\(k\)</span> 个 token 进行一次（批量）前向传递</p></li>
<li><p><strong>验证</strong>：从左到右依次接受 token，只要 <span class="arithmatex">\(P_{\text{target}}(\hat{x}_i) \geq r_i \cdot P_{\text{draft}}(\hat{x}_i)\)</span>（其中 <span class="arithmatex">\(r_i \sim U[0,1]\)</span>）</p></li>
<li><p>在位置 <span class="arithmatex">\(j\)</span> 首次被拒绝时：从调整后的分布重新采样 <span class="arithmatex">\(x_j\)</span>，并丢弃 <span class="arithmatex">\(\hat{x}_{j+1}, \ldots, \hat{x}_k\)</span></p></li>
</ol>

<p><strong>关键性质</strong>：这一接受/拒绝方案保证最终分布与 <span class="arithmatex">\(P_{\text{target}}\)</span> 完全一致。</p>

<p><strong>加速比</strong>：若接受率为 <span class="arithmatex">\(\alpha\)</span>，每步期望 token 数 <span class="arithmatex">\(= \frac{1 - \alpha^{k+1}}{1 - \alpha}\)</span>。在 <span class="arithmatex">\(\alpha=0.8\)</span>、<span class="arithmatex">\(k=5\)</span> 时：每步期望 3.4 个 token，相较标准解码的 1 个。</p>
</div>

## 1.15.2　方法对比 {#methods-comparison}

<p class="table-caption" id="table-1-19"><span class="table-number">表 1.19　</span>现代推理引擎支持的投机解码方法。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>草稿来源</strong></th><th scope="col"><strong>加速比</strong></th><th scope="col"><strong>核心思想</strong></th></tr></thead>
<tbody><tr><td><strong>标准方案</strong> <a class="book-citation" id="cite-leviathan2023fast-2" href="../../references.md#ref-leviathan2023fast" aria-label="参考文献 205">[205]</a></td><td>小模型（1–7B）</td><td>2–3<span class="arithmatex">\(\times\)</span></td><td>独立的草稿模型生成候选。简单但需要同时加载 2 个模型。</td></tr>
<tr><td><strong>Medusa</strong> <a class="book-citation" id="cite-cai2024medusa-1" href="../../references.md#ref-cai2024medusa" aria-label="参考文献 35">[35]</a></td><td>并行 LM 头</td><td>2–3<span class="arithmatex">\(\times\)</span></td><td>在目标模型上增加 <span class="arithmatex">\(k\)</span> 个额外的预测头。每个分别预测 <span class="arithmatex">\(+1, +2, \ldots, +k\)</span> 位置的 token。</td></tr>
<tr><td><strong>Eagle</strong> <a class="book-citation" id="cite-li2024eagle-1" href="../../references.md#ref-li2024eagle" aria-label="参考文献 212">[212]</a></td><td>特征层级</td><td>2.5–3.5<span class="arithmatex">\(\times\)</span></td><td>轻量解码器根据目标模型的隐藏状态生成草稿 token。接受率高于 Medusa。</td></tr>
<tr><td><strong>Eagle-2</strong> <a class="book-citation" id="cite-li2024eagle-2" href="../../references.md#ref-li2024eagle" aria-label="参考文献 212">[212]</a></td><td>上下文感知</td><td>3–4<span class="arithmatex">\(\times\)</span></td><td>基于置信度扩展的动态草稿树。当前最先进的接受率。</td></tr>
<tr><td><strong>N-gram Lookup</strong></td><td>N-gram 缓存</td><td>1.5–2<span class="arithmatex">\(\times\)</span></td><td>将 Prompt 的 n-gram 与已生成文本进行匹配。零成本；对重复输出极佳。</td></tr>
<tr><td><strong>Lookahead</strong> <a class="book-citation" id="cite-fu2024lookahead-1" href="../../references.md#ref-fu2024lookahead" aria-label="参考文献 101">[101]</a></td><td>Jacobi 迭代</td><td>2–2.5<span class="arithmatex">\(\times\)</span></td><td>并行 Jacobi 解码配合 n-gram 验证。无需草稿模型；使用目标模型自身。</td></tr>
<tr><td><strong>多 token 预测</strong> <a class="book-citation" id="cite-gloeckle2024multi-1" href="../../references.md#ref-gloeckle2024multi" aria-label="参考文献 111">[111]</a></td><td>修改架构</td><td>2–3<span class="arithmatex">\(\times\)</span></td><td>训练模型原生地在每步预测多个 token（Meta 在 Llama 中的方案）。</td></tr></tbody>
</table>
</div>

## 1.15.3　Medusa 多头推测解码 {#medusa-multi-head-speculative-decoding}

<div class="admonition note"><p class="admonition-title">Medusa 工作原理</p>
<p>Medusa 在 LLM 上额外增加 <span class="arithmatex">\(k\)</span> 个 “预测头”（共享同一主干）：</p>

<ul>
<li><p>Head 0（原始）：预测位置 <span class="arithmatex">\(t+1\)</span> 的 token（标准的下一个 token）</p></li>
<li><p>Head 1：预测位置 <span class="arithmatex">\(t+2\)</span> 的 token（跳过一个）</p></li>
<li><p>Head <span class="arithmatex">\(i\)</span>：预测位置 <span class="arithmatex">\(t+i+1\)</span> 的 token</p></li>
<li><p>所有头在一次前向传递中并行运行</p></li>
<li><p>一种<strong>树状结构</strong>的验证同时校验多条候选序列</p></li>
</ul>

<p><strong>训练</strong>：只微调 Medusa 头（主干冻结）。成本：在代表性数据上约 1 个 epoch。</p>

<p><strong>优势</strong>：无需独立的草稿模型；每个头都很小（一层线性层）。内存开销：<span class="arithmatex">\(&lt;\)</span>1%。</p>
</div>

## 1.15.4　Eagle：特征级草稿 {#eagle-feature-level-drafting}

<div class="admonition tip"><p class="admonition-title">为何 Eagle 优于 Medusa</p>
<p>Medusa 的各个头在不同位置独立预测——它们无法以自己先前的预测为条件（<span class="arithmatex">\(t+2\)</span> 位置的 token 不知道 <span class="arithmatex">\(t+1\)</span> 预测了什么）。Eagle 通过一个作用在目标模型隐藏状态上的轻量自回归解码器修正了这一点：</p>

<ol>
<li><p>从目标模型最后一层提取隐藏状态</p></li>
<li><p>输入一个小型（1 层）解码器，以先前隐藏状态为条件自回归地生成草稿 token</p></li>
<li><p>这捕捉到了 Medusa 所缺失的 token 间依赖</p></li>
</ol>

<p>结果：Eagle 实现 85–95% 的接受率，而 Medusa 为 60–80%。</p>
</div>

## 1.15.5　N-gram 投机解码 {#n-gram-speculative-decoding}

<div class="admonition note"><p class="admonition-title">N-gram 查找方法</p>
<p>最简单的投机解码——无需额外模型或训练：</p>

<ol>
<li><p>维护一份来自 Prompt 与已生成文本的 n-gram 缓存</p></li>
<li><p>每一步检查当前上下文的最后 <span class="arithmatex">\(n-1\)</span> 个 token 是否匹配缓存中的任一 n-gram</p></li>
<li><p>若匹配：将其后继作为草稿 token 提议</p></li>
<li><p>像往常一样在目标模型上进行验证</p></li>
</ol>

<p><strong>最佳场景</strong>：代码生成（重复模式）、结构化输出（JSON/XML），以及带有重复元素的 Prompt。<strong>成本</strong>：基本为零。</p>
</div>

## 1.15.6　与 vLLM 的集成 {#integration-with-vllm}

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">vllm</span><span class="w"> </span><span class="kn">import</span> <span class="n">LLM</span><span class="p">,</span> <span class="n">SamplingParams</span>

<span class="c1"># 标准投机解码（独立的草稿模型）</span>
<span class="n">llm</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B"</span><span class="p">,</span>
    <span class="n">tensor_parallel_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">speculative_config</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"model"</span><span class="p">:</span> <span class="s2">"meta-llama/Llama-3-8B"</span><span class="p">,</span>
        <span class="s2">"num_speculative_tokens"</span><span class="p">:</span> <span class="mi">5</span><span class="p">,</span>
    <span class="p">},</span>
<span class="p">)</span>

<span class="c1"># N-gram 投机（零成本，无需草稿模型）</span>
<span class="n">llm</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Llama-3-70B"</span><span class="p">,</span>
    <span class="n">speculative_config</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"ngram"</span><span class="p">,</span>
        <span class="s2">"num_speculative_tokens"</span><span class="p">:</span> <span class="mi">5</span><span class="p">,</span>
        <span class="s2">"prompt_lookup_max"</span><span class="p">:</span> <span class="mi">4</span><span class="p">,</span>  <span class="c1"># 从 Prompt 中匹配最长 4-gram</span>
    <span class="p">},</span>
<span class="p">)</span>

<span class="c1"># EAGLE 风格（特征级草稿，高接受率）</span>
<span class="n">llm</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Meta-Llama-3-8B-Instruct"</span><span class="p">,</span>
    <span class="n">tensor_parallel_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">speculative_config</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"model"</span><span class="p">:</span> <span class="s2">"yuhuili/EAGLE-LLaMA3-Instruct-8B"</span><span class="p">,</span>
        <span class="s2">"num_speculative_tokens"</span><span class="p">:</span> <span class="mi">2</span><span class="p">,</span>
        <span class="s2">"method"</span><span class="p">:</span> <span class="s2">"eagle"</span><span class="p">,</span>
        <span class="s2">"draft_tensor_parallel_size"</span><span class="p">:</span> <span class="mi">1</span><span class="p">,</span>
    <span class="p">},</span>
<span class="p">)</span>

<span class="c1"># MLP 推测器（IBM 风格，轻量预测头）</span>
<span class="n">llm</span> <span class="o">=</span> <span class="n">LLM</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"meta-llama/Meta-Llama-3.1-70B-Instruct"</span><span class="p">,</span>
    <span class="n">tensor_parallel_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">speculative_config</span><span class="o">=</span><span class="p">{</span>
        <span class="s2">"model"</span><span class="p">:</span> <span class="s2">"ibm-ai-platform/llama3-70b-accelerator"</span><span class="p">,</span>
        <span class="s2">"draft_tensor_parallel_size"</span><span class="p">:</span> <span class="mi">1</span><span class="p">,</span>
    <span class="p">},</span>
<span class="p">)</span>
</code></pre></div>

<div class="admonition warning"><p class="admonition-title">何时<em>不</em>该使用投机解码</p>
<ul>
<li><p><strong>高批量</strong>：当 batch <span class="arithmatex">\(\geq 64\)</span> 时，生成本身已经计算密集。投机解码引入的开销（草稿生成 + 验证）得不偿失。</p></li>
<li><p><strong>分布差异过大</strong>：若草稿模型与目标模型差异过大，接受率会跌至 50% 以下，反而比标准解码更慢。</p></li>
<li><p><strong>短输出</strong>：对少于 20 个 token 的输出，投机解码的启动开销超过收益。</p></li>
<li><p><strong>经验法则</strong>：投机解码对延迟敏感的单流生成（聊天机器人、交互式代码补全）帮助最大。</p></li>
</ul>
</div>
