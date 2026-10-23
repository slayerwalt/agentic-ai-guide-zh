# 第 1 章 · LLM 架构与优化方法 {#llm-architecture-and-optimization-methods}

<p>本部分覆盖大语言模型的基础架构，以及让训练和推理高效化的关键优化技术。内容按课程顺序组织：先介绍 Transformer 本身，然后讨论如何高效训练、如何低成本适配、如何压缩、如何扩展规模，以及如何加速推理。</p>

<ul class="chapter-outline" aria-label="本章阅读">
<li><a href="#how-llms-work-an-intuitive-overview">1.1–1.4 · 从输入到输出</a></li>
<li><a href="reading-02.md#optimization-theory-for-llm-training">1.5–1.8 · 训练理论与实践</a></li>
<li><a href="reading-03.md#lora-and-parameter-efficient-fine-tuning">1.9–1.11 · 高效模型与训练多样性</a></li>
<li><a href="reading-04.md#text-generation-decoding-methods">1.12–1.13 · 文本生成与 Prompt 工程</a></li>
<li><a href="reading-05.md#model-compression-methods">1.14–1.17 · 压缩、加速与安全</a></li>
</ul>

## 1.1　LLM 工作原理：直觉概览 {#how-llms-work-an-intuitive-overview}

<p>在深入架构细节之前，让我们先建立大语言模型如何将文本转换为文本的直觉。整个过程遵循一条简单的流水线：<strong>文本 <span class="arithmatex">\(\to\)</span> Token <span class="arithmatex">\(\to\)</span> 表征 <span class="arithmatex">\(\to\)</span> Token <span class="arithmatex">\(\to\)</span> 文本</strong>。</p>

<figure id="fig:llm-pipeline"><a href="../../assets/diagrams/figure-1-1.svg" target="_blank" rel="noopener" aria-label="查看图 1.1 原图" title="点击查看原图"><img src="../../assets/diagrams/figure-1-1.svg" alt="LLM 流水线：文本被分词为子词单元，转换为整数 ID，嵌入为稠密向量，经 Transformer 层处理，投影到词表 logits，最后解码回文本。虚线回路展示自回归生成——每个输出 Token 被追加到输入中作为下一次前向传播的输入。" loading="lazy"></a><figcaption><span class="figure-number">图 1.1　</span>LLM 流水线：文本被分词为子词单元，转换为整数 ID，嵌入为稠密向量，经 Transformer 层处理，投影到词表 logits，最后解码回文本。虚线回路展示自回归生成——每个输出 Token 被追加到输入中作为下一次前向传播的输入。</figcaption></figure>

<div class="admonition note"><p class="admonition-title">四个关键阶段</p>
<ol>
<li><p><strong>分词（Tokenization）</strong>：使用学习得到的词表将原始文本切分为子词片段（既非字符也非完整单词）。“unhappiness” 可能被切为 [“un”, “happiness”] 或 [“unhapp”, “iness”]。</p></li>
<li><p><strong>嵌入（Embedding）</strong>：每个 Token ID 索引到一张学习得到的嵌入表中，产生 <span class="arithmatex">\(\mathbb{R}^d\)</span>（通常 <span class="arithmatex">\(d = 4096\)</span>）中的稠密向量。这些向量捕捉语义含义——相似的词获得相似的向量。</p></li>
<li><p><strong>上下文处理</strong>：Transformer 栈并行处理所有嵌入，通过自注意力让每个位置“读取”所有其他位置。经过 <span class="arithmatex">\(L\)</span> 层之后，每个位置的隐藏状态编码了丰富的上下文信息。</p></li>
<li><p><strong>预测</strong>：最终的隐藏状态被投影为整个词表上的概率分布，由解码策略选择下一个 Token。</p></li>
</ol>
</div>

## 1.2　分词（Tokenization） {#sec:tokenization}

<p>分词（Tokenization）是将原始文本转换为语言模型所操作的离散符号的关键第一步。分词器的选择直接影响模型质量、多语种能力和计算效率。</p>

<div class="admonition tip"><p class="admonition-title">为什么用子词？</p>
<p>字符级模型需要很长的序列（注意力代价高昂）。词级模型无法处理稀有词或新词。子词分词达到了理想的平衡：常见词作为单个 Token（“the” <span class="arithmatex">\(\to\)</span> [the]），稀有词被分解为已知片段（“cryptocurrency” <span class="arithmatex">\(\to\)</span> [“crypt”, “ocur”, “rency”]），同时词表保持可控规模（32K–128K Token）。</p>
</div>

### 1.2.1　为什么不用字符或词？ {#why-not-characters-or-words}

<p class="table-caption" id="table-1-1"><span class="table-number">表 1.1　</span>不同分词粒度的权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>粒度</strong></th><th scope="col"><strong>词表大小</strong></th><th scope="col"><strong>序列长度</strong></th><th scope="col"><strong>问题</strong></th></tr></thead>
<tbody><tr><td>字符</td><td><span class="arithmatex">\(\sim\)</span>256</td><td>极长</td><td>注意力开销 <span class="arithmatex">\(O(n^2)\)</span>；难以学习长程语义</td></tr>
<tr><td>词</td><td><span class="arithmatex">\(\sim\)</span>500K+</td><td>短</td><td>无法处理稀有/新词；嵌入表巨大</td></tr>
<tr><td>子词</td><td>32K–128K</td><td>适中</td><td>最佳权衡：短序列、开放词表</td></tr></tbody>
</table>
</div>

### 1.2.2　字节对编码（Byte-Pair Encoding，BPE） {#byte-pair-encoding-bpe}

<p>字节对编码（Byte-Pair Encoding，BPE） <a class="book-citation" id="cite-sennrich2016bpe-1" href="../../references.md#ref-sennrich2016bpe" aria-label="参考文献 320">[320]</a> 是 GPT、Llama、Mistral 以及大多数现代 LLM 采用的主流分词算法。</p>

<div class="admonition note"><p class="admonition-title">BPE 算法</p>
<ol>
<li><p>从单个字符（字节）的词表开始</p></li>
<li><p>统计训练语料中所有相邻符号对</p></li>
<li><p>将出现频率最高的对合并为一个新符号</p></li>
<li><p>重复步骤 2–3 共 <span class="arithmatex">\(k\)</span> 次迭代（直到达到目标词表大小）</p></li>
</ol>
</div>

<figure id="figure-1-2"><a href="../../assets/figures/fig_003_fig3.png" target="_blank" rel="noopener" aria-label="查看图 1.2 原图" title="点击查看原图"><img src="../../assets/figures/fig_003_fig3.png" alt="BPE 分词示例：从字符开始，算法迭代地合并出现频率最高的相邻对，直到该词成为单个 Token 或词表预算耗尽。" loading="lazy"></a><figcaption><span class="figure-number">图 1.2　</span>BPE 分词示例：从字符开始，算法迭代地合并出现频率最高的相邻对，直到该词成为单个 Token 或词表预算耗尽。</figcaption></figure>

### 1.2.3　其他分词方法 {#other-tokenization-methods}

<p class="table-caption" id="table-1-2"><span class="table-number">表 1.2　</span>子词分词算法对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>使用者</strong></th><th scope="col"><strong>核心思想</strong></th></tr></thead>
<tbody><tr><td>BPE</td><td>GPT-4 <a class="book-citation" id="cite-openai2023gpt4-1" href="../../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a>, Llama-3 <a class="book-citation" id="cite-grattafiori2024llama3-1" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>, Mistral <a class="book-citation" id="cite-jiang2023mistral-1" href="../../references.md#ref-jiang2023mistral" aria-label="参考文献 165">[165]</a></td><td>自底向上合并频繁对；确定性</td></tr>
<tr><td>WordPiece</td><td>BERT <a class="book-citation" id="cite-devlin2019bert-1" href="../../references.md#ref-devlin2019bert" aria-label="参考文献 76">[76]</a>, DistilBERT <a class="book-citation" id="cite-sanh2019distilbert-1" href="../../references.md#ref-sanh2019distilbert" aria-label="参考文献 312">[312]</a></td><td>类似 BPE，但最大化训练数据似然</td></tr>
<tr><td>Unigram LM</td><td>SentencePiece（T5 <a class="book-citation" id="cite-raffel2020t5-1" href="../../references.md#ref-raffel2020t5" aria-label="参考文献 303">[303]</a>，XLNet <a class="book-citation" id="cite-yang2019xlnet-1" href="../../references.md#ref-yang2019xlnet" aria-label="参考文献 406">[406]</a>）</td><td>自顶向下：从大词表开始，按似然影响剪枝</td></tr>
<tr><td>字节级 BPE（Byte-level BPE）</td><td>GPT-2 <a class="book-citation" id="cite-radford2019gpt2-1" href="../../references.md#ref-radford2019gpt2" aria-label="参考文献 301">[301]</a>+</td><td>在原始字节上做 BPE（不可能出现未知 Token）；256 个基础词</td></tr></tbody>
</table>
</div>

### 1.2.4　分词最佳实践 {#tokenization-best-practices}

<ol>
<li><p><strong>词表大小很重要</strong>：32K 是最低限度；128K 能提供更好的多语种覆盖和代码处理能力。Llama-3 使用 128K Token。</p></li>
<li><p><strong>特殊 Token</strong>：始终包含 <code>&lt;bos&gt;</code>、<code>&lt;eos&gt;</code>、<code>&lt;pad&gt;</code>、<code>&lt;unk&gt;</code>。对于指令微调模型，需添加角色标记（<code>&lt;|user|&gt;</code>、<code>&lt;|assistant|&gt;</code>）。</p></li>
<li><p><strong>繁殖度（Fertility）</strong>：度量各语种的每词 Token 数。高繁殖度（每词产生很多 Token）表示对该语种的覆盖较差。</p></li>
<li><p><strong>切勿跨边界分词</strong>：空格、标点和数字应一致处理。大多数现代分词器会在前面添加空格标记（如 “the”）以区分词首 Token 与词中续接 Token。</p></li>
<li><p><strong>数字</strong>：对算术任务考虑数字级分词。“2024” 切为 [“2”,“0”,“2”,“4”] 可以支持逐位推理。</p></li>
<li><p><strong>代码</strong>：确保高效地分词空白（缩进）。Llama-3 将连续空格分词为单个 Token。</p></li>
</ol>

### 1.2.5　分词实战：HuggingFace 示例 {#tokenization-in-practice-huggingface-example}

<p><code>transformers</code> 库为所有分词器提供了统一接口。以下演示了使用现代 LLM 分词器进行编码和解码：</p>

<p class="code-caption"><span class="code-number">代码 1.1　</span>使用 HuggingFace Transformers 进行分词的编码/解码。</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoTokenizer</span>

<span class="c1"># 加载 Llama-3 分词器（128K 词表，字节级 BPE）</span>
<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="s2">"meta-llama/Meta-Llama-3-8B"</span><span class="p">)</span>

<span class="n">text</span> <span class="o">=</span> <span class="s2">"Reinforcement learning optimizes long-term rewards."</span>

<span class="c1"># 编码：文本 -&gt; Token ID</span>
<span class="n">token_ids</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">encode</span><span class="p">(</span><span class="n">text</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">token_ids</span><span class="p">)</span>
<span class="c1"># [128000, 29934, 262, 11008, 4815, 6900, 1317, 9860, 21845, 13]</span>

<span class="c1"># 将单个 Token 解码以查看子词切分</span>
<span class="n">tokens</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">convert_ids_to_tokens</span><span class="p">(</span><span class="n">token_ids</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">tokens</span><span class="p">)</span>
<span class="c1"># ['&lt;|begin_of_text|&gt;', 'Re', 'inforce', 'ment', ' learning',</span>
<span class="c1">#  ' optimizes', ' long', '-term', ' rewards', '.']</span>

<span class="c1"># 解码回文本（双向往返）</span>
<span class="n">reconstructed</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">token_ids</span><span class="p">,</span> <span class="n">skip_special_tokens</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="k">assert</span> <span class="n">reconstructed</span> <span class="o">==</span> <span class="n">text</span>  <span class="c1"># 完美重建</span>

<span class="c1"># 带注意力掩码的分词（用于带 padding 的批量输入）</span>
<span class="n">batch</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span>
    <span class="p">[</span><span class="s2">"Short text."</span><span class="p">,</span> <span class="s2">"A much longer input sentence for comparison."</span><span class="p">],</span>
    <span class="n">padding</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span>
<span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">batch</span><span class="o">.</span><span class="n">keys</span><span class="p">())</span>  <span class="c1"># dict_keys(['input_ids', 'attention_mask'])</span>
</code></pre></div>

### 1.2.6　特殊 Token 与结构化提示 {#special-tokens-and-structured-prompts}

<p>特殊 Token 是保留的词表条目，承载结构性含义而非语言内容。它们对控制模型行为至关重要。</p>

<p class="table-caption" id="table-1-3"><span class="table-number">表 1.3　</span>各类 LLM 中常见的特殊 Token。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>Token</strong></th><th scope="col"><strong>别名</strong></th><th scope="col"><strong>用途</strong></th></tr></thead>
<tbody><tr><td><code>&lt;bos&gt;</code> / <code>&lt;|begin_of_text|&gt;</code></td><td>BOS</td><td>标记序列开始</td></tr>
<tr><td><code>&lt;eos&gt;</code> / <code>&lt;|end_of_text|&gt;</code></td><td>EOS</td><td>标记序列结束；停止生成</td></tr>
<tr><td><code>&lt;|user|&gt;</code></td><td>—</td><td>标记对话中用户轮次的开始</td></tr>
<tr><td><code>&lt;|assistant|&gt;</code></td><td>—</td><td>标记对话中助手轮次的开始</td></tr>
<tr><td><code>&lt;pad&gt;</code></td><td>PAD</td><td>将批次填充到统一长度；在注意力中被掩盖</td></tr>
<tr><td><code>&lt;unk&gt;</code></td><td>UNK</td><td>词表外占位符（在 BPE 中很少出现）</td></tr>
<tr><td><code>[SEP]</code></td><td>SEP</td><td>分隔片段（BERT 风格）</td></tr>
<tr><td><code>[CLS]</code></td><td>CLS</td><td>分类 Token（BERT）</td></tr>
<tr><td><code>[MASK]</code></td><td>MASK</td><td>用于掩码语言模型（MLM）预训练的掩码 Token</td></tr></tbody>
</table>
</div>

##### 指令微调模型的角色标记。 {#role-markers-for-instruction-tuned-models.}

<p>现代对话模型使用特殊 Token 来界定对话结构。它们<strong>并非</strong>被训练来承载语义含义——而是结构分隔符，模型学习去解析它们：</p>

<p class="code-caption"><span class="code-number">代码 1.2　</span>带特殊 Token 的对话模板（Llama-3 格式）。</p>
<div class="highlight"><pre><span></span><code><span class="c1"># Llama-3 对话模板</span>
<span class="n">messages</span> <span class="o">=</span> <span class="p">[</span>
    <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"system"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"You are a helpful assistant."</span><span class="p">},</span>
    <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"Explain PPO in one sentence."</span><span class="p">},</span>
<span class="p">]</span>

<span class="c1"># apply_chat_template 处理所有特殊 Token 的插入</span>
<span class="n">prompt</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">apply_chat_template</span><span class="p">(</span><span class="n">messages</span><span class="p">,</span> <span class="n">tokenize</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">prompt</span><span class="p">)</span>
<span class="c1"># &lt;|begin_of_text|&gt;&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;</span>
<span class="c1">#</span>
<span class="c1"># You are a helpful assistant.&lt;|eot_id|&gt;&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;</span>
<span class="c1">#</span>
<span class="c1"># Explain PPO in one sentence.&lt;|eot_id|&gt;&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;</span>
<span class="c1">#</span>
<span class="c1">#</span>
</code></pre></div>

<div class="admonition note"><p class="admonition-title">特殊 Token 最佳实践</p>
<ul>
<li><p><strong>切勿拆分特殊 Token</strong>：它们必须是原子单位——确保分词器将其视为单个单元，而非字符序列。</p></li>
<li><p><strong>对特殊 Token 屏蔽损失</strong>：在 SFT 期间，不要对结构性 Token（角色标记、分隔符）计算损失。模型不应“学习”预测格式本身。</p></li>
<li><p><strong>用模板表达结构</strong>：通过特殊 Token 而非自然语言指令编码任务语义。例如，<code>&lt;|tool_call|&gt;</code> 比 “Now I will call a tool:” 更可靠。</p></li>
<li><p><strong>工具/函数调用</strong>：定义专用 Token，如 <code>&lt;|function|&gt;</code>、<code>&lt;|result|&gt;</code>，以在推理与动作之间建立无歧义的边界。</p></li>
<li><p><strong>RL 中的一致处理</strong>：在 PPO/GRPO 期间，确保参考模型与策略模型使用相同的分词和特殊 Token 处理——不一致会破坏 KL 计算。</p></li>
<li><p><strong>EOS 处理</strong>：在生成时，确保 EOS 包含在动作空间中。若模型无法发出 EOS，回复将无界增长（常见的 RL 失败模式）。</p></li>
</ul>
</div>

## 1.3　Transformer 架构 {#the-transformer-architecture}

<p>Transformer <a class="book-citation" id="cite-vaswani2017attention-1" href="../../references.md#ref-vaswani2017attention" aria-label="参考文献 357">[357]</a> 是所有现代 LLM 的基础。理解其组件对于把握本指南中的每种优化和训练方法都至关重要。</p>

### 1.3.1　整体结构 {#high-level-structure}

<p>仅解码器（decoder-only）的 Transformer 依次通过嵌入层、重复的注意力+FFN 块以及最终对词表 logits 的投影来处理 Token。图 <a href="#fig:decoder-only">1.3</a> 展示了完整架构。</p>

<figure id="fig:decoder-only"><a href="../../assets/figures/fig_004_decoder-only.png" target="_blank" rel="noopener" aria-label="查看图 1.3 原图" title="点击查看原图"><img src="../../assets/figures/fig_004_decoder-only.png" alt="仅解码器 Transformer 块（GPT 风格，Pre-Norm 变体）。每个子层（注意力、FFN）之前先经过 LayerNorm，之后接残差相加：\(\mathbf{x} + \text{SubLayer}(\text{LN}(\mathbf{x}))\)。这种 Pre-Norm 顺序（被 Llama、GPT-3、Mistral 采用）在无需 warmup 的情况下稳定训练，不同于原始的 Post-Norm（在相加之后应用 LayerNorm）。\(L\) 个相同的块被堆叠，之后是一个最终的 LayerNorm 和到词表 logits 的线性投影。" loading="lazy"></a><figcaption><span class="figure-number">图 1.3　</span>仅解码器 Transformer 块（GPT 风格，Pre-Norm 变体）。每个子层（注意力、FFN）之前先经过 LayerNorm，之后接残差相加：<span class="arithmatex">\(\mathbf{x} + \text{SubLayer}(\text{LN}(\mathbf{x}))\)</span>。这种 Pre-Norm 顺序（被 Llama、GPT-3、Mistral 采用）在无需 warmup 的情况下稳定训练，不同于原始的 Post-Norm（在相加之后应用 LayerNorm）。<span class="arithmatex">\(L\)</span> 个相同的块被堆叠，之后是一个最终的 LayerNorm 和到词表 logits 的线性投影。</figcaption></figure>

### 1.3.2　原始的编码器-解码器 Transformer {#the-original-encoder-decoder-transformer}

<p>Transformer 最初被提出 <a class="book-citation" id="cite-vaswani2017attention-2" href="../../references.md#ref-vaswani2017attention" aria-label="参考文献 357">[357]</a> 时是一种用于序列到序列任务（机器翻译、摘要）的<strong>编码器-解码器</strong>架构。尽管现代 LLM 主要使用仅解码器变体（GPT 风格），理解完整架构仍然至关重要，因为交叉注意力和带掩码的自注意力——两者都起源于此——仍是基础构建模块。</p>

<figure id="fig:transformer-original"><a href="../../assets/figures/fig_005_transformer-original.png" target="_blank" rel="noopener" aria-label="查看图 1.4 原图" title="点击查看原图"><img src="../../assets/figures/fig_005_transformer-original.png" alt="原始 Transformer 架构（Vaswani 等，2017）。编码器（左）通过双向自注意力处理整个输入。解码器（右）使用带掩码的自注意力以及对编码器表征的交叉注意力来自回归生成 Token。虚线框表示重复的层块（\(\times N\)）；灰线表示绕过每个子层的残差连接。注意：原始工作使用 Post-Norm（LayerNorm 在残差相加之后应用：\(\text{LN}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))\)），不同于现代 LLM 使用 Pre-Norm。" loading="lazy"></a><figcaption><span class="figure-number">图 1.4　</span>原始 Transformer 架构（Vaswani 等，2017）。编码器（左）通过双向自注意力处理整个输入。解码器（右）使用带掩码的自注意力以及对编码器表征的交叉注意力来自回归生成 Token。虚线框表示重复的层块（<span class="arithmatex">\(\times N\)</span>）；灰线表示绕过每个子层的残差连接。注意：原始工作使用 <strong>Post-Norm</strong>（LayerNorm 在残差相加<em>之后</em>应用：<span class="arithmatex">\(\text{LN}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))\)</span>），不同于现代 LLM 使用 Pre-Norm。</figcaption></figure>

##### 编码器（Encoder）。 {#encoder.}

<p>编码器<em>双向</em>地处理整个输入序列——每个 Token 关注所有其他 Token（无因果掩码）。这产生了丰富的上下文表征 <span class="arithmatex">\(\mathbf{H}^{\text{enc}} \in \mathbb{R}^{n \times d}\)</span>，其中每个位置都编码了关于整个输入的信息：</p>

<ul>
<li><p><strong>输入</strong>：Token 嵌入 + 正弦位置编码</p></li>
<li><p><strong>每一层</strong>：多头自注意力 <span class="arithmatex">\(\to\)</span> Add &amp; Norm <span class="arithmatex">\(\to\)</span> FFN <span class="arithmatex">\(\to\)</span> Add &amp; Norm</p></li>
<li><p><strong>无因果掩码</strong>：位置 <span class="arithmatex">\(i\)</span> 关注所有位置 <span class="arithmatex">\(1, \ldots, n\)</span></p></li>
<li><p><strong>输出</strong>：整个输入序列的上下文表征</p></li>
</ul>

##### 解码器——带掩码的多头自注意力。 {#decoder-masked-multi-head-self-attention.}

<p>解码器一次生成一个输出 Token（自回归地）。为了防止模型“看到未来”，解码器中的自注意力使用<strong>因果掩码（causal mask）</strong>：</p>



<div class="arithmatex">\[\text{MaskedAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
\tag{1.1}\]</div>

<p>其中掩码 <span class="arithmatex">\(M\)</span> 为：</p>



<div class="arithmatex">\[M_{ij} = \begin{cases} 0 &amp; \text{if } i \geq j \text{ (can attend)} \\ -\infty &amp; \text{if } i &lt; j \text{ (future token --- blocked)} \end{cases}\]</div>

<div class="admonition tip"><p class="admonition-title">为什么掩码很重要</p>
<p>在训练期间，解码器并行处理整个目标序列（teacher forcing），但每个位置只能关注之前的位置以保持自回归特性。掩码确保生成 Token <span class="arithmatex">\(t\)</span> 时仅使用 Token <span class="arithmatex">\(1, \ldots, t{-}1\)</span> 的信息。在推理时，Token 是逐一生成的，因此掩码是隐式的——但在训练期间，它使并行计算成为可能，同时保持因果性。</p>
</div>

##### 解码器——交叉注意力。 {#decoder-cross-attention.}

<p>在带掩码的自注意力之后，每个解码器层应用<strong>交叉注意力（cross-attention）</strong>，解码器关注编码器的输出表征。这是解码器“读取”输入的机制：</p>



<div class="arithmatex">\[\text{CrossAttn}(Q_{\text{dec}}, K_{\text{enc}}, V_{\text{enc}}) = \text{softmax}\!\left(\frac{Q_{\text{dec}} K_{\text{enc}}^T}{\sqrt{d_k}}\right) V_{\text{enc}}
\tag{1.2}\]</div>

<ul>
<li><p><strong>Query</strong> 来自解码器之前的子层（带掩码自注意力的输出）</p></li>
<li><p><strong>Key 和 Value</strong> 来自编码器的最终输出 <span class="arithmatex">\(\mathbf{H}^{\text{enc}}\)</span></p></li>
<li><p><strong>不应用掩码</strong>——每个解码器位置可以关注每个编码器位置</p></li>
<li><p>这使得解码器在每个生成步骤可以动态聚焦于输入的不同部分（例如，在英语<span class="arithmatex">\(\to\)</span>西班牙语翻译中，生成 “gato” 时关注 “cat”）</p></li>
</ul>

##### 完整的解码器层。 {#full-decoder-layer.}

<p>每个解码器层包含三个子层（编码器只有两个）：</p>

<ol>
<li><p><strong>带掩码的多头自注意力</strong> + 残差 + LayerNorm</p></li>
<li><p><strong>多头交叉注意力</strong>（对编码器输出）+ 残差 + LayerNorm</p></li>
<li><p><strong>前馈网络（Feed-Forward Network）</strong> + 残差 + LayerNorm</p></li>
</ol>

##### 从编码器-解码器到仅解码器。 {#from-encoder-decoder-to-decoder-only.}

<p>现代 LLM（GPT、Llama、Qwen）只使用解码器，完全移除了编码器和交叉注意力层。关键洞察：对于生成式语言建模，单个因果（掩码）自注意力栈就足够了——模型在单次传递中学会编码上下文并生成后续内容。这简化了架构、训练和推理，并且更有效地扩展规模。编码器-解码器模型（T5、BART）对于具有明确输入/输出结构的任务（翻译、摘要）仍然有意义，而交叉注意力在多模态模型中再次出现，其中视觉编码器为语言解码器提供 Key/Value。</p>

### 1.3.3　仅解码器 vs 编码器-解码器 {#decoder-only-vs-encoder-decoder}

<p>现代 LLM 几乎只使用仅解码器架构，但理解与编码器-解码器设计的权衡有助于阐明原因。</p>

<span id="table-1-inline-4"></span>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>架构</strong></th><th scope="col"><strong>示例</strong></th><th scope="col"><strong>使用场景</strong></th></tr></thead>
<tbody><tr><td>仅解码器</td><td>GPT-4 <a class="book-citation" id="cite-openai2023gpt4-1--section-03" href="../../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a>, Llama <a class="book-citation" id="cite-grattafiori2024llama3-1--section-03" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>, Mistral <a class="book-citation" id="cite-jiang2023mistral-1--section-03" href="../../references.md#ref-jiang2023mistral" aria-label="参考文献 165">[165]</a>, Qwen <a class="book-citation" id="cite-qwen2024qwen25-1" href="../../references.md#ref-qwen2024qwen25" aria-label="参考文献 350">[350]</a></td><td>自回归生成；在对话/推理领域占主导</td></tr>
<tr><td>编码器-解码器</td><td>T5 <a class="book-citation" id="cite-raffel2020t5-1--section-03" href="../../references.md#ref-raffel2020t5" aria-label="参考文献 303">[303]</a>, BART <a class="book-citation" id="cite-lewis2020bart-1" href="../../references.md#ref-lewis2020bart" aria-label="参考文献 207">[207]</a>, Flan-T5 <a class="book-citation" id="cite-chung2022flan-1" href="../../references.md#ref-chung2022flan" aria-label="参考文献 59">[59]</a></td><td>Seq2seq（翻译、摘要）；现在较少使用</td></tr>
<tr><td>仅编码器</td><td>BERT <a class="book-citation" id="cite-devlin2019bert-1--section-03" href="../../references.md#ref-devlin2019bert" aria-label="参考文献 76">[76]</a>, RoBERTa <a class="book-citation" id="cite-liu2019roberta-1" href="../../references.md#ref-liu2019roberta" aria-label="参考文献 233">[233]</a></td><td>分类/嵌入；不用于生成</td></tr></tbody>
</table>
</div>

<div class="admonition warning"><p class="admonition-title">为什么仅解码器胜出</p>
<p>仅解码器模型更简单（单一模型，单一损失），扩展性更好（所有参数都参与生成），并且支持统一的训练（预训练 = 下一个 Token 预测 = 微调目标）。对于纯生成任务，编码器-解码器模型在编码器上浪费了容量。</p>
</div>

### 1.3.4　嵌入：从离散 Token 到连续空间 {#embeddings-from-discrete-tokens-to-continuous-space}

<p>在任何注意力或计算发生之前，Transformer 必须将离散的 Token ID 转换为神经网络可处理的连续向量。这就是<strong>嵌入层（embedding layer）</strong>的作用。</p>

##### 什么是嵌入？ {#what-is-an-embedding}

<p>嵌入是一个离散符号的学习得到的稠密向量表示。我们不再把单词 “king” 表示为大小为 <span class="arithmatex">\(|\mathcal{V}| = 128{,}000\)</span> 的独热向量（大部分为零），而是表示为 <span class="arithmatex">\(\mathbb{R}^d\)</span>（例如 <span class="arithmatex">\(d = 4096\)</span>）中一个紧凑的向量，捕捉其<em>含义</em>。</p>

<p>关键洞察：<strong>相似的概念得到相近的向量</strong>。在一个训练良好的嵌入空间中：</p>

<ul>
<li><p>“king” 和 “queen” 接近（都是王室）</p></li>
<li><p>“king” 和 “bicycle” 距离很远（无关）</p></li>
<li><p>向量运算捕捉关系：<span class="arithmatex">\(\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}\)</span></p></li>
</ul>

<figure id="figure-1-5"><a href="../../assets/figures/fig_006_fig6.png" target="_blank" rel="noopener" aria-label="查看图 1.5 原图" title="点击查看原图"><img src="../../assets/figures/fig_006_fig6.png" alt="嵌入空间可视化（二维投影）：语义相似的词聚集在一起。嵌入表在预训练期间学习这些位置，纯粹从文本中的共现模式中捕捉含义。" loading="lazy"></a><figcaption><span class="figure-number">图 1.5　</span>嵌入空间可视化（二维投影）：语义相似的词聚集在一起。嵌入表在预训练期间学习这些位置，纯粹从文本中的共现模式中捕捉含义。</figcaption></figure>

##### 嵌入表。 {#the-embedding-table.}

<p>在实践中，嵌入层就是一个矩阵 <span class="arithmatex">\(\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d}\)</span>，其中第 <span class="arithmatex">\(i\)</span> 行存储 Token <span class="arithmatex">\(i\)</span> 的嵌入向量：</p>



<div class="arithmatex">\[\text{embed}(x_t) = \mathbf{E}[x_t] \in \mathbb{R}^d
\tag{1.3}\]</div>

<p>对于一个 Token ID 序列 <span class="arithmatex">\([x_1, x_2, \ldots, x_n]\)</span>，嵌入只是一个简单的表查询（索引操作）：</p>



<div class="arithmatex">\[\mathbf{H}_0 = [\mathbf{E}[x_1];\; \mathbf{E}[x_2];\; \ldots;\; \mathbf{E}[x_n]] \in \mathbb{R}^{n \times d}\]</div>

<div class="admonition note"><p class="admonition-title">Transformer 中的嵌入表</p>
<ul>
<li><p><strong>尺寸</strong>：<span class="arithmatex">\(|\mathcal{V}| \times d\)</span>。对于 Llama-3：<span class="arithmatex">\(128{,}256 \times 4{,}096 = 525\)</span>M 参数（占 8B 模型的 6.5%）。</p></li>
<li><p><strong>初始化</strong>：随机（Xavier/正态分布），然后通过反向传播学习。</p></li>
<li><p><strong>权重共享（Weight tying）</strong>：许多模型<em>共享</em>嵌入矩阵与输出投影头：<span class="arithmatex">\(W_{\text{head}} = \mathbf{E}^T\)</span>。这节省参数并创建对称的编码-解码结构。</p></li>
<li><p><strong>输入</strong>：Token ID（整数）<span class="arithmatex">\(\to\)</span> <strong>输出</strong>：<span class="arithmatex">\(\mathbb{R}^d\)</span> 中的稠密向量。</p></li>
<li><p><strong>梯度流</strong>：在训练期间，只有当前批次中 Token 对应的行接收梯度更新（稀疏更新）。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">为什么嵌入有效</p>
<p>嵌入表与模型其余部分端到端地一起学习。因为模型被训练去预测下一个 Token，它必须学到这样的表征：出现在相似上下文中的 Token 得到相似的向量。这就是分布假说：“你应当通过一个词的伙伴来了解它” <a class="book-citation" id="cite-firth1957synopsis-1" href="../../references.md#ref-firth1957synopsis" aria-label="参考文献 91">[91]</a>。嵌入层将这种统计结构压缩为稠密几何。</p>
</div>

##### 各向异性（anisotropy）问题。 {#the-anisotropy-problem.}

<p>当使用预训练嵌入（例如来自 BERT 或 GPT-2）用于下游任务（如检索 RAG 或推荐系统冷启动）时，会出现一个关键问题：学习到的表征高度<strong>各向异性（anisotropic）</strong>——它们占据嵌入空间中一个狭窄的锥形区域，而非均匀分布在所有方向上 <a class="book-citation" id="cite-ethayarajh2019contextual-1" href="../../references.md#ref-ethayarajh2019contextual" aria-label="参考文献 86">[86]</a>。</p>

<figure id="figure-1-6"><a href="../../assets/figures/fig_007_fig7.png" target="_blank" rel="noopener" aria-label="查看图 1.6 原图" title="点击查看原图"><img src="../../assets/figures/fig_007_fig7.png" alt="嵌入空间中的各向同性（isotropy）与各向异性（anisotropy）。左：各向同性的嵌入均匀分布，使余弦相似度成为可靠的语义相关性度量。右：各向异性的嵌入（如 BERT 中观察到的）聚集在狭窄的锥形内，导致所有对的余弦相似度都很高，无论语义内容如何。白化（whitening）变换空间以恢复各向同性。" loading="lazy"></a><figcaption><span class="figure-number">图 1.6　</span>嵌入空间中的各向同性（isotropy）与各向异性（anisotropy）。左：各向同性的嵌入均匀分布，使余弦相似度成为可靠的语义相关性度量。右：各向异性的嵌入（如 BERT 中观察到的）聚集在狭窄的锥形内，导致所有对的余弦相似度都很高，无论语义内容如何。白化（whitening）变换空间以恢复各向同性。</figcaption></figure>

<p><strong>这为何对应用很重要：</strong></p>

<ul>
<li><p><strong>RAG / 检索</strong>：如果所有嵌入无论内容如何余弦相似度都 <span class="arithmatex">\(&gt;0.7\)</span>，检索排名几乎成为随机——系统无法区分相关与不相关的段落。</p></li>
<li><p><strong>推荐系统</strong>：使用预训练 LLM 嵌入来表示物品/用户，只有在几何结构保留了有意义的相似性结构时才有效。</p></li>
<li><p><strong>聚类</strong>：各向异性的嵌入使聚类塌缩，无法发现自然分组。</p></li>
</ul>

##### 解决方案：白化（whitening）。 {#resolution-whitening.}

<p>一个简单有效的解决方法是<strong>白化（whitening）</strong> <a class="book-citation" id="cite-su2021whitening-1" href="../../references.md#ref-su2021whitening" aria-label="参考文献 340">[340]</a>——一种使嵌入分布变得各向同性（零均值、单位协方差）的线性变换：</p>



<div class="arithmatex">\[\tilde{\mathbf{h}} = \mathbf{D}^{-1/2} \mathbf{U}^T (\mathbf{h} - \boldsymbol{\mu})
\tag{1.4}\]</div>

<p>其中 <span class="arithmatex">\(\boldsymbol{\mu}\)</span> 是平均嵌入，<span class="arithmatex">\(\mathbf{U}\mathbf{D}\mathbf{U}^T\)</span> 是协方差矩阵 <span class="arithmatex">\(\Sigma = \frac{1}{N}\sum_i (\mathbf{h}_i - \boldsymbol{\mu})(\mathbf{h}_i - \boldsymbol{\mu})^T\)</span> 的特征分解。</p>

<div class="admonition note"><p class="admonition-title">白化的实践</p>
<ul>
<li><p><strong>作用</strong>：旋转并缩放嵌入空间，使所有方向具有相等的方差（单位协方差）。</p></li>
<li><p><strong>效果</strong>：余弦相似度变得有意义——语义相似的对得分高，不相似的对得分低。</p></li>
<li><p><strong>额外好处</strong>：可以同时通过仅保留前 <span class="arithmatex">\(k\)</span> 个特征向量来降维（类似 PCA），使检索更快。</p></li>
<li><p><strong>成本</strong>：需要在一个代表性语料上计算协方差矩阵（一次性，<span class="arithmatex">\(O(N \cdot d^2)\)</span>）。变换本身在推理时只是简单的矩阵乘法。</p></li>
<li><p><strong>替代方法</strong>：对比微调（SimCSE）、基于 flow 的归一化，或使用促进各向同性的正则项进行训练。</p></li>
</ul>
</div>

### 1.3.5　自注意力机制 {#self-attention-mechanism}

<p>自注意力是核心操作，它允许每个 Token 关注序列中的每一个其他 Token，基于相关性计算加权组合。</p>

<div class="admonition note"><p class="admonition-title">缩放点积注意力（Scaled Dot-Product Attention）</p>
<p>给定输入序列 <span class="arithmatex">\(X \in \mathbb{R}^{n \times d}\)</span>，我们计算：</p>



<div class="arithmatex">\[Q = XW_Q, \quad K = XW_K, \quad V = XW_V \quad (W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k})\]</div>



<div class="arithmatex">\[\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V\]</div>

<p>其中 <span class="arithmatex">\(M\)</span> 是<strong>因果掩码</strong>（用于自回归模型）：若 <span class="arithmatex">\(i \geq j\)</span> 则 <span class="arithmatex">\(M_{ij} = 0\)</span>，否则为 <span class="arithmatex">\(-\infty\)</span>。</p>

<p><strong>直觉</strong>：每个 Token“关注”所有先前的 Token，基于 query-key 相似度计算其 value 的加权平均。</p>
</div>

##### 计算复杂度。 {#computational-complexity.}

<p>朴素的注意力计算在序列长度上具有<strong>平方代价</strong>：</p>

<ul>
<li><p><strong>时间</strong>：<span class="arithmatex">\(O(n^2 \cdot d)\)</span>——计算 <span class="arithmatex">\(QK^T\)</span> 需要 <span class="arithmatex">\(n^2\)</span> 个点积，每个维度为 <span class="arithmatex">\(d_k\)</span>。</p></li>
<li><p><strong>内存</strong>：<span class="arithmatex">\(O(n^2)\)</span>——必须物化完整的注意力矩阵才能应用 Softmax。</p></li>
</ul>

<p>对于一个 <span class="arithmatex">\(d = 4096\)</span> 的 128K Token 上下文，仅注意力矩阵就是 <span class="arithmatex">\(128\text{K} \times 128\text{K} = 164\)</span> 亿个元素（FP32 下为 64 GB）。这种平方扩展是长上下文 LLM 的根本瓶颈。</p>

<p class="table-caption" id="table-1-4"><span class="table-number">表 1.4　</span>注意力代价的扩展：为何朴素实现对长序列不可行。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>序列长度</strong></th><th scope="col"><strong>注意力操作数</strong></th><th scope="col"><strong>矩阵大小</strong></th><th scope="col"><strong>实际影响</strong></th></tr></thead>
<tbody><tr><td>2K</td><td>4M</td><td>16 MB</td><td>快速；可放入 SRAM</td></tr>
<tr><td>8K</td><td>64M</td><td>256 MB</td><td>用 FlashAttention 可应对</td></tr>
<tr><td>32K</td><td>1B</td><td>4 GB</td><td>需要内存高效内核</td></tr>
<tr><td>128K</td><td>16B</td><td>64 GB</td><td>超出单块 GPU 的 HBM</td></tr>
<tr><td>1M</td><td>1T</td><td>4 TB</td><td>不使用次平方方法则不可行</td></tr></tbody>
</table>
</div>

##### 驯服注意力代价的方法。 {#approaches-to-taming-attention-cost.}

<p>有几类方案可以应对这种平方瓶颈：</p>

<ol>
<li><p><strong>具备 IO 感知的精确注意力（FlashAttention <a class="book-citation" id="cite-dao2022flashattention-1" href="../../references.md#ref-dao2022flashattention" aria-label="参考文献 65">[65]</a>）</strong>：不降低计算复杂度，但通过将注意力计算分块到能放入 SRAM 的瓦片中，消除了在 HBM 中物化 <span class="arithmatex">\(n \times n\)</span> 矩阵的需要。关键的是，FlashAttention 与下面的稀疏模式<strong>正交</strong>——它是一种执行引擎，而非注意力模式。生产系统通常将 FlashAttention 与滑动窗口或块稀疏掩码结合使用，同时获得 IO 效率和 FLOPs 减少。我们在第 <a href="reading-02.md#flash-attention-algorithm-and-hardware-awareness">1.6</a> 节详细介绍该算法。</p></li>
<li><p><strong>滑动窗口 / 局部注意力</strong>：每个 Token 只关注最近的 <span class="arithmatex">\(w\)</span> 个 Token（例如 <span class="arithmatex">\(w = 4096\)</span>）。代价变为 <span class="arithmatex">\(O(n \cdot w)\)</span>——在 <span class="arithmatex">\(n\)</span> 上是线性的。被 Mistral <a class="book-citation" id="cite-jiang2023mistral-2" href="../../references.md#ref-jiang2023mistral" aria-label="参考文献 165">[165]</a>（窗口 <span class="arithmatex">\(= 4096\)</span>）和 Longformer <a class="book-citation" id="cite-beltagy2020longformer-1" href="../../references.md#ref-beltagy2020longformer" aria-label="参考文献 23">[23]</a> 采用。以全局上下文换效率；之所以有效，是因为实践中大部分注意力是局部的。在现代技术栈中，滑动窗口掩码在 FlashAttention 内核<em>内部</em>执行。</p></li>
<li><p><strong>稀疏注意力模式</strong>：将局部窗口与周期性的全局 Token 结合（例如，每隔 512 个 Token 关注全部）。BigBird <a class="book-citation" id="cite-zaheer2020bigbird-1" href="../../references.md#ref-zaheer2020bigbird" aria-label="参考文献 416">[416]</a> 和 LongT5 <a class="book-citation" id="cite-guo2022longt5-1" href="../../references.md#ref-guo2022longt5" aria-label="参考文献 125">[125]</a> 使用此方法。以 <span class="arithmatex">\(O(n\sqrt{n})\)</span> 代价保留一些长程连通性。同样，FlashAttention 作为非零注意力块的底层内核。</p></li>
<li><p><strong>线性注意力 / 状态空间模型</strong>：利用结合律将 <span class="arithmatex">\(\text{softmax}(QK^T)V\)</span> 替换为 <span class="arithmatex">\(\phi(Q)(\phi(K)^T V)\)</span>，或重构为递归形式（Mamba <a class="book-citation" id="cite-gu2023mamba-1" href="../../references.md#ref-gu2023mamba" aria-label="参考文献 122">[122]</a>、RWKV <a class="book-citation" id="cite-peng2023rwkv-1" href="../../references.md#ref-peng2023rwkv" aria-label="参考文献 288">[288]</a>）。理论上总代价为 <span class="arithmatex">\(O(n \cdot d^2)\)</span>。与上面方法 2–3 不同，这些是<em>架构上的替换</em>，改变了模型的表达能力——无 Softmax 的注意力本质上表达力更弱，经验上这些模型在需要精确长程检索或复杂推理的任务上仍然落后于 Transformer。</p></li>
<li><p><strong>KV 缓存压缩</strong>：在推理时，压缩或驱逐旧的 KV 对以限制内存。技术包括：H<span class="arithmatex">\(_2\)</span>O <a class="book-citation" id="cite-zhang2023h2o-1" href="../../references.md#ref-zhang2023h2o" aria-label="参考文献 429">[429]</a>（重击者预言机——只保留高注意力的 Key）、StreamingLLM <a class="book-citation" id="cite-xiao2024streamingllm-1" href="../../references.md#ref-xiao2024streamingllm" aria-label="参考文献 394">[394]</a>（保留初始“注意力陷阱（Attention Sink）”Token + 最近窗口），以及量化 KV 缓存 <a class="book-citation" id="cite-liu2024kivi-1" href="../../references.md#ref-liu2024kivi" aria-label="参考文献 237">[237]</a>。</p></li>
</ol>

<div class="admonition tip"><p class="admonition-title">FlashAttention + 稀疏模式 = 两全其美</p>
<p>一个常见误解是 FlashAttention 是稀疏注意力的<em>替代</em>。事实并非如此——它是注意力内核的 IO 优化，可以与任何注意力掩码自由组合。现代生产系统（如 Mistral、DeepSeek）使用 FlashAttention 作为滑动窗口或块稀疏掩码<em>底层</em>的执行引擎。这同时为你带来 FLOPs 的减少（来自稀疏性）和最优的内存访问模式（来自瓦片化）。RingAttention <a class="book-citation" id="cite-liu2023ringattention-1" href="../../references.md#ref-liu2023ringattention" aria-label="参考文献 221">[221]</a> 将其进一步扩展到多设备场景，沿序列维度将瓦片化的计算分布到多块 GPU 上。</p>

<p>线性注意力和状态空间模型（Mamba、RWKV）是一种真正不同的架构选择——它们为了 <span class="arithmatex">\(O(n)\)</span> 计算而牺牲完整的成对交互。虽然理论上优雅，它们在知识密集或长程推理任务上仍未达到 Transformer 的质量，前沿实验室继续使用精确注意力（结合 FlashAttention + 稀疏性）作为骨干。</p>
</div>

### 1.3.6　多头注意力 {#multi-head-attention}

<p>多头注意力不是计算单一的注意力函数，而是并行运行多个注意力操作，每个都学习聚焦于输入的不同方面（句法、语义、位置等）。</p>

<div class="admonition note"><p class="admonition-title">多头注意力（Multi-Head Attention）</p>
<p>不使用一个 <span class="arithmatex">\(d\)</span> 维 Key/Value 的注意力函数，而是使用 <span class="arithmatex">\(H\)</span> 个维度为 <span class="arithmatex">\(d_k = d/H\)</span> 的并行头：</p>



<div class="arithmatex">\[\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O\]</div>

<p>每个头可以学习不同的注意力模式（例如，一个头处理句法，另一个处理语义，再一个处理位置邻近性）。</p>

<p><strong>分组查询注意力（Grouped Query Attention，GQA）</strong>：Llama-3 <a class="book-citation" id="cite-grattafiori2024llama3-2" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a> 使用比 Q 头更少的 K、V 头（例如，8 个 KV 头被 32 个 Q 头共享）。这将 KV 缓存大小减小 <span class="arithmatex">\(4\times\)</span>，质量损失极小。</p>

<h4 id="heading-1-29">多头潜在注意力（Multi-head Latent Attention，MLA）。</h4>

<p>DeepSeek-V2 <a class="book-citation" id="cite-deepseekv2-1" href="../../references.md#ref-deepseekv2" aria-label="参考文献 69">[69]</a> 引入了更激进的压缩：与 GQA 共享 KV 头不同，MLA 把<em>全部</em>键值信息压缩为每个 token 的单一低秩潜向量：</p>



<div class="arithmatex">\[c_t = W_{DKV} \, h_t, \quad K_t = W_{UK} \, c_t, \quad V_t = W_{UV} \, c_t\]</div>

<p>其中 <span class="arithmatex">\(c_t \in \mathbb{R}^{d_c}\)</span>，且 <span class="arithmatex">\(d_c \ll n_h \cdot d_h\)</span>。KV 缓存只存储 <span class="arithmatex">\(c_t\)</span>——甚至比 GQA 还小——而解压矩阵 <span class="arithmatex">\(W_{UK}, W_{UV}\)</span> 会即时重建完整的键与值。由于压缩是端到端学到的，其质量达到甚至超过 GQA。MLA 使 DeepSeek 后续的 Sparse Attention 得以实现次二次复杂度的长上下文扩展，如今已是 DeepSeek-V3/V4 架构家族的基础。</p>
</div>

### 1.3.7　位置编码 {#positional-encodings}

<p>Transformer 在构造上是置换等变的——没有位置信息，模型无法区分 “the cat sat on the mat” 和 “mat the on sat cat the”。位置编码注入序列顺序信号，使注意力能够推理 Token 的距离和方向。</p>

<p class="table-caption" id="table-1-5"><span class="table-number">表 1.5　</span>现代 LLM 中的位置编码方法。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>方法</strong></th><th scope="col"><strong>使用者</strong></th><th scope="col"><strong>核心思想</strong></th></tr></thead>
<tbody><tr><td>正弦（Sinusoidal）</td><td>原始 Transformer</td><td>不同频率的固定 <span class="arithmatex">\(\sin/\cos\)</span>。无需学习。</td></tr>
<tr><td>学习的绝对位置</td><td>GPT-2 <a class="book-citation" id="cite-radford2019gpt2-1--section-03" href="../../references.md#ref-radford2019gpt2" aria-label="参考文献 301">[301]</a>, BERT <a class="book-citation" id="cite-devlin2019bert-2" href="../../references.md#ref-devlin2019bert" aria-label="参考文献 76">[76]</a></td><td>每个位置的学习嵌入。受限于训练长度。</td></tr>
<tr><td>RoPE</td><td>Llama <a class="book-citation" id="cite-grattafiori2024llama3-3" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>, Qwen <a class="book-citation" id="cite-qwen2024qwen25-2" href="../../references.md#ref-qwen2024qwen25" aria-label="参考文献 350">[350]</a>, Mistral <a class="book-citation" id="cite-jiang2023mistral-3" href="../../references.md#ref-jiang2023mistral" aria-label="参考文献 165">[165]</a></td><td>将 Q、K 向量按位置相关的角度旋转。通过 NTK 感知缩放进行外推。</td></tr>
<tr><td>ALiBi</td><td>BLOOM <a class="book-citation" id="cite-workshop2023bloom-1" href="../../references.md#ref-workshop2023bloom" aria-label="参考文献 383">[383]</a>, MPT <a class="book-citation" id="cite-mosaicml2023mpt-1" href="../../references.md#ref-mosaicml2023mpt" aria-label="参考文献 260">[260]</a></td><td>不使用位置嵌入；在注意力分数上添加线性偏置 <span class="arithmatex">\(-m|i-j|\)</span>。简单，外推性好。</td></tr></tbody>
</table>
</div>

##### 正弦（固定）位置编码。 {#sinusoidal-fixed-positional-encoding.}

<p>原始 Transformer <a class="book-citation" id="cite-vaswani2017attention-3" href="../../references.md#ref-vaswani2017attention" aria-label="参考文献 357">[357]</a> 中提出，该方法在几何间隔的频率上使用固定的正弦函数：</p>



<div class="arithmatex">\[\text{PE}(pos, 2i) = \sin\!\Bigl(\frac{pos}{10000^{2i/d}}\Bigr), \qquad
  \text{PE}(pos, 2i{+}1) = \cos\!\Bigl(\frac{pos}{10000^{2i/d}}\Bigr)\]</div>

<p>其中 <span class="arithmatex">\(pos\)</span> 是 Token 位置，<span class="arithmatex">\(i\)</span> 是维度索引，<span class="arithmatex">\(d\)</span> 是模型维度。</p>

<p><strong>动机：</strong>每个频率以不同尺度编码位置（类比二进制计数）。作者假设模型可以学会关注相对位置，因为 <span class="arithmatex">\(\text{PE}(pos+k)\)</span> 可以表示为 <span class="arithmatex">\(\text{PE}(pos)\)</span> 的线性函数。</p>

<p><strong>优点：</strong>零学习参数；确定性；理论上支持任意长度。<br><strong>缺点：</strong>实践中外推超出训练长度时表现不佳；模型必须间接地从绝对信号中学习解码相对位置；已大体被取代。</p>

##### 学习的绝对位置嵌入。 {#learned-absolute-positional-embedding.}

<p>被 GPT-2 <a class="book-citation" id="cite-radford2019gpt2-2" href="../../references.md#ref-radford2019gpt2" aria-label="参考文献 301">[301]</a> 和 BERT <a class="book-citation" id="cite-devlin2019bert-3" href="../../references.md#ref-devlin2019bert" aria-label="参考文献 76">[76]</a> 采用：一个可学习的嵌入矩阵 <span class="arithmatex">\(\mathbf{E}_{\text{pos}} \in \mathbb{R}^{L_{\max} \times d}\)</span> 被加到 Token 嵌入上：</p>



<div class="arithmatex">\[h_0^{(pos)} = \text{TokenEmbed}(x_{pos}) + \mathbf{E}_{\text{pos}}[pos]\]</div>

<p><strong>动机：</strong>让模型自己学习对任务最优的位置表示，而不是强加一个固定结构。</p>

<p><strong>优点：</strong>最大的灵活性；实现简单；对短序列通常优于正弦编码。<br><strong>缺点：</strong>硬编码的最大长度 <span class="arithmatex">\(L_{\max}\)</span>；无法泛化到其之外；<span class="arithmatex">\(L_{\max}\)</span> 末尾附近的嵌入训练不足；增加 <span class="arithmatex">\(L_{\max} \times d\)</span> 个参数。</p>

##### 旋转位置编码（Rotary Position Embedding，RoPE）。 {#rotary-position-embedding-rope.}

<p>RoPE <a class="book-citation" id="cite-su2024roformer-1" href="../../references.md#ref-su2024roformer" aria-label="参考文献 341">[341]</a> 通过在 2D 子空间中<em>旋转</em> Query 和 Key 向量来编码位置：</p>



<div class="arithmatex">\[\text{RoPE}(x_m, m) = \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \\ \vdots \\ x_m^{(d-1)} \\ x_m^{(d)} \end{pmatrix}
  \odot
  \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d/2} \\ \cos m\theta_{d/2} \end{pmatrix}
  +
  \begin{pmatrix} -x_m^{(2)} \\ x_m^{(1)} \\ \vdots \\ -x_m^{(d)} \\ x_m^{(d-1)} \end{pmatrix}
  \odot
  \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d/2} \\ \sin m\theta_{d/2} \end{pmatrix}\]</div>

<p>其中 <span class="arithmatex">\(\theta_i = 10000^{-2i/d}\)</span>，<span class="arithmatex">\(m\)</span> 是位置索引。关键性质是：旋转后的 Query 与 Key 之间的点积只依赖于相对位置：</p>



<div class="arithmatex">\[\langle \text{RoPE}(q_m, m),\; \text{RoPE}(k_n, n) \rangle = f(q_m, k_n, m-n)\]</div>

<p><strong>动机：</strong>在无需显式偏置项的情况下实现相对位置编码，同时保持与线性注意力和 KV 缓存的兼容性。</p>

<p><strong>优点：</strong>天然相对；无额外参数；与高效推理兼容；可以通过 NTK 感知缩放 <a class="book-citation" id="cite-peng2023yarn-1" href="../../references.md#ref-peng2023yarn" aria-label="参考文献 289">[289]</a> 或 YaRN（调整 <span class="arithmatex">\(\theta\)</span> 基数或插值频率）扩展到更长的上下文。<br><strong>缺点：</strong>每次注意力操作的计算略多（旋转 + 交错）；外推需要显式的缩放策略；在 2D 子空间中的旋转施加了一种结构，对所有任务未必最优。</p>

<div class="admonition tip"><p class="admonition-title">RoPE 长度扩展</p>
<p>将训练在 <span class="arithmatex">\(L\)</span> 长度上的 RoPE 模型扩展到上下文长度 <span class="arithmatex">\(L&#x27; &gt; L\)</span>：</p>

<ul>
<li><p><strong>位置插值（Position interpolation）：</strong>将位置按 <span class="arithmatex">\(L/L&#x27;\)</span> 缩放，使所有位置落在 <span class="arithmatex">\([0, L]\)</span> 内。简单但压缩了分辨率。</p></li>
<li><p><strong>NTK 感知缩放：</strong>增大 <span class="arithmatex">\(\theta\)</span> 基数（例如 <span class="arithmatex">\(10000 \to 10000 \cdot (L&#x27;/L)^{d/(d-2)}\)</span>），有效地拉伸高频成分，同时保留低频成分。</p></li>
<li><p><strong>YaRN</strong> <a class="book-citation" id="cite-peng2023yarn-2" href="../../references.md#ref-peng2023yarn" aria-label="参考文献 289">[289]</a>：将 NTK 缩放与注意力温度校正 <span class="arithmatex">\(t = 0.1 \ln(s) + 1\)</span> 结合，以补偿更长距离上熵的增加。</p></li>
</ul>
</div>

##### ALiBi（带线性偏置的注意力，Attention with Linear Biases）。 {#alibi-attention-with-linear-biases.}

<p>ALiBi <a class="book-citation" id="cite-press2022train-1" href="../../references.md#ref-press2022train" aria-label="参考文献 292">[292]</a> 采取了根本不同的方法：<em>完全不使用位置嵌入</em>。取而代之，从注意力分数中减去一个静态的线性惩罚：</p>



<div class="arithmatex">\[\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} - m \cdot \bigl[|i-j|\bigr]_{i,j}\right) V\]</div>

<p>其中 <span class="arithmatex">\(m\)</span> 是头特定的斜率（几何设置：对总共 <span class="arithmatex">\(H\)</span> 个头中的第 <span class="arithmatex">\(h\)</span> 个头取 <span class="arithmatex">\(m_h = 2^{-8h/H}\)</span>）。偏置 <span class="arithmatex">\(-m|i-j|\)</span> 创建了一个软局部注意力窗口，其宽度因头而异。</p>

<p><strong>动机：</strong>位置应使注意力偏向邻近 Token（近因先验），同时不干扰嵌入空间。通过纯粹在注意力分数空间中操作，ALiBi 避免了用位置信号污染 Token 表征。</p>

<p><strong>优点：</strong>出色的长度外推（在 1k 上训练，在 8k+ 仍可用）；零参数；实现极简；头特定的斜率提供多尺度局部性。<br><strong>缺点：</strong>对需要精确长程位置推理的任务（如“第 5 个词是什么？”）表达力较弱；线性衰减是一种强归纳偏置，可能不适合所有领域；由于 RoPE 在短上下文上表现更好，最近的模型中已大体被 RoPE 取代。</p>

<p class="table-caption" id="table-1-6"><span class="table-number">表 1.6　</span>位置编码对比：实际权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"></th><th scope="col"><strong>正弦（Sinusoidal）</strong></th><th scope="col"><strong>学习绝对位置</strong></th><th scope="col"><strong>RoPE</strong></th><th scope="col"><strong>ALiBi</strong></th></tr></thead>
<tbody><tr><td>额外参数</td><td>无</td><td><span class="arithmatex">\(L_{\max} \times d\)</span></td><td>无</td><td>无</td></tr>
<tr><td>位置类型</td><td>绝对</td><td>绝对</td><td>相对</td><td>相对（隐式）</td></tr>
<tr><td>长度外推</td><td>差</td><td>无</td><td>好（带缩放）</td><td>优秀</td></tr>
<tr><td>计算开销</td><td>可忽略</td><td>可忽略</td><td>小</td><td>可忽略</td></tr>
<tr><td>主导时期</td><td>2017–19</td><td>2018–20</td><td>2022–至今</td><td>2022–23</td></tr></tbody>
</table>
</div>

##### 扩展到极长上下文（100K–1M+ Token）。 {#scaling-to-extremely-long-contexts-100k1m-tokens.}

<p>现代前沿模型（Claude <a class="book-citation" id="cite-anthropic2024claude3-1" href="../../references.md#ref-anthropic2024claude3" aria-label="参考文献 11">[11]</a> 拥有 200K–1M 上下文、Gemini 1.5 <a class="book-citation" id="cite-geminiteam2024gemini15-1" href="../../references.md#ref-geminiteam2024gemini15" aria-label="参考文献 108">[108]</a> 达到 1M+、GPT-4 <a class="book-citation" id="cite-openai2023gpt4-2" href="../../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a> 128K）需要在远超训练长度时仍然忠实的位置编码。当今的主流解决方案：</p>

<ol>
<li><p><strong>带频率缩放的 RoPE</strong>：将 RoPE 扩展到训练长度之外的标准方法。无需重新训练，对基础频率 <span class="arithmatex">\(\theta\)</span> 进行重新缩放：</p>



<div class="arithmatex">\[\theta&#x27;_i = \theta_i \cdot \left(\frac{L_{\text{target}}}{L_{\text{train}}}\right)^{2i/d}\]</div>

<p>变体包括：</p>

<ul>
<li><p><strong>线性缩放</strong>（位置插值） <a class="book-citation" id="cite-chen2023extending-1" href="../../references.md#ref-chen2023extending" aria-label="参考文献 48">[48]</a>：简单地将位置索引除以因子 <span class="arithmatex">\(s\)</span>。便宜但在高扩展比时降低质量。</p></li>
<li><p><strong>NTK 感知缩放</strong> <a class="book-citation" id="cite-peng2023yarn-3" href="../../references.md#ref-peng2023yarn" aria-label="参考文献 289">[289]</a>：缩放基础频率 <span class="arithmatex">\(\theta = 10000 \to 10000 \cdot s^{d/(d-2)}\)</span>。保留高频（局部）信息的同时扩展低频（全局）范围。</p></li>
<li><p><strong>YaRN</strong> <a class="book-citation" id="cite-peng2023yarn-4" href="../../references.md#ref-peng2023yarn" aria-label="参考文献 289">[289]</a>（Yet another RoPE extensioN）：将 NTK 缩放与注意力温度校正以及在一个小型长上下文语料上的微调结合。Llama-3 用它将 8K 训练扩展到 128K 部署。</p></li>
<li><p><strong>动态 NTK（Dynamic NTK）</strong> <a class="book-citation" id="cite-peng2023yarn-5" href="../../references.md#ref-peng2023yarn" aria-label="参考文献 289">[289]</a>：在推理时根据实际序列长度即时调整缩放因子。无需固定扩展比——模型随上下文增长而适应。</p></li>
</ul></li>
<li><p><strong>在长数据上继续预训练</strong>：即使使用 RoPE 缩放，模型也能从在长文档上的短期继续预训练阶段（1–5B Token）中受益。这教会模型真正地<em>使用</em>远距离上下文，而不仅是在位置上容忍它。Llama-3.1 使用渐进式计划：8K <span class="arithmatex">\(\to\)</span> 64K <span class="arithmatex">\(\to\)</span> 128K。</p></li>
<li><p><strong>Ring Attention / 分块并行</strong> <a class="book-citation" id="cite-liu2023ringattention-2" href="../../references.md#ref-liu2023ringattention" aria-label="参考文献 221">[221]</a>：对于超出单 GPU 内存的序列（1M+ Token），Ring Attention 以环形拓扑将序列分布在 GPU 之间。每块 GPU 持有一个块并在环上传递 KV 块，计算局部注意力瓦片。这使内存能够随 GPU 数量线性扩展，同时保留精确注意力。</p></li>
<li><p><strong>混合架构</strong>：一些系统将大多数层的局部滑动窗口（例如 4K）与选定层（如每 4 层一次）的全注意力结合。这为大部分计算提供 <span class="arithmatex">\(O(n \cdot w)\)</span> 代价，同时维持全局信息流。</p></li>
</ol>

<div class="admonition warning"><p class="admonition-title">长上下文 <span class="arithmatex">\(\neq\)</span> 长上下文利用</p>
<p>拥有 1M 上下文长度的模型并<em>不</em>必然能有效利用所有 1M Token。“迷失中间（Lost in the Middle）”现象 <a class="book-citation" id="cite-liu2024lost-1" href="../../references.md#ref-liu2024lost" aria-label="参考文献 226">[226]</a> 表明，模型倾向于聚焦于长上下文的开始和结尾，对中间的信息利用不足。有效的长上下文利用需要位置编码支持<em>和</em>在奖励长程检索的任务上进行训练。</p>
</div>

### 1.3.8　前馈网络（MLP） {#feed-forward-network-mlp}

<p>每个 Transformer 块包含一个独立应用于每个位置的 MLP：</p>



<div class="arithmatex">\[\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2\]</div>

<p>其中 <span class="arithmatex">\(W_1 \in \mathbb{R}^{d \times 4d}\)</span>，<span class="arithmatex">\(W_2 \in \mathbb{R}^{4d \times d}\)</span>。现代 LLM 使用：</p>

<ul>
<li><p><strong>SwiGLU 激活</strong>：<span class="arithmatex">\(\text{FFN}(x) = W_2 (\text{Swish}(W_1 x) \odot W_3 x)\)</span>——被 Llama <a class="book-citation" id="cite-grattafiori2024llama3-4" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>、Mistral <a class="book-citation" id="cite-jiang2023mistral-4" href="../../references.md#ref-jiang2023mistral" aria-label="参考文献 165">[165]</a> 采用。需要 3 个权重矩阵，但带来更好的性能。</p></li>
<li><p>隐藏维度通常为 <span class="arithmatex">\(8/3 \times d\)</span>（取整到 256 的倍数以适配 Tensor Core 效率）。</p></li>
</ul>

<div class="admonition tip"><p class="admonition-title">FFN 作为存储器</p>
<p>近期工作 <a class="book-citation" id="cite-geva2021transformer-1" href="../../references.md#ref-geva2021transformer" aria-label="参考文献 110">[110]</a> 表明 FFN 层充当一种<em>键值存储器</em>：<span class="arithmatex">\(W_1\)</span> 的行是 Key（要匹配的模式），<span class="arithmatex">\(W_2\)</span> 的列是 Value（要输出的信息）。FFN 基于当前隐藏状态“检索”存储的知识。</p>
</div>

### 1.3.9　层归一化（Layer Normalization） {#layer-normalization}

<p>层归一化通过在特征维度上对激活进行归一化来稳定训练。它相对于注意力/FFN 子层的位置显著影响训练动态。</p>

##### LayerNorm 如何工作。 {#how-layernorm-works.}

<p>给定一个隐藏状态向量 <span class="arithmatex">\(\mathbf{x} \in \mathbb{R}^d\)</span>（单个 Token 的表征），LayerNorm <a class="book-citation" id="cite-ba2016layernorm-1" href="../../references.md#ref-ba2016layernorm" aria-label="参考文献 15">[15]</a> 计算：</p>



<div class="arithmatex">\[\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\tag{1.5}\]</div>

<p>其中：</p>

<ul>
<li><p><span class="arithmatex">\(\mu = \frac{1}{d}\sum_{i=1}^{d} x_i\)</span>（在 <span class="arithmatex">\(d\)</span> 个特征维度上的均值）</p></li>
<li><p><span class="arithmatex">\(\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2\)</span>（在特征上的方差）</p></li>
<li><p><span class="arithmatex">\(\gamma, \beta \in \mathbb{R}^d\)</span> 是<strong>学习的</strong>缩放和平移参数（按维度）</p></li>
<li><p><span class="arithmatex">\(\epsilon \approx 10^{-5}\)</span> 防止除以零</p></li>
</ul>

<p><strong>与 BatchNorm 的关键区别</strong>：LayerNorm 在单个样本的<em>特征维度</em>上归一化，而非跨批次。这使其独立于批次大小，并在训练和推理时表现一致。</p>

##### RMSNorm——现代简化版。 {#rmsnorm-the-modern-simplification.}

<p>均方根归一化（Root Mean Square Normalization，RMSNorm） <a class="book-citation" id="cite-zhang2019rmsnorm-1" href="../../references.md#ref-zhang2019rmsnorm" aria-label="参考文献 421">[421]</a> 去掉了均值中心化步骤，仅按均方根归一化：</p>



<div class="arithmatex">\[\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}, \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
\tag{1.6}\]</div>

<p>没有 <span class="arithmatex">\(\beta\)</span>（平移）参数，也没有均值减法——只有缩放。这每个 Token 节省一次归约操作，在 GPU 上快约 5–10%，同时保持等同的模型质量。所有现代 LLM（Llama、Mistral、Qwen）都使用 RMSNorm。</p>

<div class="admonition note"><p class="admonition-title">Pre-LN vs Post-LN</p>
<ul>
<li><p><strong>Post-LN</strong>（原始 Transformer）：<span class="arithmatex">\(\text{LayerNorm}(h + \text{Attn}(h))\)</span>。需要仔细的 warmup；训练可能不稳定。</p></li>
<li><p><strong>Pre-LN</strong>（GPT-2+，所有现代 LLM）：<span class="arithmatex">\(h + \text{Attn}(\text{LayerNorm}(h))\)</span>。稳定训练；允许更高的学习率。</p></li>
<li><p><strong>RMSNorm</strong>（Llama <a class="book-citation" id="cite-grattafiori2024llama3-5" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a>, Mistral <a class="book-citation" id="cite-jiang2023mistral-5" href="../../references.md#ref-jiang2023mistral" aria-label="参考文献 165">[165]</a>）：不带均值中心化的简化 LayerNorm：<span class="arithmatex">\(\text{RMSNorm}(x) = x / \text{RMS}(x) \cdot \gamma\)</span>。稍快，同等质量。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">为什么归一化对深度网络很重要</p>
<p>没有归一化，激活倾向于在层之间指数增长或收缩（爆炸/消失激活）。一个无 LayerNorm 的 128 层 Transformer 在第一层和最后一层之间，量级会变化 <span class="arithmatex">\(10^{30}\times\)</span>。归一化将每层输出约束在可预测的范围内，使梯度流动稳定，并允许优化器在整个网络中使用一致的学习率。</p>
</div>

### 1.3.10　模型规模参考 {#model-size-reference}

<p>下表总结了广泛使用的开放权重模型（截至 2025 年的最新版本）的关键架构参数，为理解规模和设计选择提供了快速参考。</p>

<p class="table-caption" id="table-1-7"><span class="table-number">表 1.7　</span>流行开放权重 LLM 的架构参数（2024–2025 代）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模型</strong></th><th scope="col"><strong>参数</strong></th><th scope="col"><strong>层数</strong></th><th scope="col"><strong><span class="arithmatex">\(d\)</span></strong></th><th scope="col"><strong>头数</strong></th><th scope="col"><strong>KV 头数</strong></th><th scope="col"><strong>上下文</strong></th></tr></thead>
<tbody><tr><td>Llama-3.1 8B <a class="book-citation" id="cite-grattafiori2024llama3-6" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a></td><td>8B</td><td>32</td><td>4096</td><td>32</td><td>8</td><td>128K</td></tr>
<tr><td>Llama-3.1 405B <a class="book-citation" id="cite-grattafiori2024llama3-7" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a></td><td>405B</td><td>126</td><td>16384</td><td>128</td><td>8</td><td>128K</td></tr>
<tr><td>Llama-4 Maverick <a class="book-citation" id="cite-meta2025llama4-1" href="../../references.md#ref-meta2025llama4" aria-label="参考文献 4">[4]</a></td><td>400B (17B 活跃)</td><td>48</td><td>5120</td><td>40</td><td>8</td><td>1M</td></tr>
<tr><td>Mistral Large 2 <a class="book-citation" id="cite-jiang2024mistrallarge2-1" href="../../references.md#ref-jiang2024mistrallarge2" aria-label="参考文献 5">[5]</a></td><td>123B</td><td>88</td><td>12288</td><td>96</td><td>8</td><td>128K</td></tr>
<tr><td>Qwen-2.5 72B <a class="book-citation" id="cite-qwen2024qwen25-3" href="../../references.md#ref-qwen2024qwen25" aria-label="参考文献 350">[350]</a></td><td>72B</td><td>80</td><td>8192</td><td>64</td><td>8</td><td>128K</td></tr>
<tr><td>DeepSeek-V3 <a class="book-citation" id="cite-deepseekv3-1" href="../../references.md#ref-deepseekv3" aria-label="参考文献 70">[70]</a></td><td>671B (37B 活跃)</td><td>61</td><td>7168</td><td>128</td><td>MLA</td><td>128K</td></tr></tbody>
</table>
</div>

<p><em>注</em>：标注“活跃”参数的模型使用混合专家（Mixture-of-Experts，MoE）架构——总参数表示模型容量，而活跃参数反映每个 Token 的计算代价。DeepSeek-V3 使用多头潜在注意力（Multi-head Latent Attention，MLA）而非标准 GQA，将 KV 压缩到低秩潜空间中。</p>

### 1.3.11　注意力病态 {#attention-pathologies}

<p>虽然注意力机制功能强大，但它表现出实践者必须理解的系统性失败模式——特别是在扩展到长上下文或解释模型行为时。</p>

#### 注意力陷阱（Attention Sink） {#attention-sink}

##### 现象。 {#the-phenomenon.}

<p>Xiao 等 <a class="book-citation" id="cite-xiao2024efficient-1" href="../../references.md#ref-xiao2024efficient" aria-label="参考文献 393">[393]</a> 发现 Transformer 模型对序列中的<em>第一个 Token</em> 分配了不成比例的高注意力分数——无论其语义内容如何。即使第一个 Token 是无意义的 <code>&lt;BOS&gt;</code> 标记，所有层中的注意力头都持续地关注它，有时占总注意力质量的 20–50%。</p>

##### 为什么会发生。 {#why-it-happens.}

<p>Softmax 注意力必须产生有效的概率分布（<span class="arithmatex">\(\sum_j \alpha_j = 1\)</span>）。当没有 Key 对 Query 特别相关时，模型需要一个“倾倒”位置来安放未使用的注意力质量。在训练期间，第一个 Token 成为这个默认陷阱，因为它总是存在且在位置上可预测。它作为一个<em>无操作的注意力目标</em>——模型已经学会将不相关的注意力路由到那里，而不是不可预测地分配它。</p>



<div class="arithmatex">\[\alpha_{\text{sink}} = \frac{\exp(q^\top k_0 / \sqrt{d})}{\sum_{j} \exp(q^\top k_j / \sqrt{d})} \gg \frac{1}{n} \quad \text{(even when } k_0 \text{ is semantically irrelevant)}\]</div>

##### 后果。 {#consequences.}

<ul>
<li><p><strong>流式推理失败</strong>：使用滑动窗口 KV 缓存时，驱逐第一个 Token 会导致困惑度灾难性地飙升——模型失去了它的注意力陷阱。</p></li>
<li><p><strong>误导性的可解释性</strong>：朴素的注意力可视化暗示第一个 Token“重要”，而实际上它只是一种数学伪影。</p></li>
<li><p><strong>上下文窗口浪费</strong>：陷阱 Token 占用一个 KV 缓存槽位，却不携带有用信息。</p></li>
</ul>

##### 解决方案。 {#solutions.}

<ul>
<li><p><strong>StreamingLLM</strong> <a class="book-citation" id="cite-xiao2024efficient-2" href="../../references.md#ref-xiao2024efficient" aria-label="参考文献 393">[393]</a>：始终在 KV 缓存中保留前 <span class="arithmatex">\(k\)</span> 个 Token（“注意力陷阱”）以及最近的滑动窗口。在有界内存下实现无限长度生成。</p></li>
<li><p><strong>设计上的陷阱 Token</strong>：一些模型（如 Mistral）在训练期间预置专门的陷阱 Token，明确用于吸收残余注意力。</p></li>
<li><p><strong>Softmax 替代</strong>：用 ReLU 注意力或 Sigmoid 门控替换 Softmax，使零注意力可以表示而无需倾倒目标。</p></li>
</ul>

#### 注意力稀释（Attention Dilution） {#attention-dilution}

##### 现象。 {#the-phenomenon.-1}

<p>随着序列长度 <span class="arithmatex">\(n\)</span> 增长，每个 Query 必须将其注意力预算分配到更多 Key 上。每个 Token 的平均注意力权重以 <span class="arithmatex">\(O(1/n)\)</span> 下降，使得模型越来越难以集中在少数真正相关的位置上——这个问题被称为<em>注意力稀释（attention dilution）</em>或<em>注意力扩散（attention diffusion）</em> <a class="book-citation" id="cite-liu2024lost-2" href="../../references.md#ref-liu2024lost" aria-label="参考文献 226">[226]</a>。</p>

##### “迷失中间（Lost in the Middle）”效应。 {#the-lost-in-the-middle-effect.}

<p>Liu 等 <a class="book-citation" id="cite-liu2024lost-3" href="../../references.md#ref-liu2024lost" aria-label="参考文献 226">[226]</a> 表明 LLM 表现出 U 形检索曲线：放在长上下文<em>开始</em>或<em>结尾</em>的信息被可靠地检索，但<em>中间</em>的信息常常被忽略。这是注意力稀释叠加 RoPE/ALiBi 位置偏差的直接后果：</p>

##### 为什么会发生。 {#why-it-happens.-1}

<ul>
<li><p><strong>Softmax 饱和</strong>：在 Key 很多时，Softmax 温度有效降低，使分布更加均匀（熵化）。</p></li>
<li><p><strong>位置衰减</strong>：RoPE 的相对位置编码引入了一种随距离的自然衰减，抑制了对距离开始和结尾都较远的中间位置的注意力。</p></li>
<li><p><strong>训练分布</strong>：在较短序列上训练的模型形成了偏向近期上下文的注意力模式。</p></li>
</ul>

##### 缓解策略。 {#mitigation-strategies.}

<ul>
<li><p><strong>显式检索</strong>：将相关上下文放在提示的开始或结尾；使用 RAG 来避免依赖中间位置。</p></li>
<li><p><strong>长上下文训练</strong>：在关键信息位置多样化的长文档上训练 <a class="book-citation" id="cite-fu2024data-1" href="../../references.md#ref-fu2024data" aria-label="参考文献 100">[100]</a>。</p></li>
<li><p><strong>层次化注意力</strong>：如 Mamba <a class="book-citation" id="cite-gu2024mamba-1" href="../../references.md#ref-gu2024mamba" aria-label="参考文献 123">[123]</a> 或 RWKV 等完全避免 <span class="arithmatex">\(O(n^2)\)</span> 注意力瓶颈的架构。</p></li>
<li><p><strong>地标 Token（Landmark tokens）</strong>：在上下文中插入可检索的标记，作为注意力的“路标”。</p></li>
<li><p><strong>温度缩放</strong>：一些实现将注意力 logits 按 <span class="arithmatex">\(\log n\)</span> 缩放，以抵消长序列中的稀释。</p></li>
</ul>

#### 其他注意力现象 {#other-attention-phenomena}

<p class="table-caption" id="table-1-8"><span class="table-number">表 1.8　</span>在大型 Transformer 中观察到的其他注意力模式。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>模式</strong></th><th scope="col"><strong>描述</strong></th><th scope="col"><strong>影响</strong></th></tr></thead>
<tbody><tr><td><strong>注意力头专门化</strong></td><td>不同头学习不同角色：句法头、共指头、位置头 <a class="book-citation" id="cite-voita2019analyzing-1" href="../../references.md#ref-voita2019analyzing" aria-label="参考文献 361">[361]</a></td><td>并非所有头都同等重要；很多可以被剪枝</td></tr>
<tr><td><strong>归纳头（Induction heads）</strong></td><td>实现 [A][B]...[A] <span class="arithmatex">\(\to\)</span> [B] 复制的头 <a class="book-citation" id="cite-olsson2022context-1" href="../../references.md#ref-olsson2022context" aria-label="参考文献 273">[273]</a></td><td>对上下文学习至关重要；在 2 层及以上的模型中涌现</td></tr>
<tr><td><strong>注意力塌缩（Attention collapse）</strong></td><td>在深度网络中，注意力分布可能收敛（所有头关注相同位置）</td><td>损害表达力；通过注意力多样性损失来解决</td></tr>
<tr><td><strong>检索头（Retrieval heads）</strong></td><td>特定头专门从上下文中检索事实信息 <a class="book-citation" id="cite-wu2024retrieval-1" href="../../references.md#ref-wu2024retrieval" aria-label="参考文献 387">[387]</a></td><td>解释了为何剪枝某些头会导致幻觉激增</td></tr></tbody>
</table>
</div>

### 1.3.12　为可解释性而进行注意力可视化 {#visualizing-attention-for-explainability}

<p>注意力权重为模型推理提供了一扇窗户——但必须谨慎解读。</p>

#### 注意力可视化方法 {#attention-visualization-methods}

##### 原始注意力图。 {#raw-attention-maps.}

<p>最简单的方法：为每个头和每层将 <span class="arithmatex">\(n \times n\)</span> 注意力矩阵 <span class="arithmatex">\(A = \text{softmax}(QK^\top/\sqrt{d})\)</span> 绘制为热图。BertViz <a class="book-citation" id="cite-vig2019bertviz-1" href="../../references.md#ref-vig2019bertviz" aria-label="参考文献 359">[359]</a> 等工具可以渲染交互式的多头可视化。</p>

##### 注意力展开（Attention rollout）。 {#attention-rollout.}

<p>单层的原始注意力是有误导性的，因为信息通过残差连接流经<em>所有</em>层。Abnar 和 Zuidema <a class="book-citation" id="cite-abnar2020quantifying-1" href="../../references.md#ref-abnar2020quantifying" aria-label="参考文献 2">[2]</a> 提出了<em>注意力展开（attention rollout）</em>：将各层的注意力矩阵相乘以近似从输入到输出的总信息流：</p>



<div class="arithmatex">\[R^{(l)} = A^{(l)} \cdot R^{(l-1)}, \quad R^{(0)} = I\]</div>

<p>其中 <span class="arithmatex">\(A^{(l)}\)</span> 是第 <span class="arithmatex">\(l\)</span> 层的注意力矩阵（跨头平均），经过调整以包含残差连接：<span class="arithmatex">\(A^{(l)} = 0.5 \cdot A^{(l)}_{\text{raw}} + 0.5 \cdot I\)</span>。</p>

##### 梯度加权的注意力。 {#gradient-weighted-attention.}

<p>将注意力权重与梯度信息结合以识别哪些被关注的 Token 实际<em>影响</em>输出 <a class="book-citation" id="cite-barkan2021grad-1" href="../../references.md#ref-barkan2021grad" aria-label="参考文献 20">[20]</a>：</p>



<div class="arithmatex">\[\text{Relevance}(i) = \alpha_i \cdot \left|\frac{\partial y}{\partial h_i}\right|\]</div>

<p>这回应了“高注意力 <span class="arithmatex">\(\neq\)</span> 高影响”的批评（一个 Token 可以收到高注意力，但通过近零权重路径处理）。</p>

<div class="admonition warning"><p class="admonition-title">注意力不是解释</p>
<p>Jain 和 Wallace <a class="book-citation" id="cite-jain2019attention-1" href="../../references.md#ref-jain2019attention" aria-label="参考文献 160">[160]</a> 表明，注意力权重往往与基于梯度的特征重要性不相关，对抗性的注意力分布可以产生相同的输出。将注意力可视化作为一种<em>假设生成器</em>，而非忠实的解释。对于因果归因，优先选择基于梯度的方法、探测（probing）或机制可解释性。</p>
</div>

#### 使用稀疏自编码器（Sparse Autoencoder，SAE）的机制可解释性 {#mechanistic-interpretability-with-sparse-autoencoders-saes}

##### 可解释性问题。 {#the-interpretability-problem.}

<p>Transformer MLP 和残差流中的单个神经元通常是<em>多义的（polysemantic）</em>——单个神经元对多个不相关的概念激活（例如“蓝色 AND 学术引文 AND 单词‘the’”）。这使得直接的神经元级解释变得不可靠。</p>

##### 稀疏自编码器（Sparse Autoencoder，SAE）。 {#sparse-autoencoders.}

<p>Cunningham 等 <a class="book-citation" id="cite-cunningham2023sparse-1" href="../../references.md#ref-cunningham2023sparse" aria-label="参考文献 63">[63]</a> 和 Bricken 等 <a class="book-citation" id="cite-bricken2023monosemanticity-1" href="../../references.md#ref-bricken2023monosemanticity" aria-label="参考文献 30">[30]</a> 证明，在模型激活上训练稀疏自编码器（Sparse Autoencoder，SAE）可以将多义表征分解为<em>单义特征（monosemantic features）</em>——每个对应单一概念的可解释方向：</p>



<div class="arithmatex">\[h = W_{\text{dec}} \cdot \text{ReLU}(W_{\text{enc}} \cdot x + b_{\text{enc}}) + b_{\text{dec}}\]</div>

<p>其中 <span class="arithmatex">\(W_{\text{enc}} \in \mathbb{R}^{m \times d}\)</span>，<span class="arithmatex">\(m \gg d\)</span>（超完备基），ReLU + 稀疏惩罚确保每个输入只有少数特征被激活。</p>

##### SAE 可解释性的关键发现： {#key-findings-from-sae-interpretability}

<ul>
<li><p>特征是<em>单义的</em>：每个编码一个人类可解释的概念（“Python 代码”、“提及金门大桥”、“第一人称叙述”） <a class="book-citation" id="cite-bricken2023monosemanticity-2" href="../../references.md#ref-bricken2023monosemanticity" aria-label="参考文献 30">[30]</a>。</p></li>
<li><p>特征是<em>可操控的</em>：将某个特征的激活钳制为高/低直接控制模型行为（例如，强制“金门大桥”特征开启会使模型在每个回复中都提到它） <a class="book-citation" id="cite-templeton2024scaling-1" href="../../references.md#ref-templeton2024scaling" aria-label="参考文献 353">[353]</a>。</p></li>
<li><p>特征可组合：复杂行为从简单特征的组合中涌现。</p></li>
<li><p>SAE 可扩展：Templeton 等 <a class="book-citation" id="cite-templeton2024scaling-2" href="../../references.md#ref-templeton2024scaling" aria-label="参考文献 353">[353]</a> 在 Claude 3 Sonnet 上训练了多达 34M 个特征的 SAE，找到了与安全相关的概念（欺骗、谄媚、危险请求）的可解释特征。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">SAE 训练配方</p>
<ol>
<li><p>从特定模型层在一个大型语料上收集激活。</p></li>
<li><p>在隐藏层上以 <span class="arithmatex">\(L_1\)</span> 惩罚训练稀疏自编码器：<span class="arithmatex">\(\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1\)</span>。</p></li>
<li><p>学到的编码器方向（<span class="arithmatex">\(W_{\text{enc}}\)</span> 的行）即候选特征。</p></li>
<li><p>验证：对每个特征，找出最大激活样本并检查语义一致性。</p></li>
<li><p>可选：测量<em>特征吸收（feature absorption）</em>和<em>死特征（dead features）</em>以评估 SAE 质量。</p></li>
</ol>
</div>

#### 自然语言自编码器（Anthropic，2026） {#natural-language-autoencoders-anthropic-2026}

<p>虽然 SAE 将激活分解为可解释的<em>向量</em>，其特征仍需人工检查最大激活样本才能理解。Anthropic 的自然语言自编码器（Natural Language Autoencoders，NLAE） <a class="book-citation" id="cite-anthropic2026nla-1" href="../../references.md#ref-anthropic2026nla" aria-label="参考文献 12">[12]</a> 采取了根本不同的方法：用<em>自然语言描述</em>替代稀疏瓶颈，使可解释性自动化。</p>

##### NLAE 如何工作。 {#how-nlaes-work.}

<ol>
<li><p><strong>编码器</strong>：一个语言模型读取隐藏激活（或输入文本）并产生关于活跃概念的自然语言描述：例如，“文本讨论法国菜并使用正式的学术语调。”</p></li>
<li><p><strong>解码器</strong>：第二个语言模型读取自然语言描述并重建原始激活（或预测下一个 Token）。</p></li>
<li><p><strong>训练</strong>：编码器和解码器端到端地训练以最小化重建损失，瓶颈是一个可变长度的自然语言字符串而非稀疏向量。</p></li>
</ol>

##### 相对 SAE 的优势。 {#advantages-over-saes.}

<ul>
<li><p><strong>自解释</strong>：特征<em>字面上</em>就是自然语言——无需人工标注。</p></li>
<li><p><strong>可组合</strong>：可以表达 SAE 特征无法作为单一方向表示的复杂关系概念（“对事实声明的讽刺回应”）。</p></li>
<li><p><strong>层次化</strong>：描述可以在同一表征中同时捕捉细粒度（词级）和粗粒度（文档级）属性。</p></li>
<li><p><strong>可审计</strong>：瓶颈描述是人类可读的，能够直接检查模型“认为”存在的信息。</p></li>
</ul>

##### 局限性。 {#limitations.}

<p>NLAE 引入了一个“语言模型在回路中”的设计，使其计算昂贵，并且可能与任何模型生成的解释一样面临忠实性担忧。它们也不能轻易表示亚符号特征（几何模式、精确数值），而 SAE 能将这些自然地作为激活幅度处理。</p>

<div class="admonition tip"><p class="admonition-title">可解释性层级栈</p>
<p>将可解释性工具视为一个层级：</p>

<ol>
<li><p><strong>注意力图</strong>：“模型在看什么？”（最便宜，最不忠实）</p></li>
<li><p><strong>探测分类器</strong>：“这一层编码了什么信息？”</p></li>
<li><p><strong>稀疏自编码器</strong>：“哪些单义特征是活跃的？”（可扩展，需要人工标注）</p></li>
<li><p><strong>自然语言自编码器</strong>：“模型认为发生了什么？”（自解释，昂贵）</p></li>
<li><p><strong>因果追踪 / 修补（Causal tracing / patching）</strong>：“哪些组件实际导致了这个输出？”（最忠实，最昂贵）</p></li>
</ol>

<p>每个层级都在解释的成本、可扩展性和忠实性之间权衡。</p>
</div>

## 1.4　预测头：Transformer 的输出 {#prediction-heads-what-transformers-output}

<p>Transformer 主干网络为每个位置产生上下文隐藏状态 <span class="arithmatex">\(\mathbf{h}_t \in \mathbb{R}^d\)</span>。我们<em>如何处理</em>这些隐藏状态——即<strong>预测头（Prediction Head）</strong>——定义了任务本身。同一个 Transformer 主干网络只需更换预测头，就能服务于完全不同的目的。</p>

<figure id="fig:prediction-heads"><a href="../../assets/figures/fig_008_prediction-heads.png" target="_blank" rel="noopener" aria-label="查看图 1.7 原图" title="点击查看原图"><img src="../../assets/figures/fig_008_prediction-heads.png" alt="同一个 Transformer 主干网络通过更换预测头即可支持不同任务。本文使用的全部三种预测头在最终投影层之下具有完全相同的架构。" loading="lazy"></a><figcaption><span class="figure-number">图 1.7　</span>同一个 Transformer 主干网络通过更换预测头即可支持不同任务。本文使用的全部三种预测头在最终投影层之下具有完全相同的架构。</figcaption></figure>

### 1.4.1　语言建模头（预训练阶段） {#language-modeling-head-pretraining}

<p>标准的语言建模头（Language Modeling Head, LM Head）将最终隐藏状态投影到词表 logits，并以下一个 token 的交叉熵损失进行训练：</p>



<div class="arithmatex">\[P(x_{t+1} | x_{\leq t}) = \text{softmax}(\mathbf{W}_{\text{head}} \cdot \mathbf{h}_t + \mathbf{b})
\tag{1.7}\]</div>

<p>其中 <span class="arithmatex">\(\mathbf{W}_{\text{head}} \in \mathbb{R}^{|\mathcal{V}| \times d}\)</span>（通常与 embedding 矩阵权重共享：<span class="arithmatex">\(\mathbf{W}_{\text{head}} = \mathbf{E}^T\)</span>）。</p>

<div class="admonition note"><p class="admonition-title">LM 头的特性</p>
<ul>
<li><p><strong>训练目标</strong>：因果语言建模（对每个位置预测下一个 token）</p></li>
<li><p><strong>损失</strong>：<span class="arithmatex">\(\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{&lt;t})\)</span></p></li>
<li><p><strong>标签</strong>：每个 token 同时作为输入（右移一位）和目标（左移一位）</p></li>
<li><p><strong>使用阶段</strong>：在大规模语料上的预训练（万亿级 token）</p></li>
<li><p><strong>关键洞察</strong>：模型在下一 token 预测的副产品中学到通用语言理解能力</p></li>
</ul>
</div>

### 1.4.2　条件生成头（SFT / 指令跟随） {#conditional-generation-head-sft-instruction-following}

<p>对于监督微调（Supervised Fine-Tuning, SFT），其架构与 LM 头<em>完全相同</em>——都是将隐藏状态线性投影到词表 logits。区别仅在于<em>在哪些 token 上计算损失</em>：</p>



<div class="arithmatex">\[\mathcal{L}_{\text{SFT}} = -\frac{1}{|y|}\sum_{t=1}^{|y|} \log P(y_t | x_{\text{prompt}}, y_{&lt;t})
\tag{1.8}\]</div>

<div class="admonition note"><p class="admonition-title">条件生成头 – 与 LM 头的关键区别</p>
<ul>
<li><p><strong>损失掩码</strong>：只在<em>回复</em>的 token 上计算损失，不计算 prompt/指令部分。Prompt 仅提供上下文，不提供梯度信号。</p></li>
<li><p><strong>条件化</strong>：模型学会在特定指令格式（system prompt、用户查询、工具调用）的<em>条件下</em>生成回复。</p></li>
<li><p><strong>格式 token</strong>：特殊 token（<code>&lt;|user|&gt;</code>、<code>&lt;|assistant|&gt;</code>）引导模型产生结构化输出。</p></li>
<li><p><strong>使用阶段</strong>：在精心策划的指令-回复对上进行 SFT；也在强化学习的策略生成阶段使用（即产出动作/回复的策略头）。</p></li>
</ul>
</div>

<div class="admonition tip"><p class="admonition-title">同一个头 – 不同的训练信号</p>
<p>LM 头和 SFT 头在架构上完全相同（同一个 <span class="arithmatex">\(\mathbf{W}_{\text{head}}\)</span>）。唯一的区别是 SFT 阶段会掩码掉 prompt token 的损失。这个细微的改动就把一个通用文本预测器转变为一个指令跟随助手。预测头学会根据上下文条件「激活」不同的生成模式。</p>
</div>

### 1.4.3　价值头（用于 RL 的回归头） {#value-head-regression-for-rl}

<p>在强化学习（PPO、GRPO）中，我们需要估计某个状态<em>有多好</em>——这需要一个标量输出，而不是词表 logits。<strong>价值头（Value Head）</strong>用一个简单的回归层替换了 LM 投影：</p>



<div class="arithmatex">\[V(s_t) = \mathbf{w}_{\text{value}}^T \cdot \mathbf{h}_t + b \in \mathbb{R}
\tag{1.9}\]</div>

<p>其中 <span class="arithmatex">\(\mathbf{w}_{\text{value}} \in \mathbb{R}^d\)</span>，<span class="arithmatex">\(b \in \mathbb{R}\)</span>。</p>

<div class="admonition note"><p class="admonition-title">价值头的特性</p>
<ul>
<li><p><strong>输出</strong>：单个标量（从该状态出发的期望累计奖励）</p></li>
<li><p><strong>损失</strong>：预测回报与真实回报之间的均方误差：<span class="arithmatex">\(\mathcal{L}_V = \frac{1}{T}\sum_t (V(s_t) - R_t)^2\)</span></p></li>
<li><p><strong>架构</strong>：线性层 <span class="arithmatex">\(\mathbb{R}^d \to \mathbb{R}^1\)</span>（有时会用一个小型 MLP：<span class="arithmatex">\(d \to 256 \to 1\)</span>）</p></li>
<li><p><strong>主干共享</strong>：通常与策略共享 Transformer 主干（但有独立的价值头），也可能使用完全独立的 critic 网络</p></li>
<li><p><strong>使用阶段</strong>：PPO 优势估计（GAE）、奖励模型评分</p></li>
</ul>
</div>

### 1.4.4　预测头选择总览 {#head-selection-summary}

<p class="table-caption" id="table-1-9"><span class="table-number">表 1.9　</span>本文使用的各类预测头及其训练场景。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>预测头</strong></th><th scope="col"><strong>输出</strong></th><th scope="col"><strong>损失</strong></th><th scope="col"><strong>阶段</strong></th><th scope="col"><strong>用途</strong></th></tr></thead>
<tbody><tr><td>LM Head</td><td><span class="arithmatex">\(\mathbb{R}^{|\mathcal{V}|}\)</span></td><td>交叉熵（所有 token）</td><td>预训练</td><td>从原始文本学习语言</td></tr>
<tr><td>Conditional Head</td><td><span class="arithmatex">\(\mathbb{R}^{|\mathcal{V}|}\)</span></td><td>交叉熵（仅回复部分）</td><td>SFT</td><td>学习跟随指令</td></tr>
<tr><td>Value Head</td><td><span class="arithmatex">\(\mathbb{R}^1\)</span></td><td>MSE</td><td>RL (PPO)</td><td>估计状态价值以计算优势</td></tr>
<tr><td>Reward Head</td><td><span class="arithmatex">\(\mathbb{R}^1\)</span></td><td>成对排序</td><td>奖励模型训练</td><td>给回复质量打分</td></tr></tbody>
</table>
</div>

<div class="admonition warning"><p class="admonition-title">预测头初始化很重要</p>
<p>为预训练 LM 添加价值头时，应将其初始化为接近零的小随机权重。若初始化值过大，初始价值估计会严重偏离，导致优势值过大、PPO 更新不稳定。常见做法：将最后一层线性层用 <span class="arithmatex">\(\mathcal{N}(0, 1/\sqrt{d})\)</span> 初始化，或直接置零。</p>
</div>

### 1.4.5　HuggingFace 实现 {#huggingface-implementation}

<p class="code-caption"><span class="code-number">代码 1.3　</span>使用 HuggingFace 加载与使用不同的预测头。</p>
<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">transformers</span><span class="w"> </span><span class="kn">import</span> <span class="p">(</span>
    <span class="n">AutoModelForCausalLM</span><span class="p">,</span>          <span class="c1"># LM 头（预训练 + SFT）</span>
    <span class="n">AutoModelForSequenceClassification</span><span class="p">,</span>  <span class="c1"># 奖励头</span>
    <span class="n">AutoTokenizer</span><span class="p">,</span>
<span class="p">)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">AutoModelForCausalLMWithValueHead</span>  <span class="c1"># 价值头（PPO）</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">torch</span>

<span class="n">model_name</span> <span class="o">=</span> <span class="s2">"meta-llama/Llama-3.1-8B-Instruct"</span>
<span class="n">tokenizer</span> <span class="o">=</span> <span class="n">AutoTokenizer</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span><span class="n">model_name</span><span class="p">)</span>

<span class="c1"># === 1. LM 头（预训练 / SFT）===</span>
<span class="c1"># 默认的 CausalLM 模型——将隐藏状态投影到词表 logits</span>
<span class="n">lm_model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLM</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="n">model_name</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
    <span class="n">device_map</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
<span class="p">)</span>
<span class="c1"># lm_model.lm_head: Linear(hidden_size -&gt; vocab_size)</span>
<span class="c1"># 输出：形状为 (batch, seq_len, vocab_size) 的 logits</span>

<span class="n">inputs</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span><span class="s2">"The capital of France is"</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span>
<span class="n">outputs</span> <span class="o">=</span> <span class="n">lm_model</span><span class="p">(</span><span class="o">**</span><span class="n">inputs</span><span class="p">)</span>
<span class="n">next_token_logits</span> <span class="o">=</span> <span class="n">outputs</span><span class="o">.</span><span class="n">logits</span><span class="p">[:,</span> <span class="o">-</span><span class="mi">1</span><span class="p">,</span> <span class="p">:]</span>  <span class="c1"># (batch, vocab_size)</span>
<span class="n">probs</span> <span class="o">=</span> <span class="n">torch</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">next_token_logits</span><span class="p">,</span> <span class="n">dim</span><span class="o">=-</span><span class="mi">1</span><span class="p">)</span>

<span class="c1"># === 2. 条件生成头（SFT）===</span>
<span class="c1"># 架构上与 LM 头完全相同——区别在于损失掩码</span>
<span class="c1"># SFT 时只在回复 token 上计算损失：</span>
<span class="n">messages</span> <span class="o">=</span> <span class="p">[</span>
    <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"user"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"What is 2+2?"</span><span class="p">},</span>
    <span class="p">{</span><span class="s2">"role"</span><span class="p">:</span> <span class="s2">"assistant"</span><span class="p">,</span> <span class="s2">"content"</span><span class="p">:</span> <span class="s2">"4"</span><span class="p">},</span>
<span class="p">]</span>
<span class="n">formatted</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="o">.</span><span class="n">apply_chat_template</span><span class="p">(</span><span class="n">messages</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span>
<span class="n">labels</span> <span class="o">=</span> <span class="n">formatted</span><span class="o">.</span><span class="n">clone</span><span class="p">()</span>
<span class="c1"># 掩码掉 prompt token（设为 -100，交叉熵会忽略它们）</span>
<span class="n">prompt_len</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">tokenizer</span><span class="o">.</span><span class="n">apply_chat_template</span><span class="p">(</span><span class="n">messages</span><span class="p">[:</span><span class="mi">1</span><span class="p">]))</span>
<span class="n">labels</span><span class="p">[:,</span> <span class="p">:</span><span class="n">prompt_len</span><span class="p">]</span> <span class="o">=</span> <span class="o">-</span><span class="mi">100</span>
<span class="n">loss</span> <span class="o">=</span> <span class="n">lm_model</span><span class="p">(</span><span class="n">input_ids</span><span class="o">=</span><span class="n">formatted</span><span class="p">,</span> <span class="n">labels</span><span class="o">=</span><span class="n">labels</span><span class="p">)</span><span class="o">.</span><span class="n">loss</span>

<span class="c1"># === 3. 价值头（PPO Critic）===</span>
<span class="c1"># 在 LM 主干之上加一个 Linear(hidden_size -&gt; 1)</span>
<span class="n">value_model</span> <span class="o">=</span> <span class="n">AutoModelForCausalLMWithValueHead</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="n">model_name</span><span class="p">,</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
    <span class="n">device_map</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
<span class="p">)</span>
<span class="c1"># value_model.v_head: Linear(hidden_size -&gt; 1)</span>
<span class="c1"># 同时返回 LM logits 和每个 token 的价值估计</span>

<span class="n">inputs</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span><span class="s2">"Explain quantum computing"</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span>
<span class="n">lm_logits</span><span class="p">,</span> <span class="n">loss</span><span class="p">,</span> <span class="n">values</span> <span class="o">=</span> <span class="n">value_model</span><span class="p">(</span>
    <span class="o">**</span><span class="n">inputs</span><span class="p">,</span> <span class="n">return_dict</span><span class="o">=</span><span class="kc">False</span>
<span class="p">)</span>
<span class="c1"># values 形状：(batch, seq_len, 1)——每个 token 一个标量估计</span>

<span class="c1"># === 4. 奖励头（奖励模型）===</span>
<span class="c1"># 分类头：在最后一个 token 上接 Linear(hidden_size -&gt; 1)</span>
<span class="n">reward_model</span> <span class="o">=</span> <span class="n">AutoModelForSequenceClassification</span><span class="o">.</span><span class="n">from_pretrained</span><span class="p">(</span>
    <span class="n">model_name</span><span class="p">,</span>
    <span class="n">num_labels</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>              <span class="c1"># 单标量输出</span>
    <span class="n">torch_dtype</span><span class="o">=</span><span class="n">torch</span><span class="o">.</span><span class="n">bfloat16</span><span class="p">,</span>
    <span class="n">device_map</span><span class="o">=</span><span class="s2">"auto"</span><span class="p">,</span>
<span class="p">)</span>
<span class="c1"># 通过池化最后一个 token 的隐藏状态对整个序列打分</span>
<span class="n">inputs</span> <span class="o">=</span> <span class="n">tokenizer</span><span class="p">(</span><span class="s2">"Good response here"</span><span class="p">,</span> <span class="n">return_tensors</span><span class="o">=</span><span class="s2">"pt"</span><span class="p">)</span>
<span class="n">reward_score</span> <span class="o">=</span> <span class="n">reward_model</span><span class="p">(</span><span class="o">**</span><span class="n">inputs</span><span class="p">)</span><span class="o">.</span><span class="n">logits</span>  <span class="c1"># 形状：(batch, 1)</span>
</code></pre></div>

<div class="admonition tip"><p class="admonition-title">权重共享：LM 头 = embedding 矩阵的转置</p>
<p>大多数现代 LLM 会将 LM 头的权重与输入 embedding 矩阵<em>共享（tie）</em>：<code>lm_head.weight = model.embed_tokens.weight</code>。这意味着 LM 头<em>并非</em>独立学习的一层——它直接复用了 embedding 表。好处包括：参数更少（节省 <span class="arithmatex">\(|\mathcal{V}| \times d\)</span>）、泛化更好，并且 embedding 空间的几何结构直接决定了 token 概率。你可以在 HuggingFace 中验证这一点：对大多数模型而言，<code>model.lm_head.weight is model.model.embed_tokens.weight</code> 会返回 <code>True</code>。</p>
</div>
