# 1.2　分词（Tokenization） {#sec:tokenization}

<p>分词（Tokenization）是将原始文本转换为语言模型所操作的离散符号的关键第一步。分词器的选择直接影响模型质量、多语种能力和计算效率。</p>

<div class="admonition tip"><p class="admonition-title">为什么用子词？</p>
<p>字符级模型需要很长的序列（注意力代价高昂）。词级模型无法处理稀有词或新词。子词分词达到了理想的平衡：常见词作为单个 Token（“the” <span class="arithmatex">\(\to\)</span> [the]），稀有词被分解为已知片段（“cryptocurrency” <span class="arithmatex">\(\to\)</span> [“crypt”, “ocur”, “rency”]），同时词表保持可控规模（32K–128K Token）。</p>
</div>

## 1.2.1　为什么不用字符或词？ {#why-not-characters-or-words}

<p class="table-caption" id="table-1-1"><span class="table-number">表 1.1　</span>不同分词粒度的权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>粒度</strong></th><th scope="col"><strong>词表大小</strong></th><th scope="col"><strong>序列长度</strong></th><th scope="col"><strong>问题</strong></th></tr></thead>
<tbody><tr><td>字符</td><td><span class="arithmatex">\(\sim\)</span>256</td><td>极长</td><td>注意力开销 <span class="arithmatex">\(O(n^2)\)</span>；难以学习长程语义</td></tr>
<tr><td>词</td><td><span class="arithmatex">\(\sim\)</span>500K+</td><td>短</td><td>无法处理稀有/新词；嵌入表巨大</td></tr>
<tr><td>子词</td><td>32K–128K</td><td>适中</td><td>最佳权衡：短序列、开放词表</td></tr></tbody>
</table>
</div>

## 1.2.2　字节对编码（Byte-Pair Encoding，BPE） {#byte-pair-encoding-bpe}

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

## 1.2.3　其他分词方法 {#other-tokenization-methods}

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

## 1.2.4　分词最佳实践 {#tokenization-best-practices}

<ol>
<li><p><strong>词表大小很重要</strong>：32K 是最低限度；128K 能提供更好的多语种覆盖和代码处理能力。Llama-3 使用 128K Token。</p></li>
<li><p><strong>特殊 Token</strong>：始终包含 <code>&lt;bos&gt;</code>、<code>&lt;eos&gt;</code>、<code>&lt;pad&gt;</code>、<code>&lt;unk&gt;</code>。对于指令微调模型，需添加角色标记（<code>&lt;|user|&gt;</code>、<code>&lt;|assistant|&gt;</code>）。</p></li>
<li><p><strong>繁殖度（Fertility）</strong>：度量各语种的每词 Token 数。高繁殖度（每词产生很多 Token）表示对该语种的覆盖较差。</p></li>
<li><p><strong>切勿跨边界分词</strong>：空格、标点和数字应一致处理。大多数现代分词器会在前面添加空格标记（如 “the”）以区分词首 Token 与词中续接 Token。</p></li>
<li><p><strong>数字</strong>：对算术任务考虑数字级分词。“2024” 切为 [“2”,“0”,“2”,“4”] 可以支持逐位推理。</p></li>
<li><p><strong>代码</strong>：确保高效地分词空白（缩进）。Llama-3 将连续空格分词为单个 Token。</p></li>
</ol>

## 1.2.5　分词实战：HuggingFace 示例 {#tokenization-in-practice-huggingface-example}

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

## 1.2.6　特殊 Token 与结构化提示 {#special-tokens-and-structured-prompts}

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

#### 指令微调模型的角色标记。 {#role-markers-for-instruction-tuned-models.}

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
