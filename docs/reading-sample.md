# 阅读样例 {#reading-sample}

<div class="edition-status">排版样例 · 原文节选</div>

本页摘取现有译本第 1 章与第 28 章的片段，展示正文、提示框、公式、插图、表格、代码和自测题的阅读效果。片段保留原文内容，完整章节可从[全书目录](book-outline.md)进入。

## 正文与提示框

分词（Tokenization）是将原始文本转换为语言模型所操作的离散符号的关键第一步。分词器的选择直接影响模型质量、多语种能力和计算效率。

!!! tip "为什么用子词？"

    字符级模型需要很长的序列（注意力代价高昂）。词级模型无法处理稀有词或新词。子词分词达到了理想的平衡：常见词作为单个 Token（“the” $\to$ [the]），稀有词被分解为已知片段（“cryptocurrency” $\to$ [“crypt”, “ocur”, “rency”]），同时词表保持可控规模（32K--128K Token）。

<span id="bpe-citation"></span>

字节对编码（Byte-Pair Encoding，BPE）<a class="book-citation" href="references.md#ref-sennrich2016bpe" aria-label="参考文献 320">[320]</a> 是 GPT、Llama、Mistral 以及大多数现代 LLM 采用的主流分词算法。

!!! note "BPE 算法"

    1. 从单个字符（字节）的词表开始
    2. 统计训练语料中所有相邻符号对
    3. 将出现频率最高的对合并为一个新符号
    4. 重复步骤 2–3 共 $k$ 次迭代（直到达到目标词表大小）

!!! warning "为什么仅解码器胜出"

    仅解码器模型更简单（单一模型，单一损失），扩展性更好（所有参数都参与生成），并且支持统一的训练（预训练 = 下一个 Token 预测 = 微调目标）。对于纯生成任务，编码器-解码器模型在编码器上浪费了容量。

<p class="source-note">上面分别展示原文的直觉说明、关键要点和警示框；下方的代码示例与自测解答展示另外两类内容框。</p>

## 数学公式 {#equations}

下式为第 1 章首个编号公式。网页显示原编号，链接锚点独立于显示编号；窄屏可以在公式区域内横向滚动。

<span id="eq-1-1"></span>

$$
\text{MaskedAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V
\tag{1.1}
$$

[跳转到公式（1.1）](#eq-1-1)。

## 插图与表格 {#figures-tables}

<figure id="fig-1-2">
  <img src="assets/figures/fig_003_fig3.png" alt="BPE 分词逐步合并相邻子词单元的过程" width="964" height="426" loading="lazy">
  <figcaption>图 1.2　BPE 分词示例：从字符开始，算法迭代地合并出现频率最高的相邻对，直到该词成为单个 Token 或词表预算耗尽。</figcaption>
</figure>

<p class="table-caption" id="table-1-1">表 1.1　不同分词粒度的权衡。</p>
<div class="table-scroll" role="region" aria-labelledby="table-1-1" tabindex="0">
<table>
<thead><tr><th scope="col">粒度</th><th scope="col">词表大小</th><th scope="col">序列长度</th><th scope="col">问题</th></tr></thead>
<tbody>
<tr><td>字符</td><td><span class="arithmatex">\(\sim\)256</span></td><td>极长</td><td>注意力开销 <span class="arithmatex">\(O(n^2)\)</span>；难以学习长程语义</td></tr>
<tr><td>词</td><td><span class="arithmatex">\(\sim\)500K+</span></td><td>短</td><td>无法处理稀有/新词；嵌入表巨大</td></tr>
<tr><td>子词</td><td>32K–128K</td><td>适中</td><td>最佳权衡：短序列、开放词表</td></tr>
</tbody>
</table>
</div>

## 代码与示例 {#code}

<code>transformers</code> 库为所有分词器提供了统一接口。以下演示了使用现代 LLM 分词器进行编码和解码：

~~~python title="使用 HuggingFace Transformers 进行分词的编码/解码。"
from transformers import AutoTokenizer

# 加载 Llama-3 分词器（128K 词表，字节级 BPE）
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

text = "Reinforcement learning optimizes long-term rewards."

# 编码：文本 -> Token ID
token_ids = tokenizer.encode(text)
print(token_ids)
# [128000, 29934, 262, 11008, 4815, 6900, 1317, 9860, 21845, 13]

# 将单个 Token 解码以查看子词切分
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
# ['<|begin_of_text|>', 'Re', 'inforce', 'ment', ' learning',
#  ' optimizes', ' long', '-term', ' rewards', '.']

# 解码回文本（双向往返）
reconstructed = tokenizer.decode(token_ids, skip_special_tokens=True)
assert reconstructed == text  # 完美重建

# 带注意力掩码的分词（用于带 padding 的批量输入）
batch = tokenizer(
    ["Short text.", "A much longer input sentence for comparison."],
    padding=True, return_tensors="pt"
)
print(batch.keys())  # dict_keys(['input_ids', 'attention_mask'])
~~~

!!! example "诊断训练不稳定"

    ~~~python
    # 监控以下指标以诊断优化器问题：
    # 1. 梯度范数 —— 大多数时候应 < max_grad_norm
    # 2. 损失缩放因子（FP16）—— 应稳定，不应持续减小
    # 3. 参数更新范数 —— 应远小于参数自身的范数

    import torch

    def log_optimizer_stats(model, optimizer, step):
        # 梯度范数（裁剪前）
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Adam 二阶矩统计（自适应学习率的代理指标）
        v_norms = []
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'exp_avg_sq' in state:
                    v_norms.append(state['exp_avg_sq'].mean().item())

        print(f"Step {step}: grad_norm={total_norm:.3f}, "
              f"mean_v={sum(v_norms)/len(v_norms):.6f}")

    # 红色警报：
    # grad_norm 反复 >> 1.0 —— 降低学习率或加长预热
    # grad_norm == 0.0 —— 梯度消失或 loss 写错
    # loss_scale 持续下降 —— FP16 溢出，改用 BF16
    # v 非常小 —— Adam 还没预热完，加长预热
    ~~~

## 自测题与解答 {#quiz}

### Q0a：在 decoder-only Transformer 中，attention 机制的作用是什么？为什么它是因果的？ {#q-0a}

??? question "展开解答"

    <strong>答</strong>：attention 机制允许每个 Token 关注（即对其表示做加权组合）其他 Token 的表示。在 decoder-only Transformer 中，attention 是<strong>因果的</strong>（也称为*自回归*）：Token $t$ 只能关注 Token $1, \ldots, t$，绝不能关注未来 Token $t+1, \ldots, T$。


    <strong>为什么是因果的？</strong>因为模型从左到右生成文本。推理时，未来 Token 字面意义上还不存在。训练时的因果掩码模拟了这一约束，使模型学会仅用左侧上下文预测每个 Token。在数学上，attention 矩阵被掩盖：

    $$
    \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
    $$


    其中对于 $j > i$（未来位置）有 $M_{ij} = -\infty$，强制将这些 attention 权重置零。


    <strong>实践含义</strong>：这使得推理时的 KV-cache 优化成为可能——由于过去 Token 的 keys 和 values 永远不变，可被缓存复用，将每个新 Token 的生成代价从 $O(T^2)$ 降至 $O(T)$。


    <em><strong>复习：</strong> [第 1 章（LLM 架构与优化方法）](chapters/01-llm-architecture/index.md)。</em>

## 继续阅读

[第 1 章目录](chapters/01-llm-architecture/index.md) · [第 28 章目录](chapters/28-quiz/index.md) · [全书目录](book-outline.md)
