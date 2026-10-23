# 8.2　KTO —— Kahneman-Tversky Optimization {#sec:kto}

## 8.2.1　动机 {#motivation-1}

<p>DPO 需要 <em>成对</em> 偏好：对同一 prompt，你既需要好的也需要坏的响应。实际上大多数反馈是 <em>未配对</em> 的：用户对单个响应给出点赞/点踩，没有匹配的 pair。</p>

<p><strong>KTO 的洞见</strong> <a class="book-citation" id="cite-ethayarajh2024kto-1" href="../../references.md#ref-ethayarajh2024kto" aria-label="参考文献 87">[87]</a>：使用前景理论（来自行为经济学）。人类对损失的感受比对收益更强烈。“点踩”应当产生比“点赞”更强的 gradient。</p>

## 8.2.2　Loss 函数 {#loss-function}



<div class="arithmatex">\[\boxed{\mathcal{L}_\text{KTO} = \mathbb{E}_{y_w}\left[\lambda_w (1 - v(x, y_w))\right] + \mathbb{E}_{y_l}\left[\lambda_l \cdot v(x, y_l)\right]}
\tag{8.1}\]</div>

<p>其中 <span class="arithmatex">\(v(x,y) = \sigma\left(\beta \log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)} - z_\text{ref}\right)\)</span>，<span class="arithmatex">\(z_\text{ref}\)</span> 是期望 KL 散度（一个滑动 baseline）。</p>

<div class="admonition tip"><p class="admonition-title">基于前景理论的 KTO 直觉</p>
<p><strong>好响应</strong>（<span class="arithmatex">\(y_w\)</span>）：模型通过提高它们的概率获得“效用”。但收益递减——一旦它已相当可能，就不再用力推。</p>

<p><strong>坏响应</strong>（<span class="arithmatex">\(y_l\)</span>）：损失厌恶意味着生成坏文本的惩罚被加权得比生成好文本的奖励更强。默认 <span class="arithmatex">\(\lambda_l = 1.0\)</span>，<span class="arithmatex">\(\lambda_w = 1.0\)</span>，但你可以设 <span class="arithmatex">\(\lambda_l &gt; \lambda_w\)</span>。</p>

<p><strong>关键优势</strong>：每个训练样本都是独立的！无需匹配 pair。可直接使用点赞/点踩数据。</p>
</div>

<div class="admonition example"><p class="admonition-title">KTO 数据格式</p>
<p>与 DPO 需要 <code>{&quot;prompt&quot;: ..., &quot;chosen&quot;: ..., &quot;rejected&quot;: ...}</code> 不同</p>

<p>KTO 只需要：<code>{&quot;prompt&quot;: ..., &quot;completion&quot;: ..., &quot;label&quot;: true/false}</code></p>

<p>这意味着你可以使用：</p>

<ul>
<li><p>生产流量中的点赞/点踩</p></li>
<li><p>论坛上的赞/踩</p></li>
<li><p>二值化的人类评分（4–5 星 = 好，1–2 星 = 坏）</p></li>
<li><p>任何单条响应的质量信号</p></li>
</ul>
</div>

## 8.2.3　TRL 实现 {#trl-implementation-1}

<p>下面是基于 HuggingFace TRL 的最小可运行示例。</p>

<div class="highlight"><pre><span></span><code><span class="kn">from</span><span class="w"> </span><span class="nn">trl</span><span class="w"> </span><span class="kn">import</span> <span class="n">KTOConfig</span><span class="p">,</span> <span class="n">KTOTrainer</span>

<span class="c1"># 数据格式：{"prompt": str, "completion": str, "label": bool}</span>
<span class="c1"># label=True 表示好的，label=False 表示坏的</span>
<span class="n">kto_dataset</span> <span class="o">=</span> <span class="p">[</span>
    <span class="p">{</span><span class="s2">"prompt"</span><span class="p">:</span> <span class="s2">"What's 2+2?"</span><span class="p">,</span> <span class="s2">"completion"</span><span class="p">:</span> <span class="s2">"The answer is 4."</span><span class="p">,</span> <span class="s2">"label"</span><span class="p">:</span> <span class="kc">True</span><span class="p">},</span>
    <span class="p">{</span><span class="s2">"prompt"</span><span class="p">:</span> <span class="s2">"What's 2+2?"</span><span class="p">,</span> <span class="s2">"completion"</span><span class="p">:</span> <span class="s2">"It might be 5."</span><span class="p">,</span> <span class="s2">"label"</span><span class="p">:</span> <span class="kc">False</span><span class="p">},</span>
<span class="p">]</span>

<span class="n">kto_config</span> <span class="o">=</span> <span class="n">KTOConfig</span><span class="p">(</span>
    <span class="n">output_dir</span><span class="o">=</span><span class="s2">"./kto_output"</span><span class="p">,</span>
    <span class="n">beta</span><span class="o">=</span><span class="mf">0.1</span><span class="p">,</span>
    <span class="n">desirable_weight</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>        <span class="c1"># 好样本的权重</span>
    <span class="n">undesirable_weight</span><span class="o">=</span><span class="mf">1.0</span><span class="p">,</span>      <span class="c1"># 坏样本的权重（损失厌恶时调高）</span>
    <span class="n">learning_rate</span><span class="o">=</span><span class="mf">5e-7</span><span class="p">,</span>
    <span class="n">max_length</span><span class="o">=</span><span class="mi">2048</span><span class="p">,</span>
    <span class="n">per_device_train_batch_size</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">gradient_accumulation_steps</span><span class="o">=</span><span class="mi">4</span><span class="p">,</span>
    <span class="n">num_train_epochs</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span>
    <span class="n">bf16</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
<span class="p">)</span>

<span class="n">trainer</span> <span class="o">=</span> <span class="n">KTOTrainer</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="n">model</span><span class="p">,</span>
    <span class="n">ref_model</span><span class="o">=</span><span class="n">ref_model</span><span class="p">,</span>  <span class="c1"># LoRA 时可为 None</span>
    <span class="n">args</span><span class="o">=</span><span class="n">kto_config</span><span class="p">,</span>
    <span class="n">train_dataset</span><span class="o">=</span><span class="n">kto_dataset</span><span class="p">,</span>
    <span class="n">tokenizer</span><span class="o">=</span><span class="n">tokenizer</span><span class="p">,</span>
<span class="p">)</span>
<span class="n">trainer</span><span class="o">.</span><span class="n">train</span><span class="p">()</span>
</code></pre></div>

## 8.2.4　何时选择 KTO {#when-to-choose-kto}

<ul>
<li><p>你有二值反馈但 <em>没有</em> 匹配 pair</p></li>
<li><p>大规模的生产点赞/点踩数据</p></li>
<li><p>一类占主导（例如 90% 好、10% 坏）—— KTO 对不平衡更友好</p></li>
<li><p>在噪声标签下快速迭代（对噪声比 DPO 更鲁棒）</p></li>
</ul>
