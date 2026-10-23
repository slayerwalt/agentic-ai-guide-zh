# 28.10　系统架构扩展题 {#system-architecture-extension-questions}

<section class="quiz-question" id="q-44"><h2 class="question-title">Q44：投机解码（Speculative Decoding）如何工作？它在 RLHF 中何时有帮助？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>问题</strong>：大模型每次前向只产出一个 Token（70B 约 <span class="arithmatex">\(\sim\)</span>70ms）。慢。</p>

<p><strong>投机解码</strong>：</p>

<ol>
<li><p><strong>草稿</strong>：小模型（1–7B）快速生成 <span class="arithmatex">\(k\)</span> 个候选 Token（全部 <span class="arithmatex">\(k\)</span> 个约 <span class="arithmatex">\(\sim\)</span>5ms）。</p></li>
<li><p><strong>验证</strong>：大模型做一次前向，并行给所有 <span class="arithmatex">\(k\)</span> 个 Token 打分。<span class="arithmatex">\(p_\text{large}(t_i) \geq p_\text{draft}(t_i)\)</span> 的 Token 总是接受。其他按概率接受。</p></li>
<li><p><strong>结果</strong>：每次验证步平均接受 3–4 个 Token。加速：2–3<span class="arithmatex">\(\times\)</span>。</p></li>
</ol>

<p><strong>关键性质</strong>：输出分布与单独从大模型采样<em>完全相同</em>。无质量损失。草稿模型只影响速度，不影响输出。</p>

<p><strong>对 RLHF 而言</strong>：生成占算力 60%。生成加速 2–3<span class="arithmatex">\(\times\)</span> = 端到端加速 1.5–2<span class="arithmatex">\(\times\)</span>。结合 vLLM + INT8：生成从瓶颈变为与训练持平。</p>

<p><strong>局限</strong>：(1) 草稿模型必须共享 tokenizer。(2) 高温下效果较差（草稿模型准确率下降）。(3) 草稿模型需要额外 GPU 显存。(4) 超过 <span class="arithmatex">\(k=5\)</span> 后边际收益递减（接受率下降）。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（系统基础；系统架构）。</em></p>
</details></section>

<section class="quiz-question" id="q-45"><h2 class="question-title">Q45：解释 roofline 模型。如何判断一个 kernel 是计算受限还是内存受限？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：roofline 模型将可达性能（FLOPS）刻画为算术强度（每字节内存流量的 FLOPS）的函数。</p>

<p><strong>两个区域</strong>：</p>

<ul>
<li><p><strong>内存受限</strong>（拐点左侧）：性能受限于向计算单元喂数据的速度。实际 FLOPS = 带宽 <span class="arithmatex">\(\times\)</span> 算术强度。GPU 利用率 <span class="arithmatex">\(&lt;\)</span> 100%。</p></li>
<li><p><strong>计算受限</strong>（拐点右侧）：性能受限于峰值 FLOPS。内存够快。GPU 达到最大利用率。</p></li>
</ul>

<p><strong>拐点</strong>：峰值 FLOPS / 峰值带宽。A100：312 TF / 2 TB/s = 156 FLOP/byte。</p>

<p><strong>LLM 操作</strong>：</p>

<ul>
<li><p><strong>自回归生成</strong>（batch=1）：读取 140GB 权重，做 140G FLOPs = 1 FLOP/byte。<em>极度</em>内存受限（比拐点低 156<span class="arithmatex">\(\times\)</span>）。GPU 利用率仅 0.6%。</p></li>
<li><p><strong>训练前向</strong>（batch=128，seq=2048）：算术强度 <span class="arithmatex">\(\approx 200\)</span>+ FLOP/byte。计算受限。接近峰值利用率。</p></li>
<li><p><strong>Attention</strong>（长序列）：<span class="arithmatex">\(O(n^2 d)\)</span> FLOPs / <span class="arithmatex">\(O(n^2 + nd)\)</span> 字节。<span class="arithmatex">\(n\)</span> 长：计算受限；<span class="arithmatex">\(n\)</span> 短：内存受限。FlashAttention 无论如何都让其留在 SRAM。</p></li>
</ul>

<p><strong>实用法</strong>：若 kernel 内存受限，减少内存流量（量化、缓存、分块）。若计算受限，减少 FLOPs（剪枝、蒸馏、降低精度）。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 章（LLM 的系统基础）。</em></p>
</details></section>

<section class="quiz-question" id="q-46"><h2 class="question-title">Q46：连续批处理（Continuous Batching）如何工作？为何它对 RLHF 生成至关重要？</h2>
<details class="question"><summary>展开解答</summary>
<p><strong>答</strong>：<strong>静态批处理</strong>：启动 <span class="arithmatex">\(B\)</span> 条序列。等<em>所有</em>序列结束。若一条生成 500 Token、另一条只生成 50 Token，那 50 Token 序列的 GPU 槽位会闲置 450 Token 的时间。</p>

<p><strong>连续批处理</strong>（迭代级调度）：每个生成步后检查哪些序列已完成。立即在腾出的槽位插入新序列。GPU 槽位从不闲置。</p>

<p><strong>为何对 RLHF 至关重要</strong>：</p>

<ol>
<li><p>RLHF 生成多样输出（高温度）。长度方差巨大——有些回答 50 Token，有些 2000+。</p></li>
<li><p>无连续批处理：平均利用率 <span class="arithmatex">\(\sim\)</span>40–50%（等最慢的序列）。</p></li>
<li><p>有连续批处理：利用率 <span class="arithmatex">\(&gt;\)</span>90%。吞吐高 2–3<span class="arithmatex">\(\times\)</span>。</p></li>
<li><p>RLHF 需要大 batch（每步 128+ 回答）。用静态批处理生成 128 个回答需要 max_tokens <span class="arithmatex">\(\times\)</span> 128 个串行步。连续批处理摊销了这一开销。</p></li>
</ol>

<p><strong>实现</strong>：vLLM 的调度器在每个解码步后检查。抢占：若有新的高优先级请求到达且显存已满，可将低优先级序列的 KV cache 换出到 CPU，稍后恢复。</p>

<p><em><strong>复习：</strong> 第 <a href="../02-systems-foundations/index.md">2</a> 与 <a href="../11-training-infrastructure/index.md">11</a> 章（系统基础；系统架构）。</em></p>
</details></section>
