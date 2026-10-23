# 7.1　动机 {#motivation-dup2}

<p>PPO 的 value model（critic）在语言任务上存在三大问题：</p>

<ol>
<li><p><strong>显存</strong>：value head 与 policy 共享主干（70B 模型占 140GB）。若分离则显存翻倍。</p></li>
<li><p><strong>精度</strong>：对部分序列预测期望 reward 极其困难。value function 经常出错 <span class="arithmatex">\(\rightarrow\)</span> advantage 出错 <span class="arithmatex">\(\rightarrow\)</span> gradient 方向出错。</p></li>
<li><p><strong>训练</strong>：value head 需要大量样本才能收敛。RL 早期它给出嘈杂的预测，破坏 policy 学习的稳定性。</p></li>
</ol>

<p><strong>GRPO 的关键洞见</strong> <a class="book-citation" id="cite-shao2024deepseekmath-1" href="../../references.md#ref-shao2024deepseekmath" aria-label="参考文献 323">[323]</a>：与其学习 <span class="arithmatex">\(V(s)\)</span>，不如从一组样本中经验地 <em>估计</em> 它。对同一 prompt 生成 <span class="arithmatex">\(G\)</span> 条响应，计算它们的 reward，并用组统计量作为 baseline。</p>
