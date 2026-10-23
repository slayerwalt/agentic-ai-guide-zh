# 5.1　动机与历史 {#motivation-and-history}

<p><strong>问题</strong>：朴素的 policy gradient 更新对步长没有任何约束。仅仅一次倒霉的 batch 就可能把 policy 推入一个生成垃圾文本的区域 <span class="arithmatex">\(\rightarrow\)</span> 垃圾文本获得低 reward <span class="arithmatex">\(\rightarrow\)</span> 下一次 gradient 让情况更糟 <span class="arithmatex">\(\rightarrow\)</span> 不可挽回地崩溃。</p>

<p><strong>解决方案的演进</strong>：</p>

<ol>
<li><p><strong>TRPO</strong> <a class="book-citation" id="cite-schulman2015trust-1" href="../../references.md#ref-schulman2015trust" aria-label="参考文献 317">[317]</a>（2015）：对新旧 policy 之间的 KL 散度施加约束。效果完美，但需要昂贵的二阶优化（Fisher 信息矩阵、共轭梯度）。</p></li>
<li><p><strong>PPO</strong>（2017） <a class="book-citation" id="cite-schulman2017proximal-1" href="../../references.md#ref-schulman2017proximal" aria-label="参考文献 319">[319]</a>：用一个简单的一阶 clipped 目标实现类似的稳定性。实现复杂度低了 10<span class="arithmatex">\(\times\)</span>，效果几乎一样，且能轻松扩展到分布式训练。</p></li>
</ol>
