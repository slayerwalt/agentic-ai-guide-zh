# 29.13　方法选择决策树 {#method-selection-decision-tree}

<ol>
<li><p><strong>是否有成对偏好数据（chosen + rejected）？</strong></p>

<ul>
<li><p>标签噪声大 <span class="arithmatex">\(\rightarrow\)</span> <strong>IPO</strong></p></li>
<li><p>显存受限、尚未做 SFT <span class="arithmatex">\(\rightarrow\)</span> <strong>ORPO</strong></p></li>
<li><p>数据干净、算力有限 <span class="arithmatex">\(\rightarrow\)</span> <strong>DPO</strong></p></li>
<li><p>DPO 进入瓶颈、希望加入探索 <span class="arithmatex">\(\rightarrow\)</span> <strong>Online DPO</strong></p></li>
</ul></li>
<li><p><strong>只有二值反馈（点赞 / 点踩）？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>KTO</strong></p></li>
<li><p><strong>有可验证奖励（数学 / 代码）？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>GRPO</strong></p></li>
<li><p><strong>不惜代价追求最高质量？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>PPO</strong></p></li>
<li><p><strong>想要免训练的改进？</strong> <span class="arithmatex">\(\rightarrow\)</span> <strong>Best-of-N</strong></p></li>
</ol>
