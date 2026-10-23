# 第 8 章 · 偏好优化变体 {#ch:po-variants}

<p>本章涵盖一类用不同目标、数据假设或架构权衡来扩展或替换 DPO 的方法。每种方法都解决标准 offline DPO 的某个具体局限：分布偏移（Online DPO）、需要配对数据（KTO）、对噪声标签过拟合（IPO）、参考模型显存开销（ORPO）或训练复杂度（Best-of-N）。</p>

## 本章内容

<ul class="chapter-outline">
<li id="sec:online-dpo"><a href="section-01.md">8.1　Online DPO</a></li>
<li id="sec:kto"><a href="section-02.md">8.2　KTO —— Kahneman-Tversky Optimization</a></li>
<li id="ipo-identity-preference-optimization"><a href="section-03.md">8.3　IPO —— Identity Preference Optimization</a></li>
<li id="sec:orpo"><a href="section-04.md">8.4　ORPO —— Odds Ratio Preference Optimization</a></li>
<li id="best-of-n-sampling-rejection-sampling"><a href="section-05.md">8.5　Best-of-N 采样（拒绝采样）</a></li>
<li id="sec:method-comparison"><a href="section-06.md">8.6　总结：如何选择对齐方法</a></li>
</ul>
