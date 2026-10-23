# 29.11　上下文窗口预算 {#context-window-budget}



<div class="arithmatex">\[C \geq \underbrace{S}_{\text{system}} + \underbrace{M}_{\text{memory/RAG}} + \underbrace{T}_{\text{tool defs}} + \underbrace{H}_{\text{history}} + \underbrace{R}_{\text{reserved output}}
\tag{29.22}\]</div>

<p><strong>经验法则</strong>（针对 128K 上下文）：</p>

<ul>
<li><p>系统提示：1–4K token（固定）</p></li>
<li><p>工具定义：2–8K（随工具数量增长）</p></li>
<li><p>RAG 上下文：4–16K（top-<span class="arithmatex">\(k\)</span> 个 chunk）</p></li>
<li><p>历史记录：无界增长 <span class="arithmatex">\(\rightarrow\)</span> 汇总或截断</p></li>
<li><p>预留输出：2–8K</p></li>
</ul>
