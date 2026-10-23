# 第 4 章 · 大语言模型的强化学习基础 {#rl-foundations-for-language-models}

<p>监督微调（Supervised Fine-Tuning, SFT）教模型模仿示例，但模仿存在天花板：模型永远无法超越其训练数据的质量。强化学习突破了这一壁垒。通过生成新文本、接收 reward 反馈，并朝着获得更高 reward 的行为更新，经过 RL 训练的模型能够<em>发现</em>任何人类示范者都未曾写出的策略——产出更有帮助、更准确、且更契合人类偏好的输出 <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a>。</p>

<p>这是每一款前沿模型背后的机制：GPT-4 <a class="book-citation" id="cite-openai2023gpt4-1" href="../../references.md#ref-openai2023gpt4" aria-label="参考文献 274">[274]</a>、Claude、Llama-3 <a class="book-citation" id="cite-grattafiori2024llama3-1" href="../../references.md#ref-grattafiori2024llama3" aria-label="参考文献 118">[118]</a> 和 DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 都在 SFT 之后施加 RL，作为把一个能力强但缺乏导向的模型转化为对齐助手的关键一步。</p>

## 本章内容

<ul class="chapter-outline">
<li id="two-paradigms-rl-llms"><a href="section-01.md">4.1　大语言模型 RL 的两大范式</a></li>
<li id="text-generation-as-mdp"><a href="section-02.md">4.2　将文本生成建模为 MDP</a></li>
<li id="the-rlhf-pipeline"><a href="section-03.md">4.3　RLHF 流水线</a></li>
<li id="what-this-part-covers"><a href="section-04.md">4.4　本部分路线图</a></li>
</ul>
