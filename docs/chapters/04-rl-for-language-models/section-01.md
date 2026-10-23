# 4.1　大语言模型 RL 的两大范式 {#two-paradigms-rl-llms}

<p>面向语言模型的 RL 方法可大致划分为两类范式，分别适用于不同的目标：</p>

#### 范式一：通过人类偏好实现对齐（RLHF/DPO）。 {#heading-4-3}

<p>将 RL 应用于大语言模型最初的动机是<strong>对齐</strong>——让模型变得有帮助、无害且诚实。<strong>基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）</strong> <a class="book-citation" id="cite-ouyang2022training-1" href="../../references.md#ref-ouyang2022training" aria-label="参考文献 280">[280]</a><a class="book-citation" id="cite-ziegler2019fine-1" href="../../references.md#ref-ziegler2019fine" aria-label="参考文献 442">[442]</a><a class="book-citation" id="cite-christiano2017deep-1" href="../../references.md#ref-christiano2017deep" aria-label="参考文献 57">[57]</a> 利用人类两两比较的判断（“哪个回答更好？”）来训练 reward 模型，然后优化 policy 以最大化这一学到的 reward。<strong>DPO</strong> <a class="book-citation" id="cite-rafailov2023direct-1" href="../../references.md#ref-rafailov2023direct" aria-label="参考文献 302">[302]</a> 通过彻底取消 reward 模型来简化这一过程，将偏好直接转化为有监督 loss。两种方法都能产出能够遵循指令且尊重安全约束的对齐助手。</p>

#### 范式二：通过可验证奖励增强能力（RLVR）。 {#heading-4-4}

<p>近期，RL 不仅被用于对齐，也被用于<strong>教授新能力</strong>——尤其是推理、数学和代码生成。此处的 reward 不再来自人类偏好，而来自<strong>可验证的结果</strong>：模型是否给出了正确答案？代码是否通过了所有测试？DeepSeek-R1 <a class="book-citation" id="cite-deepseek2025r1-1" href="../../references.md#ref-deepseek2025r1" aria-label="参考文献 72">[72]</a> 证明，配合基于规则的 reward（格式正确性 + 答案准确性）的 GRPO 能够在<em>完全不使用人类偏好数据</em>的情况下，训练模型发展出复杂的思维链（Chain-of-Thought，CoT）推理。这一范式——可验证奖励的强化学习（Reinforcement Learning from Verifiable Rewards, RLVR）——如今已成为构建推理模型和 agentic 系统的主流路线。</p>

<div class="admonition note"><p class="admonition-title">共同的底层基础</p>
<p>尽管两类范式的目标不同，但它们共享同一套核心机制：</p>

<ul>
<li><p>一个自回归生成文本的 <strong>policy</strong> <span class="arithmatex">\(\pi_\theta\)</span>（即大语言模型）</p></li>
<li><p>一个 <strong>reward 信号</strong> <span class="arithmatex">\(r(x, y)\)</span>（来自偏好学习或来自验证计算）</p></li>
<li><p>相对于参考 policy 的 <strong>KL 约束</strong>，用于防止退化解</p></li>
<li><p>用于将模型朝更高 reward 方向更新的 <strong>policy gradient 优化</strong>（PPO 或 GRPO）</p></li>
</ul>

<p>本部分的各章将逐一展开每个组件的细节。</p>
</div>
