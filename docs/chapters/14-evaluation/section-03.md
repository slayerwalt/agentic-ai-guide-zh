# 14.3　评估用合成数据生成 {#subsec:synthetic-data}

<p>人工标注昂贵且缓慢。合成数据生成利用 LLM 自身大规模生产评估数据。本节涵盖其主要范式。</p>

## 14.3.1　用于校准的 LLM-as-Judge {#llm-as-judge-for-calibration}

<p>当用 LLM 生成评估标签时，校准至关重要：评判者的分数必须与人类判断对齐。令 <span class="arithmatex">\(h_i \in [0,1]\)</span> 为样本 <span class="arithmatex">\(i\)</span> 的人类偏好分数，<span class="arithmatex">\(\hat{h}_i\)</span> 为评判者预测分数。校准误差由期望校准误差（Expected Calibration Error, ECE） <a class="book-citation" id="cite-guo2017calibration-1" href="../../references.md#ref-guo2017calibration" aria-label="参考文献 124">[124]</a> 度量：</p>



<div class="arithmatex">\[\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|
\tag{14.7}\]</div>

<p>其中 <span class="arithmatex">\(B_b\)</span> 是第 <span class="arithmatex">\(b\)</span> 个置信度区间，<span class="arithmatex">\(\text{acc}(B_b)\)</span> 是该区间内评判者与人类一致的样本比例，<span class="arithmatex">\(\text{conf}(B_b)\)</span> 是该区间内评判者的平均置信度。</p>

<p>一个良好校准的评判者满足：对所有 <span class="arithmatex">\(p \in [0,1]\)</span>，<span class="arithmatex">\(\mathbb{E}[\hat{h}_i \mid \hat{h}_i = p] = p\)</span>。校准可通过温度缩放改善：把评判者的原始 logit <span class="arithmatex">\(z\)</span> 替换为 <span class="arithmatex">\(z/T\)</span>，其中 <span class="arithmatex">\(T\)</span> 在留出的校准集上通过最小化负对数似然来调优。</p>

## 14.3.2　Self-Instruct {#self-instruct}

<p>Self-Instruct <a class="book-citation" id="cite-wang2022selfinstruct-1" href="../../references.md#ref-wang2022selfinstruct" aria-label="参考文献 371">[371]</a> 从一个由人工撰写的种子任务集合自举出指令跟随数据。算法如下：</p>

<ol>
<li><p>维护一个任务池，初始化为 <span class="arithmatex">\(175\)</span> 个种子任务。</p></li>
<li><p>从任务池中采样 <span class="arithmatex">\(8\)</span> 个任务，将其作为少样本示例，提示 LLM 生成新任务。</p></li>
<li><p>过滤生成的任务：移除近似重复项（与任何已有任务的 ROUGE-L 相似度 <span class="arithmatex">\(&gt; 0.7\)</span>），区分分类与非分类任务，并生成输入–输出实例。</p></li>
<li><p>将接受的任务加入任务池。</p></li>
<li><p>重复直到达到目标任务池规模。</p></li>
</ol>

<div class="admonition example"><p class="admonition-title">Self-Instruct Prompt 模板</p>
<div class="highlight"><pre><span></span><code><span class="n">system_prompt</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">Come up with a series of tasks:</span>
<span class="s2">Task 1: </span><span class="si">{seed_task_1_instruction}</span>
<span class="s2">Task 2: </span><span class="si">{seed_task_2_instruction}</span>
<span class="s2">...</span>
<span class="s2">Task 8: </span><span class="si">{seed_task_8_instruction}</span>
<span class="s2">Task 9:"""</span>
</code></pre></div>

<p>模型补全该 Prompt，生成新的任务指令。随后由另一个 Prompt 为该新任务生成输入–输出对。</p>
</div>

## 14.3.3　Evol-Instruct {#evol-instruct}

<p>Evol-Instruct <a class="book-citation" id="cite-xu2023wizardlm-1" href="../../references.md#ref-xu2023wizardlm" aria-label="参考文献 398">[398]</a> 通过迭代地将指令改写得更复杂或更多样，来演化一个种子指令集。应用两类演化算子：</p>

<ul>
<li><p><strong>深度演化（In-depth evolution）：</strong>增加约束、提升推理步骤数、将抽象具体化、加深对领域知识的要求。</p></li>
<li><p><strong>广度演化（In-breadth evolution）：</strong>在相关但不同的主题上生成新指令，提升主题多样性。</p></li>
</ul>

<p>若指令通过淘汰过滤器，则被接受：演化后的指令不能是简单复制、不能包含 &quot;I’m sorry&quot; 或类似的拒绝表述，且长度不能短于原指令。</p>

## 14.3.4　Constitutional AI 数据生成 {#constitutional-ai-data-generation}

<p>Constitutional AI（CAI） <a class="book-citation" id="cite-bai2022constitutional-1" href="../../references.md#ref-bai2022constitutional" aria-label="参考文献 16">[16]</a> 通过让模型依据一组原则（即“宪法”）对自身输出进行批评与修订，从而生成偏好数据。流水线如下：</p>

<ol>
<li><p><strong>监督学习阶段：</strong>采样一个有害 Prompt，生成初始回答，然后提示模型依据某条宪法原则对回答进行批评并修订。将修订后的回答作为监督微调（SFT）的目标。</p></li>
<li><p><strong>RL 阶段：</strong>生成回答对（原始 vs. 修订），用模型标注哪一个更符合宪法，并在这些标签上训练偏好模型。将偏好模型作为 RLHF 的 Reward 信号。</p></li>
</ol>

<p>这种方法无需让人工标注有害内容即可生成偏好数据，减少了标注者暴露于令人不适素材的风险。</p>

## 14.3.5　评估数据的蒸馏 {#distillation-for-evaluation-data}

<p>一个强大的教师模型（例如 GPT-4）可以生成高质量的评估数据，用于训练较小的评判模型。蒸馏流水线如下：</p>

<ol>
<li><p>收集多样化的 Prompt 与模型回答集合。</p></li>
<li><p>使用教师模型生成详尽的判断（分数 + 理由）。</p></li>
<li><p>在（Prompt、回答、判断）三元组上微调一个较小的模型。</p></li>
<li><p>在留出的人工标注上验证学生评判者。</p></li>
</ol>

<div class="admonition warning"><p class="admonition-title">蒸馏带来的偏置</p>
<p>从单一教师蒸馏得到的学生评判者会继承教师的偏置，包括冗长偏置（偏好更长的回答）、自我增强偏置（如果教师同时也是被评估的模型），以及位置偏置。始终要在独立的人工标注上验证蒸馏所得的评判者。</p>
</div>

## 14.3.6　Arena 风格的成对生成 {#arena-style-pairwise-generation}

<p>Chatbot Arena <a class="book-citation" id="cite-zheng2023judging-1" href="../../references.md#ref-zheng2023judging" aria-label="参考文献 433">[433]</a> 通过一个众包对战平台来生成评估数据：用户提交 Prompt 并在两个匿名化的模型回答之间投票选择偏好。这能产生一个大规模、自然多样的成对偏好数据集。关键设计选择包括：</p>

<ul>
<li><p><strong>匿名化：</strong>隐藏模型身份，防止品牌偏置。</p></li>
<li><p><strong>用户提交的 Prompt：</strong>保证 Prompt 的多样性以及与真实世界的相关性。</p></li>
<li><p><strong>平局处理：</strong>用户可声明平局，或表明两个回答都很差。</p></li>
<li><p><strong>去重：</strong>过滤近似重复的 Prompt，防止常见查询被过度代表。</p></li>
</ul>
