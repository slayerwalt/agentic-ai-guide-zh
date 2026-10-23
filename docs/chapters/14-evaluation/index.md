# 第 14 章 · LLM 评估 {#sec:llm-evaluation}

<p>评估是任何严谨机器学习流水线的支柱，但它或许是大语言模型开发中最被低估的组成部分。与经典的监督学习不同——在那里，一个带有真实标签的留出测试集就能提供干净的信号——评估 LLM 需要应对开放式生成、主观质量判断、多步推理链，以及无处不在的基准污染风险。本节系统性地梳理评估领域的全景：从评估类型的分类法、人工标注的机制，到排序指标的数学原理、LLM-as-Judge 的实操要点，再到那些悄无声息地腐蚀评估流水线的陷阱。</p>

<div class="admonition note"><p class="admonition-title">为何 LLM 评估如此困难</p>
<p><strong>三个根本性挑战</strong>使 LLM 评估有别于经典 ML 评估：</p>

<ol>
<li><p><strong>输出空间无界。</strong>语言模型可以生成任意字符串；很少存在唯一正确答案。</p></li>
<li><p><strong>质量是多维的。</strong>有用性、事实性、安全性、连贯性与风格是相互独立的维度，彼此之间可能存在权衡。</p></li>
<li><p><strong>评估本身就是一个语言任务。</strong>判断回答是否优良需要理解能力，这意味着评估本身会暴露在与生成相同的失败模式之下。</p></li>
</ol>
</div>

## 14.1　评估方案设计 {#subsec:eval-scheme}

<p>在收集任何一个数据点之前，从业者必须先决定要测量<em>什么</em>以及<em>如何</em>测量。一套有原则的分类法能避免一个常见错误：仅凭便利性而非与部署目标的对齐性来选择指标。</p>

### 14.1.1　评估类型分类法 {#taxonomy-of-evaluation-types}

##### 内在评估与外在评估（Intrinsic vs. Extrinsic）。 {#intrinsic-vs.-extrinsic-evaluation.}

<p><em>内在</em>（Intrinsic）评估孤立地度量模型输出的性质，不参照下游应用。留出语料上的困惑度、相对参考译文的 BLEU 分数、编码基准上的 pass@<span class="arithmatex">\(k\)</span>，都属于内在评估。<em>外在</em>（Extrinsic）评估则度量模型对真实任务或系统的影响：将 LLM 接入客服流水线后，工单升级率是否下降？编码助手是否提升了开发者的开发速度？</p>

<div class="admonition tip"><p class="admonition-title">内在—外在差距</p>
<p>内在指标廉价且可复现，但往往与真实世界的效用相关性很弱。困惑度更低的模型未必更有用。外在指标昂贵且缓慢，但能直接度量我们真正关心的东西。一套成熟的评估策略会用内在指标做快速迭代，用外在指标做最终验证。</p>
</div>

##### 自动评估与人工评估（Automatic vs. Human）。 {#automatic-vs.-human-evaluation.}

<p><em>自动</em>评估使用确定性函数（BLEU、精确匹配）或学习得到的模型（BERTScore、LLM-as-Judge），无需人工参与即可对输出打分。<em>人工</em>评估则由标注者对模型输出进行打分或排序。表 <a href="#tab:eval-taxonomy">14.1</a>总结了二者的权衡。</p>

<p class="table-caption" id="tab:eval-taxonomy"><span class="table-number">表 14.1　</span>评估方法分类法及其关键权衡。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>类型</strong></th><th scope="col"><strong>成本</strong></th><th scope="col"><strong>速度</strong></th><th scope="col"><strong>可复现性</strong></th><th scope="col"><strong>效度</strong></th></tr></thead>
<tbody><tr><td>自动（基于规则）</td><td>极低</td><td>极快</td><td>完美</td><td>低–中</td></tr>
<tr><td>自动（基于模型）</td><td>低</td><td>快</td><td>高</td><td>中–高</td></tr>
<tr><td>众包人工</td><td>中</td><td>数天</td><td>中</td><td>中</td></tr>
<tr><td>专家人工</td><td>高</td><td>数周</td><td>低–中</td><td>高</td></tr>
<tr><td>外在 / A/B 测试</td><td>极高</td><td>数月</td><td>低</td><td>极高</td></tr></tbody>
</table>
</div>

##### 基于参考与无参考评估（Reference-Based vs. Reference-Free）。 {#reference-based-vs.-reference-free-evaluation.}

<p>基于参考的指标（BLEU、ROUGE、BERTScore）将模型输出与一个或多个金标准参考进行比较。无参考指标（困惑度、LLM-as-Judge、人类偏好）则无需参考即可评估质量。当输出空间过大、无法穷尽收集参考时（例如开放式对话），无参考方法是必不可少的。</p>

### 14.1.2　何时使用何种方法 {#when-to-use-what}

<div class="admonition example"><p class="admonition-title">对话助手的评估策略</p>
<p><strong>开发阶段：</strong>使用自动指标（困惑度、摘要子任务上的 ROUGE、工具调用上的 pass@<span class="arithmatex">\(k\)</span>）进行快速迭代。每晚在标准评测套件（MMLU、HellaSwag、HumanEval）上运行基准测试。</p>

<p><strong>发布前阶段：</strong>开展人类偏好研究，将新模型与上一个检查点对比。在多样化的 Prompt 集合上使用 LLM-as-Judge 进行可扩展的成对比较。</p>

<p><strong>发布后阶段：</strong>监控外在指标（用户满意度分数、任务完成率），并留意生产 Prompt 中的分布漂移。</p>
</div>

<p>一个实用的决策框架：</p>

<ul>
<li><p>若任务有明确正确答案（数学、代码、事实型问答）：使用精确匹配或基于执行的指标。</p></li>
<li><p>若任务是开放式的但存在参考输出：使用基于参考的指标作为下界，并辅以 LLM-as-Judge。</p></li>
<li><p>若任务是主观的（有用性、语气、创造性）：使用人工评估或经过良好校准的 LLM 评判者。</p></li>
<li><p>若任务涉及多步 Agent 行为：使用任务成功率与轨迹效率（见第 <a href="#subsec:agentic-metrics">14.6</a>节）。</p></li>
</ul>

## 14.2　评估数据采集 {#subsec:data-collection}

<p>高质量的评估数据是可信基准的基础。本节涵盖人工标注流水线的设计、标注质量的统计度量，以及在众包与专家标注之间如何取舍。</p>

### 14.2.1　人工标注流水线 {#human-annotation-pipelines}

<p>一条健壮的标注流水线包含五个阶段：</p>

<ol>
<li><p><strong>任务定义。</strong>精确说明标注任务：评分对象是什么、采用何种量表、依据哪些标准。此阶段的歧义会传导为带噪标签。</p></li>
<li><p><strong>规范编写。</strong>撰写标注规范，并配以覆盖边界情形的详细示例。在全面部署前先用小规模试点组反复打磨。</p></li>
<li><p><strong>标注者招募与培训。</strong>选择具备相应背景知识的标注者。组织一次校准会议，让标注者标注相同样本并讨论分歧。</p></li>
<li><p><strong>质量控制。</strong>在标注队列中嵌入带已知标签的金标样本。对在金标样本上准确率低于阈值的标注者进行标记。</p></li>
<li><p><strong>聚合。</strong>使用多数投票、平均，或概率模型（如 Dawid–Skene）来合并每条样本的多次标注。</p></li>
</ol>

### 14.2.2　标注者间一致性 {#inter-annotator-agreement}

<p>原始一致率（所有标注者达成一致的样本比例）是一个不充分的度量，因为它没有考虑偶然一致。两个标准的偶然性修正度量是 Cohen’s <span class="arithmatex">\(\kappa\)</span> <a class="book-citation" id="cite-cohen1960coefficient-1" href="../../references.md#ref-cohen1960coefficient" aria-label="参考文献 60">[60]</a>（两位标注者）和 Fleiss’ <span class="arithmatex">\(\kappa\)</span> <a class="book-citation" id="cite-fleiss1971measuring-1" href="../../references.md#ref-fleiss1971measuring" aria-label="参考文献 92">[92]</a>（多位标注者）。</p>

##### Cohen’s Kappa。 {#cohens-kappa.}

<p>给定两位标注者将 <span class="arithmatex">\(N\)</span> 个样本标注为 <span class="arithmatex">\(k\)</span> 个类别，令 <span class="arithmatex">\(p_o\)</span> 为观察到的一致率，<span class="arithmatex">\(p_e\)</span> 为独立性假设下的期望一致率：</p>

<span id="eq:cohens-kappa"></span>

<div class="arithmatex">\[\kappa = \frac{p_o - p_e}{1 - p_e}

\tag{14.1}\]</div>

<p>其中</p>



<div class="arithmatex">\[p_o = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\text{annotator 1 agrees with annotator 2 on item } i]
\tag{14.2}\]</div>

<p>以及</p>



<div class="arithmatex">\[p_e = \sum_{c=1}^{k} p_{1c} \cdot p_{2c}
\tag{14.3}\]</div>

<p>其中 <span class="arithmatex">\(p_{jc}\)</span> 是标注者 <span class="arithmatex">\(j\)</span> 将样本判为类别 <span class="arithmatex">\(c\)</span> 的比例。Cohen’s <span class="arithmatex">\(\kappa\)</span> 的取值从 <span class="arithmatex">\(-1\)</span>（完全不一致）经 <span class="arithmatex">\(0\)</span>（偶然一致）到 <span class="arithmatex">\(1\)</span>（完全一致）。一般认为 <span class="arithmatex">\(0.6\)</span> 以上可以接受；<span class="arithmatex">\(0.8\)</span> 以上为强一致。</p>

##### Fleiss’ Kappa。 {#fleiss-kappa.}

<p>对于 <span class="arithmatex">\(n\)</span> 位标注者将 <span class="arithmatex">\(N\)</span> 个样本标注为 <span class="arithmatex">\(k\)</span> 个类别，令 <span class="arithmatex">\(n_{ij}\)</span> 为将样本 <span class="arithmatex">\(i\)</span> 标注为类别 <span class="arithmatex">\(j\)</span> 的标注者人数。定义：</p>



<div class="arithmatex">\[\bar{P}_i = \frac{1}{n(n-1)} \sum_{j=1}^{k} n_{ij}(n_{ij} - 1), \qquad \bar{P} = \frac{1}{N}\sum_{i=1}^{N}\bar{P}_i
\tag{14.4}\]</div>



<div class="arithmatex">\[\bar{P}_j^e = \frac{1}{Nn}\sum_{i=1}^{N} n_{ij}, \qquad P_e = \sum_{j=1}^{k} \left(\bar{P}_j^e\right)^2
\tag{14.5}\]</div>



<div class="arithmatex">\[\kappa_F = \frac{\bar{P} - P_e}{1 - P_e}
\tag{14.6}\]</div>

<div class="admonition warning"><p class="admonition-title">Kappa 的局限</p>
<p>Kappa 对类别分布敏感：当某一类别占据主导时，即使原始一致率很高，kappa 仍可能很低（即<em>kappa 悖论</em>）。对于序数量表，加权 kappa（按分歧的距离比例进行惩罚）更为合适。在 LLM 评估中，评分通常采用 1–5 的 Likert 量表，应始终报告加权 kappa。</p>
</div>

### 14.2.3　标注规范设计 {#annotation-guideline-design}

<p>有效的标注规范通常具备以下若干特征：</p>

<ul>
<li><p><strong>可操作化的标准。</strong>用具体、可观察的行为替换“有用”这类模糊词汇：例如“回答直接处理了用户的问题，并提供了完成所述任务所需的全部信息”。</p></li>
<li><p><strong>详细示例。</strong>每个评分等级至少提供两个示例，包括边界情形。</p></li>
<li><p><strong>决策树。</strong>对于复杂任务，用流程图引导标注者完成一系列二元决策，可降低认知负担并提升一致性。</p></li>
<li><p><strong>明确的范围。</strong>说明标注者<em>不</em>应考虑哪些因素（例如“不要因风格偏好而扣分；只关注事实准确性”）。</p></li>
</ul>

### 14.2.4　众包标注与专家标注 {#crowdsourcing-vs.-expert-annotation}

<p class="table-caption" id="table-14-2"><span class="table-number">表 14.2　</span>LLM 评估中众包标注与专家标注的对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>维度</strong></th><th scope="col"><strong>众包</strong></th><th scope="col"><strong>专家标注</strong></th></tr></thead>
<tbody><tr><td>单条样本成本</td><td>低（$0.01–$0.10）</td><td>高（$1–$50）</td></tr>
<tr><td>吞吐量</td><td>极高</td><td>低</td></tr>
<tr><td>领域知识</td><td>低</td><td>高</td></tr>
<tr><td>一致性</td><td>不稳定</td><td>高</td></tr>
<tr><td>适用任务</td><td>简单偏好、流畅度</td><td>技术准确性、安全性</td></tr>
<tr><td>平台</td><td>MTurk、Prolific、Scale AI</td><td>领域专家、内部团队</td></tr>
<tr><td>质量控制</td><td>金标样本、注意力检查</td><td>校准会议、同行复核</td></tr></tbody>
</table>
</div>

<p>对于安全关键型评估（例如检测有害输出、评估医疗建议），专家标注是不可妥协的。对于大规模偏好采集（例如构建 Reward 模型训练集），配以严格质量控制的众包往往是唯一可行的选择。</p>

## 14.3　评估用合成数据生成 {#subsec:synthetic-data}

<p>人工标注昂贵且缓慢。合成数据生成利用 LLM 自身大规模生产评估数据。本节涵盖其主要范式。</p>

### 14.3.1　用于校准的 LLM-as-Judge {#llm-as-judge-for-calibration}

<p>当用 LLM 生成评估标签时，校准至关重要：评判者的分数必须与人类判断对齐。令 <span class="arithmatex">\(h_i \in [0,1]\)</span> 为样本 <span class="arithmatex">\(i\)</span> 的人类偏好分数，<span class="arithmatex">\(\hat{h}_i\)</span> 为评判者预测分数。校准误差由期望校准误差（Expected Calibration Error, ECE） <a class="book-citation" id="cite-guo2017calibration-1" href="../../references.md#ref-guo2017calibration" aria-label="参考文献 124">[124]</a> 度量：</p>



<div class="arithmatex">\[\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|
\tag{14.7}\]</div>

<p>其中 <span class="arithmatex">\(B_b\)</span> 是第 <span class="arithmatex">\(b\)</span> 个置信度区间，<span class="arithmatex">\(\text{acc}(B_b)\)</span> 是该区间内评判者与人类一致的样本比例，<span class="arithmatex">\(\text{conf}(B_b)\)</span> 是该区间内评判者的平均置信度。</p>

<p>一个良好校准的评判者满足：对所有 <span class="arithmatex">\(p \in [0,1]\)</span>，<span class="arithmatex">\(\mathbb{E}[\hat{h}_i \mid \hat{h}_i = p] = p\)</span>。校准可通过温度缩放改善：把评判者的原始 logit <span class="arithmatex">\(z\)</span> 替换为 <span class="arithmatex">\(z/T\)</span>，其中 <span class="arithmatex">\(T\)</span> 在留出的校准集上通过最小化负对数似然来调优。</p>

### 14.3.2　Self-Instruct {#self-instruct}

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

### 14.3.3　Evol-Instruct {#evol-instruct}

<p>Evol-Instruct <a class="book-citation" id="cite-xu2023wizardlm-1" href="../../references.md#ref-xu2023wizardlm" aria-label="参考文献 398">[398]</a> 通过迭代地将指令改写得更复杂或更多样，来演化一个种子指令集。应用两类演化算子：</p>

<ul>
<li><p><strong>深度演化（In-depth evolution）：</strong>增加约束、提升推理步骤数、将抽象具体化、加深对领域知识的要求。</p></li>
<li><p><strong>广度演化（In-breadth evolution）：</strong>在相关但不同的主题上生成新指令，提升主题多样性。</p></li>
</ul>

<p>若指令通过淘汰过滤器，则被接受：演化后的指令不能是简单复制、不能包含 &quot;I’m sorry&quot; 或类似的拒绝表述，且长度不能短于原指令。</p>

### 14.3.4　Constitutional AI 数据生成 {#constitutional-ai-data-generation}

<p>Constitutional AI（CAI） <a class="book-citation" id="cite-bai2022constitutional-1" href="../../references.md#ref-bai2022constitutional" aria-label="参考文献 16">[16]</a> 通过让模型依据一组原则（即“宪法”）对自身输出进行批评与修订，从而生成偏好数据。流水线如下：</p>

<ol>
<li><p><strong>监督学习阶段：</strong>采样一个有害 Prompt，生成初始回答，然后提示模型依据某条宪法原则对回答进行批评并修订。将修订后的回答作为监督微调（SFT）的目标。</p></li>
<li><p><strong>RL 阶段：</strong>生成回答对（原始 vs. 修订），用模型标注哪一个更符合宪法，并在这些标签上训练偏好模型。将偏好模型作为 RLHF 的 Reward 信号。</p></li>
</ol>

<p>这种方法无需让人工标注有害内容即可生成偏好数据，减少了标注者暴露于令人不适素材的风险。</p>

### 14.3.5　评估数据的蒸馏 {#distillation-for-evaluation-data}

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

### 14.3.6　Arena 风格的成对生成 {#arena-style-pairwise-generation}

<p>Chatbot Arena <a class="book-citation" id="cite-zheng2023judging-1" href="../../references.md#ref-zheng2023judging" aria-label="参考文献 433">[433]</a> 通过一个众包对战平台来生成评估数据：用户提交 Prompt 并在两个匿名化的模型回答之间投票选择偏好。这能产生一个大规模、自然多样的成对偏好数据集。关键设计选择包括：</p>

<ul>
<li><p><strong>匿名化：</strong>隐藏模型身份，防止品牌偏置。</p></li>
<li><p><strong>用户提交的 Prompt：</strong>保证 Prompt 的多样性以及与真实世界的相关性。</p></li>
<li><p><strong>平局处理：</strong>用户可声明平局，或表明两个回答都很差。</p></li>
<li><p><strong>去重：</strong>过滤近似重复的 Prompt，防止常见查询被过度代表。</p></li>
</ul>

## 14.4　排序任务的指标 {#subsec:ranking-metrics}

<p>当目标是按质量对模型进行排序时，成对比较数据比绝对分数更为可靠。本节推导 LLM 评估中常用的主要排序系统。</p>

### 14.4.1　ELO 评分系统 {#elo-rating-system}

<p>ELO 系统 <a class="book-citation" id="cite-elo1978rating-1" href="../../references.md#ref-elo1978rating" aria-label="参考文献 84">[84]</a> 最初为国际象棋而设计，它为每位选手（模型）赋予一个标量评分 <span class="arithmatex">\(R\)</span>，使得选手 <span class="arithmatex">\(A\)</span> 对阵选手 <span class="arithmatex">\(B\)</span> 的期望得分为：</p>



<div class="arithmatex">\[E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
\tag{14.8}\]</div>

##### 推导。 {#derivation.}

<p>ELO 模型假设：每位选手在某一局比赛中的表现是一个以其评分为中心的 Logistic 分布所抽取的随机变量。<span class="arithmatex">\(A\)</span> 击败 <span class="arithmatex">\(B\)</span> 的概率为：</p>



<div class="arithmatex">\[P(A \succ B) = \sigma\!\left(\frac{R_A - R_B}{s}\right) = \frac{1}{1 + e^{-(R_A - R_B)/s}}
\tag{14.9}\]</div>

<p>其中 <span class="arithmatex">\(s = 400/\ln(10) \approx 173.7\)</span> 是一个尺度参数，其选择使得 400 分的差距对应于 <span class="arithmatex">\(10:1\)</span> 的胜率比。每场比赛后，给定结果 <span class="arithmatex">\(S_A \in \{0, 0.5, 1\}\)</span>（负、平、胜），评分更新如下：</p>



<div class="arithmatex">\[R_A \leftarrow R_A + K(S_A - E_A), \qquad R_B \leftarrow R_B + K(S_B - E_B)
\tag{14.10}\]</div>

<p>其中 <span class="arithmatex">\(K\)</span> 是控制学习率的 <span class="arithmatex">\(K\)</span>-因子。在 Chatbot Arena 中使用 <span class="arithmatex">\(K = 4\)</span>。</p>

<div class="admonition tip"><p class="admonition-title">ELO 的直觉</p>
<p>ELO 本质上是在 Logistic 模型下，对观察结果的对数似然进行随机梯度下降式更新。每场比赛提供一个带噪的 Gradient 信号；<span class="arithmatex">\(K\)</span>-因子控制步长。<span class="arithmatex">\(K\)</span> 越大，适应越快但噪声越大；<span class="arithmatex">\(K\)</span> 越小则越稳定，但反映真实实力变化的速度越慢。</p>
</div>

##### ELO 的 Bootstrap 置信区间。 {#bootstrap-confidence-intervals-for-elo.}

<p>由于 ELO 评分依赖于比赛被处理的顺序，置信区间通过 Bootstrap 重采样计算：对对战日志有放回地重采样 <span class="arithmatex">\(B = 1000\)</span> 次，对每次重采样都从头重新计算 ELO 评分，并以第 2.5 与第 97.5 百分位作为 95% 置信区间。</p>

### 14.4.2　Bradley–Terry 模型 {#bradleyterry-model}

<p>Bradley–Terry（BT）模型 <a class="book-citation" id="cite-bradley1952rank-1" href="../../references.md#ref-bradley1952rank" aria-label="参考文献 29">[29]</a> 是 ELO 的一种最大似然替代方案。给定 <span class="arithmatex">\(n\)</span> 个模型，其实力参数为 <span class="arithmatex">\(\beta_1, \ldots, \beta_n &gt; 0\)</span>，模型 <span class="arithmatex">\(i\)</span> 击败模型 <span class="arithmatex">\(j\)</span> 的概率为：</p>



<div class="arithmatex">\[P(i \succ j) = \frac{\beta_i}{\beta_i + \beta_j}
\tag{14.11}\]</div>

<p>给定一组成对结果 <span class="arithmatex">\(\{(i_k, j_k, y_k)\}_{k=1}^{M}\)</span>，其中若 <span class="arithmatex">\(i_k\)</span> 击败 <span class="arithmatex">\(j_k\)</span> 则 <span class="arithmatex">\(y_k = 1\)</span>，否则 <span class="arithmatex">\(y_k = 0\)</span>，对数似然为：</p>



<div class="arithmatex">\[\ell(\boldsymbol{\beta}) = \sum_{k=1}^{M} \left[ y_k \log \frac{\beta_{i_k}}{\beta_{i_k} + \beta_{j_k}} + (1-y_k) \log \frac{\beta_{j_k}}{\beta_{i_k} + \beta_{j_k}} \right]
\tag{14.12}\]</div>

<p>极大似然估计 <span class="arithmatex">\(\hat{\boldsymbol{\beta}}\)</span> 可通过迭代缩放或梯度上升求得。BT 模型仅在相差一个乘法常数的意义下可辨识；常用的归一化是 <span class="arithmatex">\(\sum_i \log \beta_i = 0\)</span>。在对数空间中令 <span class="arithmatex">\(\theta_i = \log \beta_i\)</span> 得：</p>



<div class="arithmatex">\[P(i \succ j) = \sigma(\theta_i - \theta_j)
\tag{14.13}\]</div>

<p>这等价于一个带条目特定截距的 Logistic 回归。在能拿到完整对战历史时，BT 模型优于 ELO，因为它同时利用了所有数据，而不是顺序处理比赛。</p>

### 14.4.3　TrueSkill {#trueskill}

<p>TrueSkill <a class="book-citation" id="cite-herbrich2006trueskill-1" href="../../references.md#ref-herbrich2006trueskill" aria-label="参考文献 138">[138]</a> 是一种贝叶斯式的实力评分系统，将每位选手的实力建模为高斯随机变量 <span class="arithmatex">\(s_i \sim \mathcal{N}(\mu_i, \sigma_i^2)\)</span>。选手 <span class="arithmatex">\(i\)</span> 在某一局中的表现为 <span class="arithmatex">\(p_i = s_i + \epsilon_i\)</span>，其中 <span class="arithmatex">\(\epsilon_i \sim \mathcal{N}(0, \beta^2)\)</span> 为比赛特定的噪声。当 <span class="arithmatex">\(p_i &gt; p_j\)</span> 时，选手 <span class="arithmatex">\(i\)</span> 击败选手 <span class="arithmatex">\(j\)</span>。</p>

<p>观察到 <span class="arithmatex">\(i \succ j\)</span> 后的后验更新通过期望传播（Expectation Propagation, EP）计算。胜方的关键更新方程为：</p>



<div class="arithmatex">\[\mu_i \leftarrow \mu_i + \frac{\sigma_i^2}{c} \cdot v\!\left(\frac{\mu_i - \mu_j}{c}\right)
\tag{14.14}\]</div>



<div class="arithmatex">\[\sigma_i^2 \leftarrow \sigma_i^2 \left[1 - \frac{\sigma_i^2}{c^2} \cdot w\!\left(\frac{\mu_i - \mu_j}{c}\right)\right]
\tag{14.15}\]</div>

<p>其中 <span class="arithmatex">\(c = \sqrt{2\beta^2 + \sigma_i^2 + \sigma_j^2}\)</span>，<span class="arithmatex">\(v(t) = \phi(t)/\Phi(t)\)</span>，<span class="arithmatex">\(w(t) = v(t)(v(t) + t)\)</span> 为截断高斯修正因子（<span class="arithmatex">\(\phi\)</span> 与 <span class="arithmatex">\(\Phi\)</span> 分别是标准正态的 PDF 与 CDF）。TrueSkill 的不确定性估计 <span class="arithmatex">\(\sigma_i\)</span> 在识别需要更多评估数据的模型时尤其有用。</p>

### 14.4.4　带置信区间的胜率 {#win-rate-with-confidence-intervals}

<p>最简单的排序指标是胜率：模型 <span class="arithmatex">\(A\)</span> 在成对比较中胜出的比例。给定 <span class="arithmatex">\(n\)</span> 次比较，其中 <span class="arithmatex">\(w\)</span> 次胜出，胜率为 <span class="arithmatex">\(\hat{p} = w/n\)</span>。在 <span class="arithmatex">\(p = 0\)</span> 与 <span class="arithmatex">\(p = 1\)</span> 附近，Wilson 分数置信区间 <a class="book-citation" id="cite-wilson1927probable-1" href="../../references.md#ref-wilson1927probable" aria-label="参考文献 380">[380]</a> 的覆盖性能优于朴素的 Wald 区间，因此更被推荐：</p>



<div class="arithmatex">\[\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
\tag{14.16}\]</div>

<p>其中 95% 区间对应 <span class="arithmatex">\(z = 1.96\)</span>。对于多方比较，胜率应针对一个固定的基线模型来计算，以确保可比性。</p>

### 14.4.5　Chatbot Arena 方法论 {#chatbot-arena-methodology}

<p>Chatbot Arena <a class="book-citation" id="cite-zheng2023judging-1--section-04" href="../../references.md#ref-zheng2023judging" aria-label="参考文献 433">[433]</a> 将上述要素组合为一个生产规模的评估系统：</p>

<ol>
<li><p>用户提交 Prompt，并获得两个匿名化模型的回答。</p></li>
<li><p>用户对偏好的回答进行投票（或声明平局）。</p></li>
<li><p>投票通过 BT 模型聚合以生成排行榜。</p></li>
<li><p>对每个模型的分数报告 Bootstrap 置信区间。</p></li>
<li><p>置信区间重叠的模型在统计上被视为无法区分。</p></li>
</ol>

<p>截至 2024 年，Chatbot Arena 已累计收集超过一百万条人类偏好投票，是目前公开可得的最大规模 LLM 偏好数据集。</p>

## 14.5　生成任务的指标 {#subsec:generation-metrics}

<p>生成指标量化了在具备参考答案或明确正确性标准的任务上模型输出的质量。</p>

### 14.5.1　BLEU {#bleu}

<p>BLEU（Bilingual Evaluation Understudy） <a class="book-citation" id="cite-papineni2002bleu-1" href="../../references.md#ref-papineni2002bleu" aria-label="参考文献 283">[283]</a> 度量假设译文 <span class="arithmatex">\(h\)</span> 与一个或多个参考 <span class="arithmatex">\(\mathcal{R}\)</span> 之间的 <span class="arithmatex">\(n\)</span>-gram 精度：</p>



<div class="arithmatex">\[\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
\tag{14.17}\]</div>

<p>其中 <span class="arithmatex">\(p_n\)</span> 是修正后的 <span class="arithmatex">\(n\)</span>-gram 精度，<span class="arithmatex">\(w_n = 1/N\)</span> 为均匀权重，BP 为简短惩罚（brevity penalty）：</p>



<div class="arithmatex">\[\text{BP} = \begin{cases} 1 &amp; \text{if } |h| &gt; |r| \\ e^{1 - |r|/|h|} &amp; \text{if } |h| \leq |r| \end{cases}
\tag{14.18}\]</div>

<p>其中 <span class="arithmatex">\(|r|\)</span> 为长度最接近的参考的长度。修正后的 <span class="arithmatex">\(n\)</span>-gram 精度将每个 <span class="arithmatex">\(n\)</span>-gram 的计数截断到其在任一参考中的最大出现次数：</p>



<div class="arithmatex">\[p_n = \frac{\sum_{\text{ngram} \in h} \min\!\left(\text{count}(\text{ngram}, h),\, \max_{r \in \mathcal{R}} \text{count}(\text{ngram}, r)\right)}{\sum_{\text{ngram} \in h} \text{count}(\text{ngram}, h)}
\tag{14.19}\]</div>

<div class="admonition warning"><p class="admonition-title">BLEU 的局限</p>
<p>BLEU 是为带多个参考的机器翻译而设计的。对于只有单一参考的开放式生成，即使输出质量很高，BLEU 分数也常常接近零。BLEU 无法捕捉语义相似性，会惩罚合理的复述，并且对分词敏感。仅在存在多个多样化参考且任务输出多样性较低时再使用 BLEU。</p>
</div>

### 14.5.2　ROUGE {#rouge}

<p>ROUGE（Recall-Oriented Understudy for Gisting Evaluation） <a class="book-citation" id="cite-lin2004rouge-1" href="../../references.md#ref-lin2004rouge" aria-label="参考文献 216">[216]</a> 是一类面向召回的指标，专为摘要任务设计：</p>



<div class="arithmatex">\[\begin{align*}
\text{ROUGE-N} &amp;= \frac{\sum_{r \in \mathcal{R}} \sum_{\text{ngram} \in r} \min(\text{count}(\text{ngram}, h), \text{count}(\text{ngram}, r))}{\sum_{r \in \mathcal{R}} \sum_{\text{ngram} \in r} \text{count}(\text{ngram}, r)} \tag{14.20}\\[6pt]

    \text{ROUGE-L} &amp;= \frac{\text{LCS}(h, r)}{|r|}
\tag{14.21}
\end{align*}\]</div>

<p>其中 LCS 表示最长公共子序列（Longest Common Subsequence）。ROUGE-1 与 ROUGE-2 度量 unigram 与 bigram 的召回；ROUGE-L 则捕捉句子级结构。F-measure 变体在精度与召回之间取得平衡：</p>



<div class="arithmatex">\[\text{ROUGE-N}_F = \frac{(1+\beta^2) \cdot P \cdot R}{\beta^2 P + R}
\tag{14.22}\]</div>

<p>当 <span class="arithmatex">\(\beta = 1\)</span> 时为等权。</p>

### 14.5.3　BERTScore {#bertscore}

<p>BERTScore <a class="book-citation" id="cite-zhang2020bertscore-1" href="../../references.md#ref-zhang2020bertscore" aria-label="参考文献 426">[426]</a> 使用预训练 BERT 模型的上下文 Embedding 来计算 Token 级相似度。给定假设 Token <span class="arithmatex">\(\hat{\mathbf{x}} = \langle \hat{x}_1, \ldots, \hat{x}_m \rangle\)</span> 与参考 Token <span class="arithmatex">\(\mathbf{x} = \langle x_1, \ldots, x_n \rangle\)</span>，及其 Embedding <span class="arithmatex">\(\hat{\mathbf{e}}_i\)</span> 与 <span class="arithmatex">\(\mathbf{e}_j\)</span>：</p>



<div class="arithmatex">\[\begin{align*}
R_{\text{BERT}} &amp;= \frac{1}{|x|} \sum_{x_j \in \mathbf{x}} \max_{\hat{x}_i \in \hat{\mathbf{x}}} \frac{\hat{\mathbf{e}}_i^\top \mathbf{e}_j}{\|\hat{\mathbf{e}}_i\| \|\mathbf{e}_j\|} \tag{14.23}\\[4pt]

    P_{\text{BERT}} &amp;= \frac{1}{|\hat{x}|} \sum_{\hat{x}_i \in \hat{\mathbf{x}}} \max_{x_j \in \mathbf{x}} \frac{\hat{\mathbf{e}}_i^\top \mathbf{e}_j}{\|\hat{\mathbf{e}}_i\| \|\mathbf{e}_j\|} \tag{14.24}\\[4pt]

    F_{\text{BERT}} &amp;= 2 \cdot \frac{P_{\text{BERT}} \cdot R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}
\tag{14.25}
\end{align*}\]</div>

<p>BERTScore 与人类判断的相关性优于 BLEU 与 ROUGE，尤其在处理复述以及语义等价但词面不同的输出时。使用逆文档频率（IDF）进行重要性加权可进一步提升相关性：</p>



<div class="arithmatex">\[R_{\text{BERT}}^{\text{idf}} = \frac{\sum_{x_j \in \mathbf{x}} \text{idf}(x_j) \max_{\hat{x}_i} \cos(\hat{\mathbf{e}}_i, \mathbf{e}_j)}{\sum_{x_j \in \mathbf{x}} \text{idf}(x_j)}
\tag{14.26}\]</div>

### 14.5.4　METEOR {#meteor}

<p>METEOR <a class="book-citation" id="cite-banerjee2005meteor-1" href="../../references.md#ref-banerjee2005meteor" aria-label="参考文献 19">[19]</a> 通过在 unigram 匹配上计算 F-score，并配合词干化与同义词匹配模块，解决了 BLEU 对召回的盲视：</p>



<div class="arithmatex">\[\text{METEOR} = F_{\text{mean}} \cdot (1 - \text{Pen})
\tag{14.27}\]</div>

<p>其中 <span class="arithmatex">\(F_{\text{mean}} = \frac{10PR}{R + 9P}\)</span>（召回加权的调和平均），碎片化惩罚 <span class="arithmatex">\(\text{Pen} = 0.5 \cdot (c/u_m)^3\)</span> 对非连续匹配进行惩罚（<span class="arithmatex">\(c\)</span> 为块数，<span class="arithmatex">\(u_m\)</span> 为匹配的 unigram 数）。</p>

### 14.5.5　困惑度（Perplexity） {#perplexity}

<p>困惑度（Perplexity）衡量语言模型对留出文本序列 <span class="arithmatex">\(w_1, w_2, \ldots, w_T\)</span> 的预测能力：</p>



<div class="arithmatex">\[\text{PPL}(w_{1:T}) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log P_\theta(w_t \mid w_{1:t-1})\right)
\tag{14.28}\]</div>

<p>困惑度越低，预测性能越好。困惑度适合在相同分词与测试集上比较模型，但在词表或分词器不同的模型间不能直接比较。在评估中，困惑度最实用的用途是作为一种合理性检查（sanity check）以及用于检测分布漂移。</p>

### 14.5.6　代码任务的 Pass@k {#passk-for-code}

<p>对于代码生成，功能正确性通过在测试用例上执行生成的代码来衡量。pass@<span class="arithmatex">\(k\)</span> 指标 <a class="book-citation" id="cite-chen2021evaluating-1" href="../../references.md#ref-chen2021evaluating" aria-label="参考文献 47">[47]</a> 估计在 <span class="arithmatex">\(k\)</span> 个生成样本中至少有一个通过全部测试的概率：</p>



<div class="arithmatex">\[\text{pass@}k = \mathbb{E}_{\text{problems}}\!\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
\tag{14.29}\]</div>

<p>其中 <span class="arithmatex">\(n\)</span> 是每个问题生成的样本总数，<span class="arithmatex">\(c\)</span> 是其中通过测试的样本数。该无偏估计量避免了朴素估计量（精确采样 <span class="arithmatex">\(k\)</span> 个解并检查是否通过）的高方差问题。在实践中通常生成 <span class="arithmatex">\(n = 200\)</span> 个样本，并报告 pass@1、pass@10、pass@100。</p>

<div class="admonition example"><p class="admonition-title">Pass@k 计算</p>
<div class="highlight"><pre><span></span><code><span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">scipy.special</span><span class="w"> </span><span class="kn">import</span> <span class="n">comb</span>

<span class="k">def</span><span class="w"> </span><span class="nf">pass_at_k</span><span class="p">(</span><span class="n">n</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">c</span><span class="p">:</span> <span class="nb">int</span><span class="p">,</span> <span class="n">k</span><span class="p">:</span> <span class="nb">int</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""pass@k 的无偏估计量。</span>

<span class="sd">    Args:</span>
<span class="sd">        n: 每个问题生成的样本总数</span>
<span class="sd">        c: 通过全部测试的样本数</span>
<span class="sd">        k: 考虑的样本数</span>
<span class="sd">    """</span>
    <span class="k">if</span> <span class="n">n</span> <span class="o">-</span> <span class="n">c</span> <span class="o">&lt;</span> <span class="n">k</span><span class="p">:</span>
        <span class="k">return</span> <span class="mf">1.0</span>
    <span class="k">return</span> <span class="mf">1.0</span> <span class="o">-</span> <span class="n">comb</span><span class="p">(</span><span class="n">n</span> <span class="o">-</span> <span class="n">c</span><span class="p">,</span> <span class="n">k</span><span class="p">,</span> <span class="n">exact</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span> <span class="o">/</span> <span class="n">comb</span><span class="p">(</span><span class="n">n</span><span class="p">,</span> <span class="n">k</span><span class="p">,</span> <span class="n">exact</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>

<span class="c1"># 示例：200 个样本，15 个通过，计算 pass@1、pass@10、pass@100</span>
<span class="k">for</span> <span class="n">k</span> <span class="ow">in</span> <span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">10</span><span class="p">,</span> <span class="mi">100</span><span class="p">]:</span>
    <span class="n">score</span> <span class="o">=</span> <span class="n">pass_at_k</span><span class="p">(</span><span class="n">n</span><span class="o">=</span><span class="mi">200</span><span class="p">,</span> <span class="n">c</span><span class="o">=</span><span class="mi">15</span><span class="p">,</span> <span class="n">k</span><span class="o">=</span><span class="n">k</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"pass@</span><span class="si">{</span><span class="n">k</span><span class="si">}</span><span class="s2">: </span><span class="si">{</span><span class="n">score</span><span class="si">:</span><span class="s2">.4f</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="c1"># pass@1:   0.0750</span>
<span class="c1"># pass@10:  0.5391</span>
<span class="c1"># pass@100: 0.9999</span>
</code></pre></div>
</div>

### 14.5.7　精确匹配与 F1 {#exact-match-and-f1}

<p>对于抽取式问答（例如 SQuAD），有两个标准指标：</p>

<ul>
<li><p><strong>精确匹配（Exact Match, EM）：</strong>二元指示量，判断归一化（小写化、去除冠词与标点）后的预测答案字符串是否与任一金标答案完全相同。</p></li>
<li><p><strong>Token 级 F1：</strong>将预测与金标答案视为 Token 的多重集，并计算 F1 分数：</p>



<div class="arithmatex">\[F1 = \frac{2 \cdot |\text{pred} \cap \text{gold}|}{|\text{pred}| + |\text{gold}|}
\tag{14.30}\]</div></li>
</ul>

<p>对于多答案设置，报告对所有金标答案取最大值后的 F1。</p>

<p class="table-caption" id="table-14-3"><span class="table-number">表 14.3　</span>生成指标汇总：适用性与关键性质。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>指标</strong></th><th scope="col"><strong>任务</strong></th><th scope="col"><strong>是否无参考？</strong></th><th scope="col"><strong>与人类判断的相关性</strong></th></tr></thead>
<tbody><tr><td>BLEU</td><td>翻译</td><td>否</td><td>低–中</td></tr>
<tr><td>ROUGE</td><td>摘要</td><td>否</td><td>中</td></tr>
<tr><td>BERTScore</td><td>通用 NLG</td><td>否</td><td>高</td></tr>
<tr><td>METEOR</td><td>翻译</td><td>否</td><td>中–高</td></tr>
<tr><td>Perplexity</td><td>语言模型质量</td><td>是</td><td>低</td></tr>
<tr><td>Pass@k</td><td>代码生成</td><td>否（依赖测试）</td><td>极高</td></tr>
<tr><td>Exact Match</td><td>抽取式问答</td><td>否</td><td>极高</td></tr>
<tr><td>Token F1</td><td>抽取式问答</td><td>否</td><td>高</td></tr></tbody>
</table>
</div>

## 14.6　Agent 任务的指标 {#subsec:agentic-metrics}

<p>Agent 化的 LLM 在环境中运作、执行动作序列，并须完成多步任务。标准生成指标在此并不充分；Agent 评估需要能够刻画任务完成度、效率以及中间步骤质量的指标。</p>

### 14.6.1　任务成功率 {#task-success-rate}

<p>Agent 任务的主要指标是任务成功率（Task Success Rate, TSR）：Agent 达成指定目标状态的任务比例：</p>



<div class="arithmatex">\[\text{TSR} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \mathbf{1}[\text{goal}(\tau) \text{ achieved}]
\tag{14.31}\]</div>

<p>目标的达成通常由一个确定性的 Oracle 来验证（例如检查数据库状态、文件系统状态或测试用例执行结果）。对于允许部分得分的任务，可以定义分级成功度量：</p>



<div class="arithmatex">\[\text{TSR}_{\text{graded}} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{score}(\tau) \in [0, 1]
\tag{14.32}\]</div>

### 14.6.2　轨迹效率 {#trajectory-efficiency}

<p>一个成功的 Agent 应以尽可能少的非必要动作完成任务。轨迹效率衡量最优轨迹长度与 Agent 实际轨迹长度之比：</p>



<div class="arithmatex">\[\eta = \frac{L^*}{L_{\text{agent}}}
\tag{14.33}\]</div>

<p>其中 <span class="arithmatex">\(L^*\)</span> 是最短成功轨迹的长度（由 Oracle 或人类专家计算得出），<span class="arithmatex">\(L_{\text{agent}}\)</span> 是 Agent 所执行的动作数。<span class="arithmatex">\(\eta \in (0, 1]\)</span>，<span class="arithmatex">\(\eta = 1\)</span> 表示最优效率。对于失败的轨迹，<span class="arithmatex">\(\eta = 0\)</span>。</p>

<p>一个互补指标是<em>冗余率</em>（redundancy rate）：Agent 动作中不出现在任何最优轨迹里的比例。</p>

### 14.6.3　工具调用准确性 {#tool-use-accuracy}

<p>对于调用外部工具（API、代码解释器、搜索引擎）的 Agent，工具调用（Tool Calling）准确性衡量调用的正确性：</p>



<div class="arithmatex">\[\text{TUA} = \frac{\text{\# correct tool calls}}{\text{\# total tool calls}}
\tag{14.34}\]</div>

<p>一次工具调用被视为正确当且仅当：（a）选择了正确的工具，（b）参数有效，（c）该调用发生在轨迹中合适的时机。当工具选择正确但参数错误时，可酌情给予部分得分。</p>

### 14.6.4　多步推理准确性 {#multi-step-reasoning-accuracy}

<p>对于需要推理链的任务（例如多跳问答、数学解题），步骤级准确性衡量正确推理步骤所占的比例：</p>



<div class="arithmatex">\[\text{SRA} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \frac{1}{|S_\tau|} \sum_{s \in S_\tau} \mathbf{1}[s \text{ is correct}]
\tag{14.35}\]</div>

<p>其中 <span class="arithmatex">\(S_\tau\)</span> 是轨迹 <span class="arithmatex">\(\tau\)</span> 中的推理步骤集合。步骤的正确性可由过程奖励模型（Process Reward Model, PRM）或人工标注来验证。</p>

### 14.6.5　SWE-bench 方法论 {#swe-bench-methodology}

<p>SWE-bench <a class="book-citation" id="cite-jimenez2024swebench-1" href="../../references.md#ref-jimenez2024swebench" aria-label="参考文献 169">[169]</a> 在真实软件工程任务上评估 LLM：给定一个 GitHub issue 描述与仓库代码库，模型必须生成解决该 issue 的 patch。评估流程如下：</p>

<ol>
<li><p>将 issue 描述与相关代码上下文提供给模型。</p></li>
<li><p>模型生成一个 patch（统一 diff 格式）。</p></li>
<li><p>将 patch 应用到仓库。</p></li>
<li><p>执行仓库的测试套件；若全部测试通过，则任务成功。</p></li>
</ol>

<p>主要指标为 <strong>% Resolved</strong>：生成的 patch 能通过全部测试的 issue 所占比例。SWE-bench Verified 是一个经过人工标注者验证为可解且无歧义的 500 题精选子集。SWE-bench Lite 则是一个 300 题的子集，专为更快的评估而设计。</p>

<div class="admonition note"><p class="admonition-title">SWE-bench 关键数据（截至 2024 年）</p>
<ul>
<li><p><strong>完整基准：</strong>来自 12 个热门 Python 仓库的 2,294 个任务。</p></li>
<li><p><strong>最佳开源 Agent：</strong>约 43% 解决率（SWE-bench Verified）。</p></li>
<li><p><strong>人类表现：</strong>约 87% 解决率（每个任务 15 分钟）。</p></li>
<li><p><strong>评估成本：</strong>基于 API 的模型每个任务约 $0.25。</p></li>
</ul>
</div>

### 14.6.6　WebArena 方法论 {#webarena-methodology}

<p>WebArena <a class="book-citation" id="cite-zhou2024webarena-1" href="../../references.md#ref-zhou2024webarena" aria-label="参考文献 440">[440]</a> 在沙箱浏览器环境中以贴近真实的网页导航任务评估 Agent。该基准包含跨五个 Web 应用（电商、社交论坛、协作开发、内容管理与地图）的 812 个任务。评估方式如下：</p>

<ul>
<li><p><strong>功能性评估：</strong>通过检查应用状态来验证任务结果（例如“商品是否已加入购物车？”“帖子是否已创建？”）。</p></li>
<li><p><strong>基于 URL 的评估：</strong>对于导航任务，将最终 URL 与期望 URL 进行比较。</p></li>
<li><p><strong>基于程序的评估：</strong>由自定义评估脚本检查复杂条件（例如“价格是否小于 $50？”）。</p></li>
</ul>

<p>主要指标是任务成功率。人类表现约为 78%；当前最先进的 Agent 约为 35–45%。</p>

<p class="table-caption" id="table-14-4"><span class="table-number">表 14.4　</span>Agent 评估基准对比。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>基准</strong></th><th scope="col"><strong>领域</strong></th><th scope="col"><strong>任务数</strong></th><th scope="col"><strong>评估方式</strong></th><th scope="col"><strong>SOTA（%）</strong></th></tr></thead>
<tbody><tr><td>SWE-bench</td><td>软件工程</td><td>2,294</td><td>测试执行</td><td><span class="arithmatex">\(\sim\)</span>43</td></tr>
<tr><td>SWE-bench Lite</td><td>软件工程</td><td>300</td><td>测试执行</td><td><span class="arithmatex">\(\sim\)</span>50</td></tr>
<tr><td>WebArena</td><td>网页导航</td><td>812</td><td>状态/URL/程序</td><td><span class="arithmatex">\(\sim\)</span>40</td></tr>
<tr><td>ALFWorld <a class="book-citation" id="cite-shridhar2021alfworld-1" href="../../references.md#ref-shridhar2021alfworld" aria-label="参考文献 329">[329]</a></td><td>家居任务</td><td>3,553</td><td>仿真器状态</td><td><span class="arithmatex">\(\sim\)</span>90</td></tr>
<tr><td>AgentBench <a class="book-citation" id="cite-liu2023agentbench-1" href="../../references.md#ref-liu2023agentbench" aria-label="参考文献 229">[229]</a></td><td>多领域</td><td>1,091</td><td>任务特定</td><td><span class="arithmatex">\(\sim\)</span>45</td></tr></tbody>
</table>
</div>

## 14.7　LLM-as-Judge {#subsec:llm-as-judge}

<p>LLM-as-Judge <a class="book-citation" id="cite-zheng2023judging-1--section-07" href="../../references.md#ref-zheng2023judging" aria-label="参考文献 433">[433]</a> 使用一个能力强的 LLM 来评估其他（或同一个）LLM 的输出。该方法无需人工标注就能扩展到大规模评估集合，并能为其判断提供详尽的理由。</p>

### 14.7.1　设置与 Prompt 模板 {#setup-and-prompt-templates}

<p>向评判者提供一个 Prompt、一个或多个模型回答，以及一份评估细则。常见的三种格式：</p>

##### 点评式评分（Pointwise scoring）。 {#pointwise-scoring.}

<p>评判者对单个回答给出绝对分数：</p>

<div class="admonition example"><p class="admonition-title">Pointwise Judge Prompt</p>
<div class="highlight"><pre><span></span><code><span class="n">POINTWISE_PROMPT</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">You are an expert evaluator. Rate the following response on a scale</span>
<span class="s2">of 1-10 for helpfulness, accuracy, and clarity.</span>

<span class="s2">[Question]</span>
<span class="si">{question}</span>

<span class="s2">[Response]</span>
<span class="si">{response}</span>

<span class="s2">Provide your evaluation in the following format:</span>
<span class="s2">Reasoning: &lt;step-by-step analysis&gt;</span>
<span class="s2">Score: &lt;integer from 1 to 10&gt;</span>
<span class="s2">"""</span>
</code></pre></div>
</div>

##### 成对比较（Pairwise comparison）。 {#pairwise-comparison.}

<p>评判者比较两个回答并选出更优者：</p>

<div class="admonition example"><p class="admonition-title">Pairwise Judge Prompt</p>
<div class="highlight"><pre><span></span><code><span class="n">PAIRWISE_PROMPT</span> <span class="o">=</span> <span class="s2">"""</span>
<span class="s2">You are an expert evaluator. Compare the two responses below and</span>
<span class="s2">determine which is better. Consider helpfulness, accuracy, and</span>
<span class="s2">depth of explanation.</span>

<span class="s2">[Question]</span>
<span class="si">{question}</span>

<span class="s2">[Response A]</span>
<span class="si">{response_a}</span>

<span class="s2">[Response B]</span>
<span class="si">{response_b}</span>

<span class="s2">Output exactly one of: [[A]], [[B]], or [[C]] (tie).</span>
<span class="s2">Reasoning: &lt;your analysis&gt;</span>
<span class="s2">Verdict: &lt;[[A]], [[B]], or [[C]]&gt;</span>
<span class="s2">"""</span>
</code></pre></div>
</div>

##### 参考引导式评分（Reference-guided scoring）。 {#reference-guided-scoring.}

<p>向评判者提供参考答案，并据此对回答进行打分。这对评判者本身可能缺乏可靠知识的事实型任务尤其有用。</p>

### 14.7.2　位置偏置的缓解 {#position-bias-mitigation}

<p>LLM 评判者会表现出<em>位置偏置</em>（position bias）：系统性地偏好出现在特定位置（首位或末位）的回答。此偏置可能高达 10–15 个百分点。缓解策略：</p>

<ol>
<li><p><strong>交换增强：</strong>对每一对回答都以两种顺序（A vs. B 与 B vs. A）进行评判。判断一致则采纳；判断不一致则记为平局。</p></li>
<li><p><strong>校准式 Prompt：</strong>显式指令评判者：“你的评估不应受到回答呈现顺序的影响。”</p></li>
<li><p><strong>评判者集成：</strong>使用多个评判者并配以不同的位置顺序，再聚合其裁决。</p></li>
<li><p><strong>强制思维链：</strong>要求评判者在给出裁决前先产出详尽理由，以降低对表层位置线索的依赖。</p></li>
</ol>

<div class="admonition warning"><p class="admonition-title">冗长偏置</p>
<p>LLM 评判者同样表现出冗长偏置：更长的回答会被系统性地偏好，即便额外内容是无关或重复的。缓解方式：指示评判者对不必要的长度进行惩罚，并关注信息的质量而非数量。或者，在评判前将回答截断到固定长度。</p>
</div>

### 14.7.3　多评判者面板 {#multi-judge-panels}

<p>单一评判者可能带有系统性偏置。来自不同模型家族的评判者面板能提供更稳健的评估。给定 <span class="arithmatex">\(J\)</span> 位评判者，其裁决为 <span class="arithmatex">\(v_1, \ldots, v_J \in \{A, B, \text{tie}\}\)</span>，面板裁决由多数投票决定。面板一致率为：</p>



<div class="arithmatex">\[\text{Agreement} = \frac{1}{\binom{J}{2}} \sum_{i &lt; j} \mathbf{1}[v_i = v_j]
\tag{14.36}\]</div>

<p>对于三人评判者面板，一致裁决（三人全部同意）视为高置信度；2–1 分裂视为中置信度；三方平局视为低置信度。</p>

### 14.7.4　LLM 评判者的一致性指标 {#agreement-metrics-for-llm-judges}

<p>为验证 LLM 评判者，将其裁决与留出集上的人工标注进行比较。关键指标：</p>

<ul>
<li><p><strong>一致率：</strong>评判者与人类标注一致的样本比例。</p></li>
<li><p><strong>Cohen’s <span class="arithmatex">\(\kappa\)</span>：</strong>经偶然性修正的一致率（公式 <a href="#eq:cohens-kappa">14.1</a>）。</p></li>
<li><p><strong>Spearman’s <span class="arithmatex">\(\rho\)</span>：</strong>评判者分数与人类分数之间的秩相关，适用于序数评分。</p></li>
<li><p><strong>Kendall’s <span class="arithmatex">\(\tau\)</span>：</strong>另一种秩相关，对平局更具鲁棒性。</p></li>
</ul>

<p>若评判者在一份具有代表性的样本上与人类标注者达到 <span class="arithmatex">\(\kappa &gt; 0.6\)</span> 且一致率 <span class="arithmatex">\(&gt; 80\%\)</span>，则视为可靠。</p>

### 14.7.5　G-Eval 框架 {#g-eval-framework}

<p>G-Eval <a class="book-citation" id="cite-liu2023geval-1" href="../../references.md#ref-liu2023geval" aria-label="参考文献 231">[231]</a> 是一个面向 LLM 评估的结构化框架，使用思维链（Chain-of-Thought，CoT）提示与 Token 概率加权来产出更可靠的分数。框架如下：</p>

<ol>
<li><p><strong>生成评估步骤：</strong>提示 LLM 为评估任务生成详尽的评分细则（例如“列出你评估一篇摘要连贯性时所采取的步骤”）。</p></li>
<li><p><strong>以概率加权方式打分：</strong>对每个分值 <span class="arithmatex">\(s \in \{1, 2, 3, 4, 5\}\)</span>，从评判模型获取对数概率 <span class="arithmatex">\(\log P_\theta(s \mid \text{prompt, steps, response})\)</span>。最终分数为以概率加权的平均：</p>



<div class="arithmatex">\[\text{G-Eval score} = \sum_{s=1}^{5} s \cdot \frac{e^{\log P_\theta(s)}}{\sum_{s&#x27;=1}^{5} e^{\log P_\theta(s&#x27;)}}
\tag{14.37}\]</div></li>
<li><p><strong>归一化：</strong>将分数除以最大分值，映射到 <span class="arithmatex">\([0, 1]\)</span>。</p></li>
</ol>

<p>G-Eval 与人类判断的相关性高于直接提示，尤其在连贯性与一致性这类细腻维度上，因为概率加权捕捉了评判者的不确定性，而非强制其作离散选择。</p>

<div class="admonition tip"><p class="admonition-title">为何 G-Eval 有效</p>
<p>标准提示要求评判者输出单个 Token（例如&quot;4&quot;），这丢弃了模型的不确定性。G-Eval 读取所有分值 Token 上的概率分布，实质上是在评判者的信念下计算期望分数。这类似于使用后验分布的均值而非众数。</p>
</div>

## 14.8　评估陷阱 {#subsec:eval-pitfalls}

<p>即便是精心设计的评估流水线也可能给出误导性的结果。本节梳理最常见的失效模式。</p>

### 14.8.1　基准污染 {#benchmark-contamination}

<p>当评估数据出现在模型的训练集中时（直接以原文出现，或间接以复述/语义相似形式出现），就发生了基准污染（benchmark contamination）。被污染的模型会取得虚高的分数，无法反映其真实的泛化能力。</p>

##### 检测方法： {#detection-methods}

<ul>
<li><p><strong><span class="arithmatex">\(n\)</span>-gram 重叠：</strong>计算与训练语料具有高 <span class="arithmatex">\(n\)</span>-gram 重叠（例如 ROUGE-L <span class="arithmatex">\(&gt; 0.8\)</span>）的评估样本比例。</p></li>
<li><p><strong>成员推断（Membership Inference）：</strong>使用成员推断攻击来估计每个评估样本曾出现在训练集中的概率。</p></li>
<li><p><strong>金丝雀串（Canary strings）：</strong>在评估样本中嵌入独特的随机生成字符串，并检查模型是否能将其补全。</p></li>
<li><p><strong>时间留出：</strong>使用模型训练截止日期之后产生的评估数据。</p></li>
</ul>

##### 缓解： {#mitigation}

<ul>
<li><p>维护一个永不公开发布的私有测试集。</p></li>
<li><p>定期用新样本刷新基准。</p></li>
<li><p>报告训练数据截止日期以及去污染流程。</p></li>
</ul>

### 14.8.2　对基准的过拟合 {#overfitting-to-benchmarks}

<p>即使没有直接污染，模型也可能通过反复评估与超参调优而被隐式地为特定基准而优化。这是一种<em>自适应过拟合</em>（adaptive overfitting）：基准向模型开发决策泄漏了信息。</p>

<div class="admonition warning"><p class="admonition-title">基准的生命周期</p>
<p>基准的效用会随着研究社区对其的优化而随时间退化。MMLU <a class="book-citation" id="cite-hendrycks2021measuring-1" href="../../references.md#ref-hendrycks2021measuring" aria-label="参考文献 137">[137]</a> 曾是对世界知识的挑战性测试，如今模型在其上已接近人类水平，但这些模型在新的知识任务上仍会失败。新基准应被视为临时的信号源，而非永久的金标准。</p>
</div>

### 14.8.3　评估中的 Goodhart 定律 {#goodharts-law-in-evaluation}

<p>Goodhart 定律指出：<em>“当一个度量成为目标时，它就不再是好的度量。”</em> <a class="book-citation" id="cite-goodhart1984problems-1" href="../../references.md#ref-goodhart1984problems" aria-label="参考文献 112">[112]</a> 在 LLM 评估中，这有多种体现：</p>

<ul>
<li><p><strong>Reward 欺骗（Reward hacking）：</strong>用基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）训练的模型可能学会利用 Reward 模型，而非真正提升质量。模型可能学会输出冗长、听起来自信但实际上事实错误的回答，这类回答在 Reward 模型上得分却很高。</p></li>
<li><p><strong>指标博弈：</strong>为最大化 BLEU 或 ROUGE 而微调的模型，可能产出在这些指标上得分良好但对人类用处不大的输出。</p></li>
<li><p><strong>评判者博弈：</strong>用 LLM-as-Judge 反馈训练的模型可能学到评判者的偏置（例如冗长偏置），而非真正提升质量。</p></li>
</ul>

<div class="admonition note"><p class="admonition-title">抵御 Goodhart 定律的策略</p>
<ol>
<li><p><strong>指标多样化：</strong>使用来自不同家族的多个指标；一个能博弈某个指标的模型，未必能同时博弈所有指标。</p></li>
<li><p><strong>留出评估：</strong>保留一些不参与训练或模型选择的评估指标。</p></li>
<li><p><strong>人工抽查：</strong>定期采样模型输出供人工审阅，独立于自动化指标之外。</p></li>
<li><p><strong>对抗性评估：</strong>主动探测自动化指标会漏掉的失效模式。</p></li>
<li><p><strong>外在验证：</strong>定期用外在结果来验证内在指标。</p></li>
</ol>
</div>

### 14.8.4　其他陷阱 {#additional-pitfalls}

##### Prompt 敏感性。 {#prompt-sensitivity.}

<p>LLM 的性能可能因评估 Prompt 的微小改动而剧烈变化（例如加入&quot;Think step by step&quot;或更改回答格式）。务必报告所使用的确切 Prompt，并考虑在多种 Prompt 变体上进行评估。</p>

##### 聚合伪影。 {#aggregation-artefacts.}

<p>在难度水平与分数分布各异的任务上对分数取平均，可能产生误导性的聚合指标。一个在简单任务上表现优异但在困难任务上失败的模型，可能与一个表现均衡的模型获得相同的平均分。</p>

##### 人工评估中的选择偏差。 {#selection-bias-in-human-evaluation.}

<p>人类评估者并不是终端用户的随机样本。众包平台上的标注者与目标用户群体可能在偏好、文化背景与领域知识上存在差异。</p>

##### 评估—部署失配。 {#evaluationdeployment-mismatch.}

<p>评估 Prompt 往往比真实用户查询更短、更干净、形式也更规整。在基准 Prompt 上表现良好的模型，在生产环境中遇到嘈杂、含糊的多轮对话（Multi-Turn）时可能显著退化。</p>

<div class="admonition question"><p class="admonition-title">评估设计的关键问题</p>
<p>在部署评估流水线之前，请自问：</p>

<ol>
<li><p>评估指标是否与部署目标对齐？</p></li>
<li><p>评估数据是否代表目标分布？</p></li>
<li><p>是否已评估污染与过拟合风险？</p></li>
<li><p>是否对所有指标都报告了置信区间？</p></li>
<li><p>评估是否可复现（固定随机种子、Prompt 版本化、公开测试集）？</p></li>
<li><p>评估是否已与人类判断或外在结果进行过验证？</p></li>
</ol>
</div>
