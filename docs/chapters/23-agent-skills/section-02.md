# 23.2　Skill 架构模式 {#skill-architecture-patterns}

## 23.2.1　静态 Skill 加载 {#static-skill-loading}

<p>最简单的模式：根据配置在 Agent 初始化时加载 skill。Agent 始终可以访问其全部 skill。</p>

<div class="highlight"><pre><span></span><code><span class="c1"># 伪代码 —— 与具体框架无关的模式</span>
<span class="n">agent</span> <span class="o">=</span> <span class="n">Agent</span><span class="p">(</span>
    <span class="n">model</span><span class="o">=</span><span class="s2">"claude-sonnet-4-20250514"</span><span class="p">,</span>
    <span class="n">skills</span><span class="o">=</span><span class="p">[</span><span class="s2">"code-review"</span><span class="p">,</span> <span class="s2">"documentation"</span><span class="p">,</span> <span class="s2">"testing"</span><span class="p">],</span>
    <span class="c1"># 每个 skill 都会向 Agent 添加 prompts、tools 与知识</span>
<span class="p">)</span>
</code></pre></div>

<p><strong>优点：</strong>简单、可预测、低延迟。<br><strong>缺点：</strong>未被使用的 skill 会浪费上下文窗口；无法扩展到数百个 skill。</p>

## 23.2.2　动态 Skill 发现 {#dynamic-skill-discovery}

<p>Agent 根据当前任务选择激活哪些 skill。一个 skill 路由器（通常是一个轻量分类器或基于 embedding 的匹配器）来判断相关性：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># 伪代码 —— 与具体框架无关的模式</span>
<span class="n">relevant_skills</span> <span class="o">=</span> <span class="n">skill_router</span><span class="o">.</span><span class="n">match</span><span class="p">(</span>
    <span class="n">user_request</span><span class="o">=</span><span class="n">message</span><span class="p">,</span>
    <span class="n">available_skills</span><span class="o">=</span><span class="n">skill_registry</span><span class="p">,</span>
    <span class="n">max_skills</span><span class="o">=</span><span class="mi">3</span>
<span class="p">)</span>
<span class="n">agent</span><span class="o">.</span><span class="n">activate</span><span class="p">(</span><span class="n">relevant_skills</span><span class="p">)</span>
</code></pre></div>

<p><strong>优点：</strong>可扩展到大型 skill 库；上下文效率高。<br><strong>缺点：</strong>路由错误可能漏掉相关 skill；引入额外延迟。</p>

## 23.2.3　层级化 Skill 组合 {#hierarchical-skill-composition}

<p>Skill 可以依赖于其他 skill，从而构成一个有向无环图（DAG）。一个高层 skill（例如“部署应用”）可能会调用一些子 skill（“跑测试”、“构建 Docker 镜像”、“更新 DNS”）：</p>

<ul>
<li><p>Skill 显式声明其依赖</p></li>
<li><p>编排器在执行前解析依赖图</p></li>
<li><p>子 skill 可被多个父 skill 共享</p></li>
</ul>
