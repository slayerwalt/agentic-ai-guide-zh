# 21.9　主要 Agent 环境对比 {#sec:env-comparison}

<p>表 <a href="#tab:env-comparison">21.2</a> 总结了本节讨论的主要 Agent 环境的关键属性。</p>

<p class="table-caption" id="tab:env-comparison"><span class="table-number">表 21.2　</span>面向 LLM Agent 的主要 Agent 环境对比。&quot;SoTA&quot;指撰写时已发表的最佳 LLM Agent 结果。给出人类性能（若有）。</p>
<div class="table-scroll" role="region" aria-label="表格" tabindex="0">
<table>
<thead><tr><th scope="col"><strong>环境</strong></th><th scope="col"><strong>观察类型</strong></th><th scope="col"><strong>动作空间</strong></th><th scope="col"><strong>领域</strong></th><th scope="col"><strong>任务数</strong></th><th scope="col"><strong>人类</strong></th><th scope="col"><strong>SoTA LLM</strong></th></tr></thead>
<tbody><tr><td>WebArena</td><td>文本 + DOM</td><td>浏览器 API</td><td>Web 导航</td><td>812</td><td>78%</td><td><span class="arithmatex">\(\sim\)</span>45%</td></tr>
<tr><td>VisualWebArena</td><td>截图 + DOM</td><td>浏览器 API</td><td>视觉 Web</td><td>910</td><td>88%</td><td><span class="arithmatex">\(\sim\)</span>35%</td></tr>
<tr><td>Mind2Web</td><td>截图 + DOM</td><td>浏览器 API</td><td>真实网站</td><td>2,000</td><td>—</td><td><span class="arithmatex">\(\sim\)</span>30%</td></tr>
<tr><td>OSWorld</td><td>截图</td><td>鼠标 + 键盘</td><td>桌面 OS</td><td>369</td><td>72%</td><td><span class="arithmatex">\(\sim\)</span>18%</td></tr>
<tr><td>WindowsAgentArena</td><td>截图</td><td>鼠标 + 键盘</td><td>Windows 应用</td><td>154</td><td>75%</td><td><span class="arithmatex">\(\sim\)</span>20%</td></tr>
<tr><td>SWE-bench Verified</td><td>文本（仓库）</td><td>Shell + 编辑器</td><td>代码修复</td><td>500</td><td>100%</td><td><span class="arithmatex">\(\sim\)</span>50%</td></tr>
<tr><td>GAIA (Level 1)</td><td>文本 + 文件</td><td>工具调用</td><td>通用 QA</td><td>165</td><td>92%</td><td><span class="arithmatex">\(\sim\)</span>55%</td></tr>
<tr><td>GAIA (Level 3)</td><td>文本 + 文件</td><td>工具调用</td><td>困难 QA</td><td>42</td><td>92%</td><td><span class="arithmatex">\(\sim\)</span>10%</td></tr>
<tr><td>NetHack (NLE)</td><td>文本 + 字符图</td><td>离散动作</td><td>Roguelike 游戏</td><td>—</td><td><span class="arithmatex">\(&gt;\)</span>10k 分</td><td><span class="arithmatex">\(\sim\)</span>5k 分</td></tr>
<tr><td>Voyager (Minecraft)</td><td>文本 + 代码</td><td>代码执行</td><td>开放世界游戏</td><td>课程</td><td>—</td><td>15+ 科技树</td></tr>
<tr><td>MLAgentBench</td><td>文本 + 代码</td><td>Shell + 编辑器</td><td>ML 工程</td><td>13</td><td>—</td><td><span class="arithmatex">\(\sim\)</span>40%</td></tr></tbody>
</table>
</div>

<div class="admonition tip"><p class="admonition-title">解读对比表</p>
<p>人类性能与 SoTA LLM 性能之间的差距，在<em>计算机使用</em>类任务上最大（OSWorld：72% vs. 18%），在<em>代码修复</em>上最小（SWE-bench：100% vs. 50%）。这一模式反映了动作空间的成熟度：LLM 已在海量代码上训练，但基于截图的交互数据相对较少。随着计算机使用类训练数据的积累，差距有望缩小。</p>
</div>
