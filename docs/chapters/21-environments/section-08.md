# 21.8　代码示例：最小的自定义 LLM Agent 环境 {#sec:code-example}

<div class="admonition example"><p class="admonition-title">用于 LLM Agent 训练的最小自定义环境</p>
<p>以下 Python 类实现了一个文件编辑环境，Agent 必须修改一个 Python 文件以使一个失败的测试通过。它遵循为 LLM Agent 适配过的 Gymnasium API。</p>
</div>

<p class="code-caption"><span class="code-number">代码 21.1　</span>遵循 Gymnasium API 的最小 LLM Agent 环境。</p>
<div class="highlight"><pre><span></span><code><span class="sd">"""</span>
<span class="sd">minimal_env.py  --  用于 LLM Agent 的最小文件编辑环境。</span>

<span class="sd">Agent 接收一个含 bug 的 Python 文件以及一个失败的测试。</span>
<span class="sd">它必须不断编辑该文件直到测试通过。</span>
<span class="sd">奖励：所有测试通过则为 1.0，否则为 0.0。</span>
<span class="sd">"""</span>

<span class="kn">from</span><span class="w"> </span><span class="nn">__future__</span><span class="w"> </span><span class="kn">import</span> <span class="n">annotations</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">subprocess</span><span class="o">,</span><span class="w"> </span><span class="nn">shutil</span><span class="o">,</span><span class="w"> </span><span class="nn">tempfile</span><span class="o">,</span><span class="w"> </span><span class="nn">textwrap</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">pathlib</span><span class="w"> </span><span class="kn">import</span> <span class="n">Path</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">dataclasses</span><span class="w"> </span><span class="kn">import</span> <span class="n">dataclass</span><span class="p">,</span> <span class="n">field</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">typing</span><span class="w"> </span><span class="kn">import</span> <span class="n">Any</span>

<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># 数据结构</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">StepResult</span><span class="p">:</span>
    <span class="n">observation</span><span class="p">:</span> <span class="nb">str</span>          <span class="c1"># 输入给 LLM 的文本</span>
    <span class="n">reward</span><span class="p">:</span> <span class="nb">float</span>             <span class="c1"># 0.0 或 1.0</span>
    <span class="n">terminated</span><span class="p">:</span> <span class="nb">bool</span>          <span class="c1"># Episode 结束（任务解决或达到最大步数）</span>
    <span class="n">truncated</span><span class="p">:</span> <span class="nb">bool</span>           <span class="c1"># Episode 被截断（预算超出）</span>
    <span class="n">info</span><span class="p">:</span> <span class="nb">dict</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="n">Any</span><span class="p">]</span> <span class="o">=</span> <span class="n">field</span><span class="p">(</span><span class="n">default_factory</span><span class="o">=</span><span class="nb">dict</span><span class="p">)</span>

<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># 环境</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">class</span><span class="w"> </span><span class="nc">FileEditEnv</span><span class="p">:</span>
<span class="w">    </span><span class="sd">"""</span>
<span class="sd">    一个 Gymnasium 风格的 LLM 代码修复环境。</span>

<span class="sd">    观察空间 : str  （文件内容 + 测试输出）</span>
<span class="sd">    动作空间 : str  （以下之一：view、edit、run_tests、submit）</span>
<span class="sd">    奖励     : 所有测试通过为 1.0，否则为 0.0</span>
<span class="sd">    """</span>

    <span class="n">MAX_STEPS</span> <span class="o">=</span> <span class="mi">20</span>          <span class="c1"># Episode 硬上限</span>
    <span class="n">TIMEOUT</span>   <span class="o">=</span> <span class="mi">30</span>          <span class="c1"># 每次测试运行的秒数</span>

    <span class="k">def</span><span class="w"> </span><span class="fm">__init__</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">buggy_code</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">test_code</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                 <span class="n">task_description</span><span class="p">:</span> <span class="nb">str</span><span class="p">):</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">buggy_code</span>       <span class="o">=</span> <span class="n">buggy_code</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">test_code</span>        <span class="o">=</span> <span class="n">test_code</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">task_description</span> <span class="o">=</span> <span class="n">task_description</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="p">:</span> <span class="n">Path</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span> <span class="o">=</span> <span class="mi">0</span>

    <span class="c1"># ------------------------------------------------------------------</span>
    <span class="c1"># 核心 API</span>
    <span class="c1"># ------------------------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">reset</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">seed</span><span class="p">:</span> <span class="nb">int</span> <span class="o">|</span> <span class="kc">None</span> <span class="o">=</span> <span class="kc">None</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">tuple</span><span class="p">[</span><span class="nb">str</span><span class="p">,</span> <span class="nb">dict</span><span class="p">]:</span>
<span class="w">        </span><span class="sd">"""初始化一个新 Episode；返回 (observation, info)。"""</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="ow">and</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="o">.</span><span class="n">exists</span><span class="p">():</span>
            <span class="n">shutil</span><span class="o">.</span><span class="n">rmtree</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="p">)</span>

        <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span>    <span class="o">=</span> <span class="n">Path</span><span class="p">(</span><span class="n">tempfile</span><span class="o">.</span><span class="n">mkdtemp</span><span class="p">(</span><span class="n">prefix</span><span class="o">=</span><span class="s2">"fileenv_"</span><span class="p">))</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span> <span class="o">=</span> <span class="mi">0</span>

        <span class="c1"># 写入初始文件</span>
        <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">write_text</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">buggy_code</span><span class="p">)</span>
        <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"test_solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">write_text</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">test_code</span><span class="p">)</span>

        <span class="n">obs</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_build_observation</span><span class="p">(</span>
            <span class="n">action_taken</span><span class="o">=</span><span class="s2">"[Episode start]"</span><span class="p">,</span>
            <span class="n">test_output</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">_run_tests</span><span class="p">()</span>
        <span class="p">)</span>
        <span class="k">return</span> <span class="n">obs</span><span class="p">,</span> <span class="p">{</span><span class="s2">"step"</span><span class="p">:</span> <span class="mi">0</span><span class="p">}</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">step</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">action</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="n">StepResult</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""执行一个 Agent 动作；返回 StepResult。"""</span>
        <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span> <span class="o">+=</span> <span class="mi">1</span>
        <span class="n">action</span> <span class="o">=</span> <span class="n">action</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>

        <span class="c1"># --- 解析并分派动作 ---</span>
        <span class="k">if</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"view"</span><span class="p">):</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_action_view</span><span class="p">()</span>
        <span class="k">elif</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"edit"</span><span class="p">):</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_action_edit</span><span class="p">(</span><span class="n">action</span><span class="p">)</span>
        <span class="k">elif</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"run_tests"</span><span class="p">):</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_run_tests</span><span class="p">()</span>
        <span class="k">elif</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"submit"</span><span class="p">):</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_run_tests</span><span class="p">()</span>
        <span class="k">else</span><span class="p">:</span>
            <span class="n">result_text</span> <span class="o">=</span> <span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Unknown action: </span><span class="si">{</span><span class="n">action</span><span class="si">!r}</span><span class="se">\n</span><span class="s2">"</span>
                <span class="s2">"Valid actions: view | edit &lt;new_content&gt; | "</span>
                <span class="s2">"run_tests | submit"</span>
            <span class="p">)</span>

        <span class="n">test_output</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_run_tests</span><span class="p">()</span>
        <span class="n">passed</span>      <span class="o">=</span> <span class="s2">"passed"</span> <span class="ow">in</span> <span class="n">test_output</span> <span class="ow">and</span> <span class="s2">"failed"</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">test_output</span>
        <span class="n">reward</span>      <span class="o">=</span> <span class="mf">1.0</span> <span class="k">if</span> <span class="n">passed</span> <span class="k">else</span> <span class="mf">0.0</span>
        <span class="n">terminated</span>  <span class="o">=</span> <span class="n">passed</span> <span class="ow">or</span> <span class="n">action</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s2">"submit"</span><span class="p">)</span>
        <span class="n">truncated</span>   <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span> <span class="o">&gt;=</span> <span class="bp">self</span><span class="o">.</span><span class="n">MAX_STEPS</span>

        <span class="n">obs</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">_build_observation</span><span class="p">(</span><span class="n">action</span><span class="p">,</span> <span class="n">test_output</span><span class="p">)</span>
        <span class="k">return</span> <span class="n">StepResult</span><span class="p">(</span><span class="n">obs</span><span class="p">,</span> <span class="n">reward</span><span class="p">,</span> <span class="n">terminated</span><span class="p">,</span> <span class="n">truncated</span><span class="p">,</span>
                          <span class="p">{</span><span class="s2">"step"</span><span class="p">:</span> <span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span><span class="p">,</span>
                           <span class="s2">"passed"</span><span class="p">:</span> <span class="n">passed</span><span class="p">})</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">render</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""返回当前状态的人类可读摘要。"""</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="ow">is</span> <span class="kc">None</span><span class="p">:</span>
            <span class="k">return</span> <span class="s2">"[Environment not initialised]"</span>
        <span class="n">code</span> <span class="o">=</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">read_text</span><span class="p">()</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"=== solution.py ===</span><span class="se">\n</span><span class="si">{</span><span class="n">code</span><span class="si">}</span><span class="se">\n</span><span class="s2">"</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">close</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="kc">None</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""释放资源。"""</span>
        <span class="k">if</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="ow">and</span> <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="o">.</span><span class="n">exists</span><span class="p">():</span>
            <span class="n">shutil</span><span class="o">.</span><span class="n">rmtree</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="p">)</span>
            <span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">=</span> <span class="kc">None</span>

    <span class="c1"># ------------------------------------------------------------------</span>
    <span class="c1"># 私有辅助方法</span>
    <span class="c1"># ------------------------------------------------------------------</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_action_view</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">code</span> <span class="o">=</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">read_text</span><span class="p">()</span>
        <span class="k">return</span> <span class="sa">f</span><span class="s2">"Current solution.py:</span><span class="se">\n</span><span class="s2">```python</span><span class="se">\n</span><span class="si">{</span><span class="n">code</span><span class="si">}</span><span class="se">\n</span><span class="s2">```"</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_action_edit</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">action</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="c1"># 期望格式：edit\n```python\n&lt;code&gt;\n```</span>
        <span class="k">try</span><span class="p">:</span>
            <span class="n">new_code</span> <span class="o">=</span> <span class="n">action</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"```python"</span><span class="p">)[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s2">"```"</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
            <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">write_text</span><span class="p">(</span><span class="n">new_code</span><span class="p">)</span>
            <span class="k">return</span> <span class="s2">"File updated successfully."</span>
        <span class="k">except</span> <span class="ne">IndexError</span><span class="p">:</span>
            <span class="k">return</span> <span class="s2">"Edit failed: wrap new code in ```python ... ```"</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_run_tests</span><span class="p">(</span><span class="bp">self</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">result</span> <span class="o">=</span> <span class="n">subprocess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span>
            <span class="p">[</span><span class="s2">"python"</span><span class="p">,</span> <span class="s2">"-m"</span><span class="p">,</span> <span class="s2">"pytest"</span><span class="p">,</span> <span class="s2">"test_solution.py"</span><span class="p">,</span>
             <span class="s2">"-v"</span><span class="p">,</span> <span class="s2">"--tb=short"</span><span class="p">,</span> <span class="s2">"--no-header"</span><span class="p">],</span>
            <span class="n">cwd</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span><span class="p">,</span>
            <span class="n">capture_output</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">text</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span>
            <span class="n">timeout</span><span class="o">=</span><span class="bp">self</span><span class="o">.</span><span class="n">TIMEOUT</span>
        <span class="p">)</span>
        <span class="k">return</span> <span class="n">result</span><span class="o">.</span><span class="n">stdout</span> <span class="o">+</span> <span class="n">result</span><span class="o">.</span><span class="n">stderr</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_build_observation</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">action_taken</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span>
                           <span class="n">test_output</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">str</span><span class="p">:</span>
        <span class="n">code</span> <span class="o">=</span> <span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">_workdir</span> <span class="o">/</span> <span class="s2">"solution.py"</span><span class="p">)</span><span class="o">.</span><span class="n">read_text</span><span class="p">()</span>
        <span class="k">return</span> <span class="n">textwrap</span><span class="o">.</span><span class="n">dedent</span><span class="p">(</span><span class="sa">f</span><span class="s2">"""</span>
<span class="s2">            TASK: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">task_description</span><span class="si">}</span>
<span class="s2">            STEP: </span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">_step_count</span><span class="si">}</span><span class="s2">/</span><span class="si">{</span><span class="bp">self</span><span class="o">.</span><span class="n">MAX_STEPS</span><span class="si">}</span>

<span class="s2">            --- Last action ---</span>
<span class="s2">            </span><span class="si">{</span><span class="n">action_taken</span><span class="si">}</span>

<span class="s2">            --- Current solution.py ---</span>
<span class="s2">            </span><span class="si">{</span><span class="n">code</span><span class="si">}</span>

<span class="s2">            --- Test output ---</span>
<span class="s2">            </span><span class="si">{</span><span class="n">test_output</span><span class="si">}</span>

<span class="s2">            --- Available actions ---</span>
<span class="s2">            view                          # show current file</span>
<span class="s2">            edit</span><span class="se">\n</span><span class="s2">```python</span><span class="se">\n</span><span class="s2">&lt;code&gt;</span><span class="se">\n</span><span class="s2">```  # replace file contents</span>
<span class="s2">            run_tests                     # run pytest</span>
<span class="s2">            submit                        # finalise and end episode</span>
<span class="s2">        """</span><span class="p">)</span><span class="o">.</span><span class="n">strip</span><span class="p">()</span>

<span class="c1"># ---------------------------------------------------------------------------</span>
<span class="c1"># 使用示例</span>
<span class="c1"># ---------------------------------------------------------------------------</span>

<span class="k">if</span> <span class="vm">__name__</span> <span class="o">==</span> <span class="s2">"__main__"</span><span class="p">:</span>
    <span class="n">BUGGY</span> <span class="o">=</span> <span class="s2">"def add(a, b):</span><span class="se">\n</span><span class="s2">    return a - b</span><span class="se">\n</span><span class="s2">"</span>   <span class="c1"># bug：本应是加号</span>
    <span class="n">TESTS</span> <span class="o">=</span> <span class="p">(</span>
        <span class="s2">"from solution import add</span><span class="se">\n</span><span class="s2">"</span>
        <span class="s2">"def test_add(): assert add(2, 3) == 5</span><span class="se">\n</span><span class="s2">"</span>
    <span class="p">)</span>

    <span class="n">env</span> <span class="o">=</span> <span class="n">FileEditEnv</span><span class="p">(</span><span class="n">BUGGY</span><span class="p">,</span> <span class="n">TESTS</span><span class="p">,</span> <span class="s2">"Fix the add() function."</span><span class="p">)</span>
    <span class="n">obs</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">env</span><span class="o">.</span><span class="n">reset</span><span class="p">(</span><span class="n">seed</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">obs</span><span class="p">)</span>

    <span class="c1"># 模拟一次正确的编辑</span>
    <span class="n">fix</span> <span class="o">=</span> <span class="s2">"edit</span><span class="se">\n</span><span class="s2">```python</span><span class="se">\n</span><span class="s2">def add(a, b):</span><span class="se">\n</span><span class="s2">    return a + b</span><span class="se">\n</span><span class="s2">```"</span>
    <span class="n">result</span> <span class="o">=</span> <span class="n">env</span><span class="o">.</span><span class="n">step</span><span class="p">(</span><span class="n">fix</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"</span><span class="se">\n</span><span class="s2">Reward: </span><span class="si">{</span><span class="n">result</span><span class="o">.</span><span class="n">reward</span><span class="si">}</span><span class="s2">  |  Terminated: </span><span class="si">{</span><span class="n">result</span><span class="o">.</span><span class="n">terminated</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
    <span class="n">env</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</code></pre></div>

<div class="admonition note"><p class="admonition-title">示例环境中的设计决策</p>
<ul>
<li><p><strong>纯文本接口</strong>：观察和动作均为纯字符串，兼容任意 LLM。</p></li>
<li><p><strong>基于执行的奖励</strong>：奖励来自实际运行测试套件，而非 LLM 裁判。这使其抗作弊且完美对齐。</p></li>
<li><p><strong>隔离子进程</strong>：测试在带超时的独立进程中运行，防止死循环拖垮训练循环。</p></li>
<li><p><strong>兼容 Gymnasium</strong>：<code>reset</code>/<code>step</code>/<code>render</code>/<code>close</code> 遵循标准 API，可直接用于 RL 训练框架。</p></li>
</ul>
</div>
