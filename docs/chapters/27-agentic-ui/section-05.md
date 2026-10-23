# 27.5　生成式 UI {#subsec:generative-ui}

<div class="admonition tip"><p class="admonition-title">生成式 UI 概念</p>
<p>传统的 LLM 界面将模型输出渲染为文本或 markdown。<em>生成式 UI</em>（generative UI）颠覆了这一模式：模型的工具调用<em>生成</em> UI 组件。模型不仅决定<em>说什么</em>，还根据内容类型与用户意图决定<em>如何呈现</em>——以图表、表单、地图或日历小部件的形式。</p>
</div>

<p>生成式 UI 标志着 LLM 与界面关系的根本转变。开发者不再预先规定所有可能的 UI 状态，而是由模型动态选择并参数化适合当前上下文的 UI 组件。</p>

## 27.5.1　用 React Server Components 构建动态界面 {#react-server-components-for-dynamic-interfaces}

<p>Vercel AI SDK 的 RSC（React Server Components10）集成是生成式 UI 最成熟的实现。其架构如下：</p>

<ol>
<li><p>用户向 Next.js11 的 server action 发送消息。</p></li>
<li><p>服务端以一组工具调用 LLM，每个工具关联一个 React 组件。</p></li>
<li><p>当 LLM 调用某个工具（例如 <code>show_weather</code>）时，服务端以工具输出作为 props 渲染相应的 React 组件。</p></li>
<li><p>渲染后的组件作为 React Server Component 流式发送给客户端，在聊天中内联出现。</p></li>
</ol>

<p class="code-caption"><span class="code-number">代码 27.2　</span>基于 Vercel AI SDK RSC 的生成式 UI（TypeScript）</p>
<div class="highlight"><pre><span></span><code><span class="c1">// app/actions.tsx (Server Action)</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">streamUI</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'ai/rsc'</span><span class="p">;</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">openai</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'@ai-sdk/openai'</span><span class="p">;</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">WeatherCard</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'@/components/WeatherCard'</span><span class="p">;</span>
<span class="k">import</span><span class="w"> </span><span class="p">{</span><span class="w"> </span><span class="nx">StockChart</span><span class="w"> </span><span class="p">}</span><span class="w"> </span><span class="kr">from</span><span class="w"> </span><span class="s1">'@/components/StockChart'</span><span class="p">;</span>

<span class="k">export</span><span class="w"> </span><span class="k">async</span><span class="w"> </span><span class="kd">function</span><span class="w"> </span><span class="nx">chat</span><span class="p">(</span><span class="nx">userMessage</span><span class="o">:</span><span class="w"> </span><span class="kt">string</span><span class="p">)</span><span class="w"> </span><span class="p">{</span>
<span class="w">  </span><span class="kd">const</span><span class="w"> </span><span class="nx">result</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">await</span><span class="w"> </span><span class="nx">streamUI</span><span class="p">({</span>
<span class="w">    </span><span class="nx">model</span><span class="o">:</span><span class="w"> </span><span class="kt">openai</span><span class="p">(</span><span class="s1">'gpt-4o'</span><span class="p">),</span>
<span class="w">    </span><span class="nx">messages</span><span class="o">:</span><span class="w"> </span><span class="p">[{</span><span class="w"> </span><span class="nx">role</span><span class="o">:</span><span class="w"> </span><span class="s1">'user'</span><span class="p">,</span><span class="w"> </span><span class="nx">content</span><span class="o">:</span><span class="w"> </span><span class="kt">userMessage</span><span class="w"> </span><span class="p">}],</span>
<span class="w">    </span><span class="nx">tools</span><span class="o">:</span><span class="w"> </span><span class="p">{</span>
<span class="w">      </span><span class="nx">show_weather</span><span class="o">:</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="nx">description</span><span class="o">:</span><span class="w"> </span><span class="s1">'Display current weather for a location'</span><span class="p">,</span>
<span class="w">        </span><span class="nx">parameters</span><span class="o">:</span><span class="w"> </span><span class="kt">z.object</span><span class="p">({</span>
<span class="w">          </span><span class="nx">location</span><span class="o">:</span><span class="w"> </span><span class="kt">z.string</span><span class="p">(),</span>
<span class="w">          </span><span class="nx">unit</span><span class="o">:</span><span class="w"> </span><span class="kt">z.enum</span><span class="p">([</span><span class="s1">'celsius'</span><span class="p">,</span><span class="w"> </span><span class="s1">'fahrenheit'</span><span class="p">]),</span>
<span class="w">        </span><span class="p">}),</span>
<span class="w">        </span><span class="c1">// 工具结果作为 React 组件渲染</span>
<span class="w">        </span><span class="c1">// Tool result rendered as a React component</span>
<span class="w">        </span><span class="nx">generate</span><span class="o">:</span><span class="w"> </span><span class="kt">async</span><span class="w"> </span><span class="p">({</span><span class="w"> </span><span class="nx">location</span><span class="p">,</span><span class="w"> </span><span class="nx">unit</span><span class="w"> </span><span class="p">})</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">          </span><span class="kd">const</span><span class="w"> </span><span class="nx">data</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">await</span><span class="w"> </span><span class="nx">fetchWeather</span><span class="p">(</span><span class="nx">location</span><span class="p">,</span><span class="w"> </span><span class="nx">unit</span><span class="p">);</span>
<span class="w">          </span><span class="k">return</span><span class="w"> </span><span class="p">&lt;</span><span class="nt">WeatherCard</span><span class="w"> </span><span class="na">data</span><span class="o">=</span><span class="p">{</span><span class="nx">data</span><span class="p">}</span><span class="w"> </span><span class="p">/&gt;;</span>
<span class="w">        </span><span class="p">},</span>
<span class="w">      </span><span class="p">},</span>
<span class="w">      </span><span class="nx">show_stock</span><span class="o">:</span><span class="w"> </span><span class="p">{</span>
<span class="w">        </span><span class="nx">description</span><span class="o">:</span><span class="w"> </span><span class="s1">'Display stock price chart'</span><span class="p">,</span>
<span class="w">        </span><span class="nx">parameters</span><span class="o">:</span><span class="w"> </span><span class="kt">z.object</span><span class="p">({</span><span class="w"> </span><span class="nx">ticker</span><span class="o">:</span><span class="w"> </span><span class="kt">z.string</span><span class="p">()</span><span class="w"> </span><span class="p">}),</span>
<span class="w">        </span><span class="nx">generate</span><span class="o">:</span><span class="w"> </span><span class="kt">async</span><span class="w"> </span><span class="p">({</span><span class="w"> </span><span class="nx">ticker</span><span class="w"> </span><span class="p">})</span><span class="w"> </span><span class="p">=&gt;</span><span class="w"> </span><span class="p">{</span>
<span class="w">          </span><span class="kd">const</span><span class="w"> </span><span class="nx">data</span><span class="w"> </span><span class="o">=</span><span class="w"> </span><span class="k">await</span><span class="w"> </span><span class="nx">fetchStockData</span><span class="p">(</span><span class="nx">ticker</span><span class="p">);</span>
<span class="w">          </span><span class="k">return</span><span class="w"> </span><span class="p">&lt;</span><span class="nt">StockChart</span><span class="w"> </span><span class="na">ticker</span><span class="o">=</span><span class="p">{</span><span class="nx">ticker</span><span class="p">}</span><span class="w"> </span><span class="na">data</span><span class="o">=</span><span class="p">{</span><span class="nx">data</span><span class="p">}</span><span class="w"> </span><span class="p">/&gt;;</span>
<span class="w">        </span><span class="p">},</span>
<span class="w">      </span><span class="p">},</span>
<span class="w">    </span><span class="p">},</span>
<span class="w">  </span><span class="p">});</span>
<span class="w">  </span><span class="k">return</span><span class="w"> </span><span class="nx">result</span><span class="p">.</span><span class="nx">value</span><span class="p">;</span>
<span class="p">}</span>
</code></pre></div>

## 27.5.2　基于内容类型的自适应界面 {#adaptive-interfaces-based-on-content-type}

<p>生成式 UI 使界面能够根据所呈现内容的性质自适应：</p>

<ul>
<li><p><strong>表格数据</strong> <span class="arithmatex">\(\rightarrow\)</span> 可排序、可过滤、带导出选项的数据表。</p></li>
<li><p><strong>地理数据</strong> <span class="arithmatex">\(\rightarrow\)</span> 带标记与图层的交互式地图。</p></li>
<li><p><strong>时间序列</strong> <span class="arithmatex">\(\rightarrow\)</span> 可缩放、带注释的折线图。</p></li>
<li><p><strong>代码</strong> <span class="arithmatex">\(\rightarrow\)</span> 带语法高亮与运行按钮的编辑器。</p></li>
<li><p><strong>文档</strong> <span class="arithmatex">\(\rightarrow\)</span> 带注释工具的格式化文档查看器。</p></li>
<li><p><strong>表单/结构化输入</strong> <span class="arithmatex">\(\rightarrow\)</span> 动态生成的表单字段。</p></li>
</ul>

<p>模型在此扮演<em>UI 编排者</em>（UI orchestrator）的角色，为每条信息选择最合适的呈现方式。这降低了开发者预判所有可能输出类型并预先构建相应组件的需求。</p>

<div class="admonition question"><p class="admonition-title">生成式 UI 的边界</p>
<p>应该将多少 UI 生成委托给模型？完全由模型驱动的 UI 存在不一致性、无障碍访问失败与安全漏洞的风险（例如，模型生成一个把数据提交到非预期端点的表单）。实践中，当模型从一个由预构建、可访问且安全组件组成的<em>精选库</em>中进行选择，而非生成任意 HTML 或 JSX 时，生成式 UI 的效果最佳。</p>
</div>
