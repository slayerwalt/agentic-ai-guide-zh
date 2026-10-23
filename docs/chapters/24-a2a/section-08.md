# 24.8　多 Agent 系统中的安全与信任 {#sec:a2a:security}

<p>多 Agent 系统带来了独特的安全挑战。当 Agent A 委派给 Agent B、Agent B 又委派给 Agent C 时，信任链必须被精心管理。</p>

## 24.8.1　Agent 身份验证 {#agent-identity-verification}

<p>每个 Agent 都必须具有可验证的身份。可选方案包括：</p>

<ul>
<li><p>由可信身份提供方签发的 <strong>JWT 令牌</strong> <a class="book-citation" id="cite-rfc7519-1" href="../../references.md#ref-rfc7519" aria-label="参考文献 172">[172]</a>，携带 Agent ID、颁发者和过期时间。接收方 Agent 使用提供方的公钥进行验证。</p></li>
<li><p>由内部 CA 颁发的 <strong>mTLS 证书</strong> <a class="book-citation" id="cite-rfc8705-1" href="../../references.md#ref-rfc8705" aria-label="参考文献 36">[36]</a>，同时提供认证与传输加密。</p></li>
<li><p>适用于不存在单一可信权威的跨组织场景的 <strong>去中心化标识符（Decentralized Identifiers, DIDs）</strong> <a class="book-citation" id="cite-w3c-did-2022-1" href="../../references.md#ref-w3c-did-2022" aria-label="参考文献 61">[61]</a>。</p></li>
</ul>

## 24.8.2　消息完整性与加密 {#message-integrity-and-encryption}

<ul>
<li><p>所有 A2A 通信都应通过 <strong>TLS 1.3</strong> <a class="book-citation" id="cite-rfc8446-1" href="../../references.md#ref-rfc8446" aria-label="参考文献 307">[307]</a> 进行，以防止窃听和中间人攻击。</p></li>
<li><p>对于敏感负载，<strong>端到端加密</strong>（例如 JWE）可以确保中间基础设施（负载均衡器、代理）无法读取消息内容。</p></li>
<li><p><strong>消息签名</strong>（JWS）提供不可否认性：接收方 Agent 可以证明某条特定消息确实来自某位特定发送者。</p></li>
</ul>

## 24.8.3　授权 scope {#authorization-scopes}

<p>并非每个 Agent 都应该能够要求其他任何 Agent 做任何事。OAuth 2.0 授权 scope <a class="book-citation" id="cite-rfc6749-1" href="../../references.md#ref-rfc6749" aria-label="参考文献 133">[133]</a> 定义了边界：</p>

<div class="highlight"><pre><span></span><code><span class="c1"># DataAgent 的 OAuth 2.0 scope 示例</span>
<span class="n">SCOPES</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s2">"data:read"</span><span class="p">:</span>        <span class="s2">"Read data from connected databases"</span><span class="p">,</span>
    <span class="s2">"data:write"</span><span class="p">:</span>       <span class="s2">"Write or modify data in connected databases"</span><span class="p">,</span>
    <span class="s2">"data:export"</span><span class="p">:</span>      <span class="s2">"Export data to external systems"</span><span class="p">,</span>
    <span class="s2">"analysis:run"</span><span class="p">:</span>     <span class="s2">"Execute statistical analyses"</span><span class="p">,</span>
    <span class="s2">"analysis:schedule"</span><span class="p">:</span><span class="s2">"Schedule recurring analyses"</span><span class="p">,</span>
    <span class="s2">"admin:config"</span><span class="p">:</span>     <span class="s2">"Modify agent configuration"</span>
<span class="p">}</span>

<span class="c1"># ReportingAgent 可能仅持有：data:read, analysis:run</span>
<span class="c1"># ETL 流水线 Agent 可能持有：data:read, data:write, data:export</span>
<span class="c1"># 只有人类管理员持有：admin:config</span>

<span class="k">class</span><span class="w"> </span><span class="nc">A2AServer</span><span class="p">:</span>
    <span class="k">def</span><span class="w"> </span><span class="nf">verify_authorization</span><span class="p">(</span><span class="bp">self</span><span class="p">,</span> <span class="n">token</span><span class="p">:</span> <span class="nb">str</span><span class="p">,</span> <span class="n">required_scope</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">bool</span><span class="p">:</span>
<span class="w">        </span><span class="sd">"""验证调用方 Agent 是否持有所需 scope。"""</span>
        <span class="n">claims</span> <span class="o">=</span> <span class="n">jwt</span><span class="o">.</span><span class="n">decode</span><span class="p">(</span><span class="n">token</span><span class="p">,</span> <span class="bp">self</span><span class="o">.</span><span class="n">public_key</span><span class="p">,</span> <span class="n">algorithms</span><span class="o">=</span><span class="p">[</span><span class="s2">"RS256"</span><span class="p">])</span>
        <span class="n">granted_scopes</span> <span class="o">=</span> <span class="n">claims</span><span class="o">.</span><span class="n">get</span><span class="p">(</span><span class="s2">"scope"</span><span class="p">,</span> <span class="s2">""</span><span class="p">)</span><span class="o">.</span><span class="n">split</span><span class="p">()</span>
        <span class="k">if</span> <span class="n">required_scope</span> <span class="ow">not</span> <span class="ow">in</span> <span class="n">granted_scopes</span><span class="p">:</span>
            <span class="k">raise</span> <span class="ne">PermissionError</span><span class="p">(</span>
                <span class="sa">f</span><span class="s2">"Caller lacks required scope '</span><span class="si">{</span><span class="n">required_scope</span><span class="si">}</span><span class="s2">'. "</span>
                <span class="sa">f</span><span class="s2">"Granted: </span><span class="si">{</span><span class="n">granted_scopes</span><span class="si">}</span><span class="s2">"</span>
            <span class="p">)</span>
        <span class="k">return</span> <span class="kc">True</span>
</code></pre></div>

## 24.8.4　审计轨迹与问责 {#audit-trails-and-accountability}

<div class="admonition warning"><p class="admonition-title">问责鸿沟</p>
<p>在一条 Agent 委派链中，<em>谁</em>应对某个行为负责可能会变得模糊。如果 Agent A 让 Agent B 删除一个文件，而 Agent B 照做了，那么应由谁负责？每一次 A2A 交互都必须记录：调用方 Agent 的身份、任务描述、所用授权令牌、时间戳以及结果。这种审计轨迹对于事件响应、合规和调试都至关重要。</p>
</div>

<p>每个 A2A 服务器都应输出结构化的审计日志：</p>

<div class="highlight"><pre><span></span><code><span class="nd">@dataclass</span>
<span class="k">class</span><span class="w"> </span><span class="nc">A2AAuditEvent</span><span class="p">:</span>
    <span class="n">timestamp</span><span class="p">:</span> <span class="nb">str</span>          <span class="c1"># ISO 8601</span>
    <span class="n">workflow_id</span><span class="p">:</span> <span class="nb">str</span>        <span class="c1"># 顶层工作流的关联 ID</span>
    <span class="n">span_id</span><span class="p">:</span> <span class="nb">str</span>            <span class="c1"># 当前任务的 span</span>
    <span class="n">parent_span_id</span><span class="p">:</span> <span class="nb">str</span>     <span class="c1"># 调用任务的 span（用于委派链）</span>
    <span class="n">caller_agent_id</span><span class="p">:</span> <span class="nb">str</span>    <span class="c1"># 经验证的调用方 Agent 身份</span>
    <span class="n">callee_agent_id</span><span class="p">:</span> <span class="nb">str</span>    <span class="c1"># 当前 Agent 的身份</span>
    <span class="n">task_id</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">skill_invoked</span><span class="p">:</span> <span class="nb">str</span>
    <span class="n">authorization_scopes</span><span class="p">:</span> <span class="nb">list</span><span class="p">[</span><span class="nb">str</span><span class="p">]</span>
    <span class="n">outcome</span><span class="p">:</span> <span class="nb">str</span>            <span class="c1"># "completed" | "failed" | "rejected"</span>
    <span class="n">duration_ms</span><span class="p">:</span> <span class="nb">int</span>
    <span class="n">error_code</span><span class="p">:</span> <span class="nb">str</span> <span class="o">|</span> <span class="kc">None</span>
</code></pre></div>
