# 29.8　RAG 流水线公式 {#rag-pipeline-formulas}



<div class="arithmatex">\[\begin{align*}
\text{Cosine similarity:}&amp;\quad \text{sim}(q, d) = \frac{q \cdot d}{\|q\| \cdot \|d\|} \tag{29.17}\\

\text{Retrieval:}&amp;\quad \mathcal{D}_k = \text{top-}k_{d \in \mathcal{C}} \; \text{sim}(\text{embed}(q),\; \text{embed}(d)) \tag{29.18}\\

\text{RAG generation:}&amp;\quad P(y|q) = P_\text{LLM}(y \;|\; q, \mathcal{D}_k) \tag{29.19}\\

\text{Chunking overlap:}&amp;\quad \text{stride} = \text{chunk\_size} - \text{overlap} \tag{29.20}\\

\text{Reranker (cross-enc):}&amp;\quad \text{score}(q, d) = \text{MLP}(\text{BERT}([q; d]))
\tag{29.21}
\end{align*}\]</div>
