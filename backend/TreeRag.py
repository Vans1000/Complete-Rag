import uuid
import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.mixture import GaussianMixture
from Tokenizer import Tokenizer
from VectorDatabase import VectorDatabase
from LLM import BaseLLM


class TreeRAG:
    def __init__(self, tokenizer: Tokenizer, vector_db: VectorDatabase, llm: BaseLLM,
                 max_summary_chars: int = 12000):
        self.tokenizer = tokenizer
        self.vector_db = vector_db
        self.llm = llm
        self.max_level = -1                       
        self.max_summary_chars = max_summary_chars

    def cluster_texts(self, embeddings: np.ndarray,
                      n_clusters: Optional[int] = None) -> List[int]:
        n = len(embeddings)
        if n <= 1:
            return [0] * n

        if n_clusters is None:
            n_clusters = max(1, int(n ** 0.5))
        n_clusters = min(n_clusters, n)

        try:
            gmm = GaussianMixture(
                n_components=n_clusters,
                random_state=42,
                reg_covar=1e-5,        
                max_iter=100,
            )
            return gmm.fit_predict(embeddings).tolist()
        except Exception as e:
            print(f"[TreeRAG] GMM failed ({e}); falling back to a single cluster.")
            return [0] * n

    def _summarize_block(self, texts: List[str]) -> str:
        combined = "\n\n---\n\n".join(texts)
        prompt = (
            "Summarize the key information, main themes, and core details "
            "from the following chunks into a concise, detailed summary:\n\n"
            f"{combined}\n\nSummary:"
        )
        return self.llm.chat([{"role": "user", "content": prompt}], stream=False)

    def summarize_cluster(self, texts: List[str]) -> str:
        blocks, current, cur_len = [], [], 0
        for t in texts:
            if cur_len + len(t) > self.max_summary_chars and current:
                blocks.append(current)
                current, cur_len = [], 0
            current.append(t)
            cur_len += len(t)
        if current:
            blocks.append(current)

        if len(blocks) == 1:
            return self._summarize_block(blocks[0])

        partials = [self._summarize_block(b) for b in blocks]
        if sum(len(p) for p in partials) > self.max_summary_chars:
            return self.summarize_cluster(partials)
        return self._summarize_block(partials)

    def build_tree_and_ingest(self, leaf_chunks: List[Dict], max_depth: int = 3):
        if not leaf_chunks:
            return

        current_nodes = leaf_chunks

        for node in current_nodes:
            node['level'] = 0
            node['is_leaf'] = True
            node['type'] = node.get('type', 'text')   
            self._upload_node(node)

        current_level = 0
        while current_level < max_depth and len(current_nodes) > 1:
            current_level += 1
            texts = [n['text'] for n in current_nodes]

            if all(n.get('dense_vector') is not None for n in current_nodes):
                emb = np.stack([
                    n['dense_vector'].cpu().numpy().squeeze()
                    for n in current_nodes
                ])
            else:
                _, emb_t = self.tokenizer.BatchDenseVectorRetrieval(text_lists=texts)
                emb = emb_t.numpy()

            labels = self.cluster_texts(emb)
            num_clusters = len(set(labels))

            if num_clusters == 1 and len(current_nodes) <= 2:
                current_level -= 1
                break

            next_level = []
            for cid in range(num_clusters):
                idxs = [i for i, l in enumerate(labels) if l == cid]
                if not idxs:
                    continue

                cluster_texts = [texts[i] for i in idxs]
                summary = self.summarize_cluster(cluster_texts)

                child_ids = [
                    current_nodes[i]['node_id']
                    for i in idxs
                    if current_nodes[i].get('node_id')
                ]
                sample_source = current_nodes[idxs[0]].get('path', 'tree_summary')

                node = {
                    'text':       summary,
                    'type':       'summary',
                    'level':      current_level,
                    'is_leaf':    False,
                    'path':       sample_source,
                    'child_ids':  child_ids,
                }
                self._upload_node(node)
                next_level.append(node)

            if not next_level:
                current_level -= 1
                break

            current_nodes = next_level

        self.max_level = current_level
        print(f"[TreeRAG] Built tree up to level {self.max_level} "
              f"from {len(leaf_chunks)} leaves.")

    def _upload_node(self, node: Dict):
        text = node['text']

        dense_vec  = node.get('dense_vector')
        sparse_vec = node.get('sparse_vector')

        if dense_vec is None:
            _, dense_vec = self.tokenizer.DenseVectorRetrieval(textList=[text])
        if sparse_vec is None:
            sparse_vec = self.tokenizer.SparseTokens(text)

        text_hash = uuid.uuid5(uuid.NAMESPACE_DNS, text).hex
        node_id = str(uuid.uuid5(
            uuid.NAMESPACE_DNS,
            f"{node.get('path', 'node')}|{node['level']}|{text_hash}"
        ))
        node['node_id'] = node_id

        metadata = {k: v for k, v in node.items()
                    if k not in ('dense_vector', 'sparse_vector')}

        self.vector_db.upload_text(
            doc_id=node_id,
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            metadata=metadata,
        )

    def query(self, query_text: str, top_k: int = 5, fanout: int = 5, summary_breadth: int = 10, leaf_breadth: int = 15) -> List[Any]:
        """
        Top-down traversal. Search level L, collect child_ids from the hits,
        search level L-1 restricted to those IDs, repeat until we hit level 0.
        """
        if self.max_level < 0:
            return self.vector_db.Search(
                tokenizer=self.tokenizer, query_text=query_text, top_k=top_k
            )

        max_candidates = max(16, fanout * 4)
        current_level  = self.max_level
        candidate_ids: Optional[List[str]] = None

        while current_level >= 0:
            breadth = leaf_breadth if current_level == 0 else summary_breadth
            results = self.vector_db.Search(
            tokenizer=self.tokenizer,
            query_text=query_text,
            top_k=breadth,
            tree_level=current_level,
            node_ids=candidate_ids,
            )

            if not results:
                current_level -= 1
                candidate_ids = None
                continue

            if current_level == 0:
                return results[:top_k]

            next_ids: List[str] = []
            for r in results[:fanout]:
                for cid in (r.payload.get('child_ids') or []):
                    if cid not in next_ids:
                        next_ids.append(cid)
                        if len(next_ids) >= max_candidates:
                            break
                if len(next_ids) >= max_candidates:
                    break

            if not next_ids:
                current_level -= 1
                candidate_ids = None
                continue

            candidate_ids = next_ids
            current_level -= 1

        # Shouldn't normally reach here.
        return self.vector_db.Search(
            tokenizer=self.tokenizer, query_text=query_text, top_k=top_k
        )