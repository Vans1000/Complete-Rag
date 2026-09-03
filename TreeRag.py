import uuid
import numpy as np
from typing import List, Dict
from sklearn.mixture import GaussianMixture
from Tokenizer import Tokenizer
from VectorDatabase import VectorDatabase
from LLM import BaseLLM

class TreeRAG:
    def __init__(self, tokenizer: Tokenizer, vector_db: VectorDatabase, llm: BaseLLM):
        self.tokenizer = tokenizer
        self.vector_db = vector_db
        self.llm = llm

    def cluster_texts(self, embeddings: np.ndarray, n_clusters: int = None) -> List[int]:
        """Cluster node embeddings using Gaussian Mixture Models (GMM)"""
        if len(embeddings) <= 1:
            return [0] * len(embeddings)
        
        if n_clusters is None:
            n_clusters = max(1, int(len(embeddings) ** 0.5))
            
        n_clusters = min(n_clusters, len(embeddings))
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        return gmm.fit_predict(embeddings)

    def summarize_cluster(self, texts: List[str]) -> str:
        """Generate a summary for a cluster of text chunks using the LLM"""
        combined_text = "\n\n---\n\n".join(texts)
        prompt = (
            "Summarize the key information, main themes, and core details from the following chunks into a concise, detailed summary:\n\n"
            f"{combined_text}\n\nSummary:"
        )
        return self.llm.chat([{"role": "user", "content": prompt}], stream=False)

    def build_tree_and_ingest(self, leaf_chunks: List[Dict], max_depth: int = 3):
        """
        Recursively builds summary nodes and uploads all levels (leaf + summaries) to Qdrant.
        `leaf_chunks` contains metadata dicts with 'text', 'source', 'page', etc.
        """
        current_nodes = leaf_chunks
        
        # 1. Ingest base leaf nodes (Level 0)
        for idx, node in enumerate(current_nodes):
            node['level'] = 0
            self._upload_node(node)

        current_level = 0
        while current_level < max_depth and len(current_nodes) > 1:
            current_level += 1
            texts = [node['text'] for node in current_nodes]
            
            # Embed current level texts
            _, dense_vectors = self.tokenizer.BatchDenseVectorRetrieval(text_lists=texts)
            embeddings = dense_vectors.numpy()

            # Cluster chunks
            cluster_labels = self.cluster_texts(embeddings)
            num_clusters = len(set(cluster_labels))

            next_level_nodes = []
            for cluster_id in range(num_clusters):
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_texts = [texts[i] for i in cluster_indices]
                
                if not cluster_texts:
                    continue

                # Generate summary for the cluster
                summary_text = self.summarize_cluster(cluster_texts)
                parent_id = str(uuid.uuid4())

                # Retain path/source context from child nodes
                sample_source = current_nodes[cluster_indices[0]].get('path', 'tree_summary')

                summary_node = {
                    'text': summary_text,
                    'type': 'summary',
                    'level': current_level,
                    'path': sample_source,
                    'child_ids': [current_nodes[i].get('node_id') for i in cluster_indices if 'node_id' in current_nodes[i]]
                }

                # Upload summary node to Vector DB
                self._upload_node(summary_node)
                next_level_nodes.append(summary_node)

            current_nodes = next_level_nodes

    def _upload_node(self, node: Dict):
        text = node['text']
        _, dense_vec = self.tokenizer.DenseVectorRetrieval(textList=[text])
        sparse_vec = self.tokenizer.SparseTokens(text)

        node_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{node.get('path', 'node')}_{node['level']}_{text[:50]}"))
        node['node_id'] = node_id

        # Remove vector fields to avoid serialization errors
        metadata = {k: v for k, v in node.items() if k not in ['dense_vector', 'sparse_vector']}

        self.vector_db.upload_text(
            doc_id=node_id,
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            metadata=metadata
        )