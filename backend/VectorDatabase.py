from PIL import Image
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector
from pathlib import Path
import math
from typing import List, Optional
from rag_config import FALLBACK_TEXT_DENSE_DIM, RERANK_MULTIPLIER, RERANK_MAX_CANDIDATES

CLIP_DIM = 512


class VectorDatabase:
    def __init__(self, host="localhost", port=6333, collection_name="documents", use_grpc=False, grpc_port=6334,
                 text_dense_dim: Optional[int] = None,
                 rerank_multiplier: int = RERANK_MULTIPLIER,
                 max_candidates: int = RERANK_MAX_CANDIDATES):

        self.client = QdrantClient(host=host, port=port, grpc_port=grpc_port, prefer_grpc=use_grpc)
        self.collection_name = collection_name
        self.rerank_multiplier = max(1, int(rerank_multiplier))
        self.max_candidates = max(1, int(max_candidates))
        self._warned = set()
        if not self.client.collection_exists(self.collection_name):
            VectorDatabase.create_collection(self.client, self.collection_name, text_dense_dim)
        self.text_dense_dim = self._read_text_dense_dim()

    @staticmethod
    def create_collection(client, name, text_dense_dim: Optional[int] = None):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "image_dense": VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
                "text_dense": VectorParams(size=text_dense_dim or FALLBACK_TEXT_DENSE_DIM, distance=Distance.COSINE),
                "caption_dense": VectorParams(size=CLIP_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "text_sparse": SparseVectorParams(index={}),
                "caption_sparse": SparseVectorParams(index={})
            }
        )

    def _read_text_dense_dim(self) -> Optional[int]:
        try:
            vectors = self.client.get_collection(self.collection_name).config.params.vectors
            params = vectors.get("text_dense") if isinstance(vectors, dict) else None
            return params.size if params is not None else None
        except Exception as e:
            print(f"Warning: could not read text_dense size of '{self.collection_name}': {e}")
            return None

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            print(f"Warning: {msg}")

    def _text_dense_usable(self, vec) -> bool:
        if vec is None:
            return False
        size = int(vec.numel()) if hasattr(vec, "numel") else len(vec)
        if self.text_dense_dim is not None and size == self.text_dense_dim:
            return True
        self._warn_once(
            f"dim{size}",
            f"text embedder produces {size}-dim vectors but collection '{self.collection_name}' has "
            f"text_dense size {self.text_dense_dim}. Using SPLADE-only text retrieval for this collection. "
            f"Re-ingest into a new collection to use the text embedder (older collections hold CLIP text vectors)."
        )
        return False

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def upload_text(self, doc_id, dense_vector, sparse_vector, metadata):
        indices = sparse_vector.nonzero().flatten().cpu().tolist()
        values = sparse_vector[indices].cpu().tolist()
        vector = {"text_sparse": SparseVector(indices=indices, values=values)}
        if self._text_dense_usable(dense_vector):
            vector["text_dense"] = dense_vector.cpu().tolist()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=metadata
                )
            ]
        )

    def upload_image(self, doc_id, dense_vector, caption_dense_vector, sparse_caption_vector, metadata):
        if sparse_caption_vector is not None:
            indices = sparse_caption_vector.nonzero().flatten().cpu().tolist()
            values = sparse_caption_vector[indices].cpu().tolist()
        else:
            indices = []
            values = []

        cap_dense = caption_dense_vector.cpu().tolist() if caption_dense_vector is not None else [0.0] * CLIP_DIM

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=doc_id,
                    vector={
                        "image_dense": dense_vector.cpu().tolist(),
                        "caption_dense": cap_dense,
                        "caption_sparse": SparseVector(indices=indices, values=values)
                    },
                    payload=metadata
                )
            ]
        )

    def candidate_pool(self, top_k: int) -> int:
        return max(top_k, min(top_k * self.rerank_multiplier, max(self.max_candidates, top_k)))

    def Search(self, tokenizer, query_text=None, query_image=None, top_k=20, tree_level: int = None, node_ids: Optional[List[str]] = None):
        results = []
        pool = self.candidate_pool(top_k)

        must_clauses = []
        if tree_level is not None:
            must_clauses.append(models.FieldCondition(
                key="level",
                match=models.MatchValue(value=tree_level),
            ))
        if node_ids:
            must_clauses.append(models.HasIdCondition(has_id=list(node_ids)))
        query_filter = models.Filter(must=must_clauses) if must_clauses else None

        if query_text:
            query_sparse = tokenizer.SparseTokens(query_text)
            sparse_indices = query_sparse.nonzero().flatten().cpu().tolist()
            sparse_values = query_sparse[sparse_indices].cpu().tolist()
            sparse_vec = models.SparseVector(indices=sparse_indices, values=sparse_values)

            text_dense = tokenizer.TextDenseVector(query_text, is_query=True) if hasattr(tokenizer, "TextDenseVector") else None
            if self._text_dense_usable(text_dense):
                text_query_results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(query=text_dense.tolist(), using="text_dense", limit=pool, filter=query_filter),
                        models.Prefetch(query=sparse_vec, using="text_sparse", limit=pool, filter=query_filter),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=pool,
                    with_payload=True
                ).points
            else:
                text_query_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=sparse_vec, using="text_sparse", limit=pool,
                    query_filter=query_filter, with_payload=True
                ).points

                for rank, p in enumerate(text_query_results):
                    p.score = 1.0 / (rank + 2)
            results.extend(text_query_results)

            query_dense = tokenizer.DenseVectorRetrieval(textList=[query_text])[1]
            image_query_results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=query_dense.tolist(), using="image_dense", limit=pool, filter=query_filter),
                    models.Prefetch(query=query_dense.tolist(), using="caption_dense", limit=pool, filter=query_filter),
                    models.Prefetch(query=sparse_vec, using="caption_sparse", limit=pool, filter=query_filter),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            ).points
            results.extend(image_query_results)

        if query_image:
            img_dense = tokenizer.DenseVectorRetrieval(pil_image=query_image)[0]

            caption = tokenizer.ImageCaptioning(query_image) or ""
            cap_dense = tokenizer.DenseVectorRetrieval(textList=[caption])[1]
            cap_sparse = tokenizer.SparseTokens(caption)
            cap_sparse_indices = cap_sparse.nonzero().flatten().cpu().tolist()
            cap_sparse_values = cap_sparse[cap_sparse_indices].cpu().tolist()
            cap_sparse_vec = models.SparseVector(indices=cap_sparse_indices, values=cap_sparse_values)

            image_query_results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=img_dense.cpu().tolist(), using="image_dense", limit=top_k, filter=query_filter),
                    models.Prefetch(query=cap_dense.cpu().tolist(), using="caption_dense", limit=top_k, filter=query_filter),
                    models.Prefetch(query=cap_sparse_vec, using="caption_sparse", limit=top_k, filter=query_filter),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True
            ).points
            results.extend(image_query_results)

        seen = set()
        unique_results = []
        for result in results:
            if result.id not in seen:
                unique_results.append(result)
                seen.add(result.id)
        unique_results.sort(key=lambda x: x.score, reverse=True)

        if query_text:
            image_results = [res for res in unique_results if res.payload.get('type') == 'image']
            pil_images = []
            valid_indices = []

            if getattr(tokenizer, "vision_model", None) is not None:
                for i, res in enumerate(image_results):
                    cache_path = res.payload.get('image_cache_path')

                    if cache_path and Path(cache_path).exists():
                        try:
                            pil_image = Image.open(cache_path).convert("RGB")
                            pil_images.append(pil_image)
                            valid_indices.append(i)
                        except Exception as e:
                            print(f"Warning: Could not load image from cache: {e}")
                            continue
                    else:
                        print(f"Warning: Image cache not found: {cache_path}")

            if pil_images:
                captions = [image_results[idx].payload.get('caption', '') for idx in valid_indices]
                vision_scores = tokenizer.VisionRerank(query_text, pil_images, captions=captions)
                for i, score in enumerate(vision_scores):
                    idx = valid_indices[i]
                    image_results[idx].score += VectorDatabase.sigmoid(score) * 2.0

                for img in pil_images:
                    img.close()

            candidates_text = [res.payload.get('caption', res.payload.get('text', '')) for res in unique_results]
            if candidates_text:
                cross_encoder_scores = tokenizer.CrossEncoderRanking(query_text, candidates_text)
                for i, score in enumerate(cross_encoder_scores):
                    unique_results[i].score += VectorDatabase.sigmoid(score) * 2.0

        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:top_k]

    def fetch_adjacent(self, path: str, chunk_idx: int, radius: int = 1):
        if chunk_idx is None or not path:
            return []
        lo, hi = chunk_idx - radius, chunk_idx + radius
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="path", match=models.MatchValue(value=path)),
                    models.FieldCondition(key="chunk_idx", range=models.Range(gte=lo, lte=hi)),
                    models.FieldCondition(key="type", match=models.MatchValue(value="text")),
                ]
            ),
            limit=(2 * radius + 1) * 2,
            with_payload=True,
            with_vectors=False,
        )[0]
        return [p for p in results if p.payload.get("chunk_idx") != chunk_idx]
