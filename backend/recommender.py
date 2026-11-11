"""
recommender.py — Similar-game lookup over the Qdrant `game_descriptions` collection.

Usage:
    from recommender import GameRecommender

    rec = GameRecommender()
    rec.find_similar("portal 2", n=5)     # by name (case-insensitive)
    rec.find_similar(620, n=5)            # by appid
    rec.find_similar("620", n=5)          # appid as numeric string
"""

import pickle
import numpy as np 
from pathlib import Path
from functools import lru_cache
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    ScoredPoint
)

DESCRIPTION_COLLECTION = "game_descriptions"
REVIEW_COLLECTION = "game_reviews"
ZERO_GUARD = 1e-12
SCROLL_SIZE = 50000

FILE_DIR = Path(__file__).resolve().parent 
DESC_CACHE_PATH = FILE_DIR / "desc_cache.pkl"
REVIEW_CACHE_PATH = FILE_DIR / "review_cache.pkl"

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < ZERO_GUARD or nb < ZERO_GUARD:
        return 0.0
    return float(np.dot(a,b)/(na * nb))

def _pairwise_cosine(va: np.ndarray, vb: np.ndarray) -> np.ndarray:
    """(na, nb) matrix of cos(va[i], vb[j])."""
    va_norms = np.linalg.norm(va, axis=1, keepdims=True)
    vb_norms = np.linalg.norm(vb, axis=1, keepdims=True)
    return (va @ vb.T) / (va_norms * vb_norms.T + ZERO_GUARD)


class GameRecommender:
    def __init__(self, client: QdrantClient | None = None):
        self.client = client or QdrantClient(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True, timeout=30.0)
        self._name_to_appid: dict[str, int] | None = None 
        self._desc_cache = None 
        self._review_cache = None 
        
    def _build_name_to_appid(self) -> None:
        if self._name_to_appid is not None:
            return

        self._name_to_appid = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=DESCRIPTION_COLLECTION,
                limit=SCROLL_SIZE,
                offset=offset,
                with_payload=["appid", "name"],
                with_vectors=False,
            )
            for r in records:
                name = r.payload.get("name")
                if not name:
                    continue
                key = name.strip().lower()
                # On name collisions, keep the first one seen
                self._name_to_appid.setdefault(key, r.payload["appid"])
            if offset is None:
                break        

    def _build_desc_cache(self):
        if self._desc_cache is not None:
            return self._desc_cache
        
        if DESC_CACHE_PATH.exists():
            with open(DESC_CACHE_PATH, "rb") as f:
                self._desc_cache = pickle.load(f)
            return self._desc_cache
        
        appids, names, review_score, vectors = [], [], [], []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=DESCRIPTION_COLLECTION, 
                limit=SCROLL_SIZE,
                offset=offset, 
                with_payload=["appid", "name", "review_score"],
                with_vectors=True,
            )
            for p in records:
                appids.append(int(p.payload["appid"]))
                names.append(p.payload["name"])
                review_score.append(int(p.payload["review_score"]))
                vectors.append(p.vector)
            if offset is None:
                break
        M = np.asarray(vectors, dtype=np.float32)
        self._desc_cache = (
            np.array(appids, dtype=np.int64), 
            names,
            np.array(review_score, dtype=np.int64), 
            M
        )
        
        with open(DESC_CACHE_PATH, "wb") as f:
            pickle.dump(self._desc_cache, f)
                    
        return self._desc_cache
    
    def _build_review_cache(self):
        if self._review_cache is not None:
            return self._review_cache
                
        if REVIEW_CACHE_PATH.exists():
            with open(REVIEW_CACHE_PATH, "rb") as f:
                self._review_cache = pickle.load(f)
            return self._review_cache
        
        buckets: dict[int, list] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=REVIEW_COLLECTION,
                limit=SCROLL_SIZE,
                offset=offset,
                with_payload=['appid'],
                with_vectors=True,
            )
            for p in records:
                buckets.setdefault(int(p.payload["appid"]), []).append(p.vector)
            if offset is None:
                break
        
        reviews_per_game = 10 
        appids = np.array(sorted(buckets), dtype=np.int64)
        vector_size = len(buckets[int(appids[0])][0])
        review_tensor = np.stack(
            [np.asarray(buckets[i], dtype=np.float32) for i in appids.tolist()]
        )
        
        assert review_tensor.shape == (len(appids), reviews_per_game, vector_size)
        
        mean_unit = review_tensor.mean(axis=1)
        centroid_norm = np.linalg.norm(mean_unit, axis=1)
        
        self._review_cache = (review_tensor, appids, mean_unit, centroid_norm)
        with open(REVIEW_CACHE_PATH, "wb") as f:
            pickle.dump(self._review_cache, f)

        return self._review_cache

    def get_appid(self, id) -> int:
        if isinstance(id, int):
            return id
        if isinstance(id, str):
            s = id.strip()
            if s.isdigit():
                return int(s)
            self._build_name_to_appid()
            appid = self._name_to_appid.get(s.lower())
            if appid is None:
                raise ValueError(f"No game found with name: {id!r}")
            return appid
        raise TypeError(f"identifier must be int or str, got {type(id)!r}")

    def get_point(self, appid: int) -> ScoredPoint:
        result = self.client.retrieve(
            collection_name=DESCRIPTION_COLLECTION,
            ids=[appid],
            with_payload=True,
            with_vectors=True,
        )
        if not result:
            raise ValueError(f"No point found with appid {appid}")
        return result[0]

    def description_similarity(self, query_point: ScoredPoint, n: int) -> list[ScoredPoint]:
        """cosine similarity of description embeddings"""
        response = self.client.query_points(
            collection_name=DESCRIPTION_COLLECTION,
            query=query_point.vector,
            limit=n + 1,                
            with_payload=True,
        )
        hits = [h for h in response.points if h.id != query_point.id]
        return hits[:n]

    def review_similarity(self, query_appid, method = 2):
        tensor, appid, mean_unit, centroid_norm = self._build_review_cache()
    
        query_index = np.searchsorted(appid, query_appid)
        if query_index >= len(appid) or query_index < 0 or appid[query_index] != query_appid:
            raise ValueError(f"No reviews cached for appid {query_appid}")
        
        scores = None 
        #centroid
        if method == 0:
            q_centroid = mean_unit[query_index]
            q_centroid_norm = centroid_norm[query_index]
            scores = (mean_unit @ q_centroid) / (centroid_norm * q_centroid_norm)
        
        #pairwise
        elif method == 1:
            q = tensor[query_index]
            cos = tensor @ q.T 
            scores = cos.mean(axis=(1,2))
            
        #best match           
        elif method == 2:
            q = tensor[query_index]
            cos = tensor @ q.T 
            row_max_sum = cos.max(axis=2).sum(axis=1)
            col_max_sum = cos.max(axis=1).sum(axis=1)
            scores = (row_max_sum + col_max_sum)/20.0
             
        else:
            raise ValueError(method)

        scores[query_index] = 0.0
        return appid, scores
        
    # @staticmethod
    # def review_centroid_similarity(va: np.ndarray, vb: np.ndarray) -> float:
    #     """
    #     Using Cos(mean_i a_i, mean_j b_j)
    #     Return cosine similarity of the two games' review centroids.
    #     """
    #     if len(va) == 0 or len(vb) == 0:
    #         return 0.0
    #     return _cosine(va.mean(axis=0), vb.mean(axis=0))

    # @staticmethod
    # def review_pairwise_similarity(va: np.ndarray, vb: np.ndarray) -> float | None:
    #     """
    #     Using (1 / (n_a * n_b)) * sum_i sum_j cos(a_i, b_j)
    #     Mean of the full pairwise cosine matrix.
    #     """
    #     if len(va) == 0 or len(vb) == 0:
    #         return 0.0
    #     cos = _pairwise_cosine(va, vb)
    #     return float(cos.mean())

    # @staticmethod
    # def review_mutual_best_match_similarity(va: np.ndarray, vb: np.ndarray) -> float:
    #     """
    #     Using
    #         (1 / (n_a + n_b)) * 
    #         (sum_i max_j cos(a_i, b_j) + sum_j max_i cos(a_i, b_j))
    #     Match review to a best match against other game's reivews;
    #     Return mean over all reviews of both games.
    #     """
    #     if len(va) == 0 or len(vb) == 0:
    #         return 0.0
    #     cos = _pairwise_cosine(va, vb)         
    #     row_max_sum = cos.max(axis=1).sum()            # sum_i max_j
    #     col_max_sum = cos.max(axis=0).sum()            # sum_j max_i
    #     return float((row_max_sum + col_max_sum) / (len(va) + len(vb)))
    
    def _all_description_scores(self, query_vec: np.ndarray) -> tuple[list, list, list, list]:
        # appids, names, review_scores, vectors = [], [], [], []
        # offset = None
        # while True:
        #     records, offset = self.client.scroll(
        #         collection_name=DESCRIPTION_COLLECTION,
        #         limit=SCROLL_SIZE, 
        #         offset=offset,
        #         with_payload=["appid", "name", "review_score"],
        #         with_vectors=True,
        #     )
        #     for p in records:
        #         appids.append(int(p.payload["appid"]))
        #         names.append(p.payload["name"])
        #         review_scores.append(int(p.payload["review_score"]))
        #         vectors.append(p.vector)
        #     if offset is None:
        #         break
        
        
        appids, names, review_scores, vectors = self._build_desc_cache()
        
        M = np.asarray(vectors, dtype=np.float32)
        desc_scores = (M @ query_vec)
        return appids, names, review_scores, desc_scores
    
    def _all_review_vectors(self) -> dict[int, np.ndarray]:
        buckets: dict[int, list] = {}
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=REVIEW_COLLECTION,
                limit=SCROLL_SIZE, 
                offset=offset,
                with_payload=["appid"], 
                with_vectors=True,
            )
            for p in records:
                buckets.setdefault(p.payload.get("appid"), []).append(p.vector)
            if offset is None:
                break
        return {aid: np.asarray(vecs, dtype=np.float32) for aid, vecs in buckets.items()}
    
    def find_similar(self, id, n: int = 10, alpha: float = 1.0) -> list[dict]:
        """
        Return the top n games most similar to id.

        Args:
            id: game appid (int / numeric str) or game name (any case).
            n: number of results to return.

        Returns:
            List of dicts: {"appid", "name", "review_score", "score"}
        """

        appid = self.get_appid(id)
        query_point = self.get_point(appid)
        if alpha >= 1.0:
            hits = self.description_similarity(query_point, n)
            return [
                {
                    "appid": h.payload.get("appid"),
                    "name": h.payload.get("name"),
                    "review_score": h.payload.get("review_score"),
                    "score": float(h.score) if h.score is not None else None,
                }
                for h in hits
            ]
            
        query_vec = np.asarray(query_point.vector, dtype=np.float32)
        desc_appid, name, review_scores, desc_sim = self._all_description_scores(query_vec)
        review_appid, review_sim = self.review_similarity(appid, 2)
        
        desc_appid = np.asarray(desc_appid, dtype=np.int64)
        review_scores = np.asarray(review_scores, dtype=np.int64)
        desc_sim = np.asarray(desc_sim, dtype=np.float32)
        
        review_lookup = dict(zip(review_appid.tolist(), review_sim.tolist()))
        review_sim = np.array(
            [review_lookup.get(i, 0.0) for i in desc_appid.tolist()], dtype=np.float32
        )
        
        combined = alpha * desc_sim + (1-alpha) * review_sim
        
        order = np.argsort(-combined)
        results = []
        for i in order:
            aid = int(desc_appid[i])
            if aid == appid:
                continue
            results.append({
                "appid": aid,
                "name": name[i],
                "review_score": int(review_scores[i]),
                "score": float(combined[i]),
            })
            if len(results) >= n:
                break

        return results
        

if __name__ == "__main__":
    rec = GameRecommender()
    
    print("=== Query by name: 'portal 2' ===")
    for r in rec.find_similar("portal 2", n=5, alpha=0.5):
        print(r)
    print()
    
    print("=== Query by appid for dark souls: 570940 ===")
    for r in rec.find_similar(570940, n=5, alpha=0.5):
        print(r)
    print()
