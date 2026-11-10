"""
recommender.py — Similar-game lookup over the Qdrant `game_descriptions` collection.

Usage:
    from recommender import GameRecommender

    rec = GameRecommender()
    rec.find_similar("portal 2", n=5)     # by name (case-insensitive)
    rec.find_similar(620, n=5)            # by appid
    rec.find_similar("620", n=5)          # appid as numeric string
"""

import numpy as np 
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
SCROLL_SIZE = 5000

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
        self.client = client or QdrantClient(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True)
        self._name_to_appid: dict[str, int] | None = None
        self.review_index = None 
        
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
        
    def _fetch_review(self, appid: int) -> np.ndarray:
        points: list[ScoredPoint] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=REVIEW_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="appid", match=MatchValue(value=appid))]
                ),
                limit=SCROLL_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=True,
            )
            points.extend(records)
            if offset is None:
                break
        if not points:
            return np.empty((0, 0), dtype=np.float32)
        return np.array([p.vector for p in points], dtype=np.float32)

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

    @staticmethod
    def review_centroid_similarity(va: np.ndarray, vb: np.ndarray) -> float:
        """
        Using Cos(mean_i a_i, mean_j b_j)
        Return cosine similarity of the two games' review centroids.
        """
        if len(va) == 0 or len(vb) == 0:
            return 0.0
        return _cosine(va.mean(axis=0), vb.mean(axis=0))

    @staticmethod
    def review_pairwise_similarity(va: np.ndarray, vb: np.ndarray) -> float | None:
        """
        Using (1 / (n_a * n_b)) * sum_i sum_j cos(a_i, b_j)
        Mean of the full pairwise cosine matrix.
        """
        if len(va) == 0 or len(vb) == 0:
            return 0.0
        cos = _pairwise_cosine(va, vb)
        return float(cos.mean())

    @staticmethod
    def review_mutual_best_match_similarity(va: np.ndarray, vb: np.ndarray) -> float:
        """
        Using
            (1 / (n_a + n_b)) * 
            (sum_i max_j cos(a_i, b_j) + sum_j max_i cos(a_i, b_j))
        Match review to a best match against other game's reivews;
        Return mean over all reviews of both games.
        """
        if len(va) == 0 or len(vb) == 0:
            return 0.0
        cos = _pairwise_cosine(va, vb)         
        row_max_sum = cos.max(axis=1).sum()            # sum_i max_j
        col_max_sum = cos.max(axis=0).sum()            # sum_j max_i
        return float((row_max_sum + col_max_sum) / (len(va) + len(vb)))
    
    def _all_description_scores(self, query_vec: np.ndarray) -> tuple[list[int], list[str], list[int], list[float]]:
        appids, names, review_scores, vectors = [], [], [], []
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
                review_scores.append(int(p.payload["review_score"]))
                vectors.append(p.vector)
            if offset is None:
                break
        v = np.asarray(vectors, dtype=np.float32)
        norm = np.linalg.norm(v, axis=1) + ZERO_GUARD
        q = query_vec / (np.linalg.norm(query_vec) + ZERO_GUARD)
        desc_scores = (v @ q) / norm
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
        desc_rows = self._all_description_scores(query_vec)
        all_reviews = self._all_review_vectors()   
        qa = all_reviews.get(appid)       

        results = []
        for row in desc_rows:
            if row["appid"] == appid:
                continue
            vb = all_reviews.get(row["appid"])
            if qa is None or vb is None:
                combined = row["desc_score"]
            else:
                review_score = self.review_mutual_best_match_similarity(qa, vb)
                combined = (row["desc_score"] if review_score is None
                            else alpha * row["desc_score"] + (1 - alpha) * review_score)
            results.append({
                "appid": row["appid"],
                "name": row["name"],
                "review_score": row["review_score"],
                "score": combined,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:n]



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