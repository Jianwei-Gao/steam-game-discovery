from fastapi import FastAPI 
from pydantic import BaseModel
from recommender import GameRecommender

engine = GameRecommender()

app = FastAPI()

class Recommendation(BaseModel):
    appid: int 
    name: str
    review_score: int 
    score: float 

@app.get("/")
def root():
    return {"msg": "hello world"}

@app.get("/query", response_model=list[Recommendation])
def query_game(id: int | str, n: int = 10, alpha:float = 0.5, method:int = 0):
    """
    Return the top n games most similar to id.

    **Args:**
    * **id**: Game appid (int / numeric str) or game name (any case).
    * **n**: Number of results to return.
    * **alpha**: Weight of description vs review. Alpha between 0 and 1.0 for similarity score using game description and review; 1.0 for description only.
    * **method**: Indicates which review similarity metrics to use:
      * `0`: centroid
      * `1`: pairwise sum
      * `2`: pairwise best match

    **Returns:**
    * List of dicts: `{"appid", "name", "review_score", "score"}`
    """
    return engine.find_similar(id, n, alpha, method)
