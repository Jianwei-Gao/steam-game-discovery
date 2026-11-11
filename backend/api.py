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
def query_game(id: int | str, n: int = 10, alpha:float = 0.5):
    return engine.find_similar(id, n, alpha)
