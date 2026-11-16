# steam-game-discovery

## Prerequisites  
[Python](https://www.python.org/downloads/release/python-31410/)

[Docker](https://www.docker.com/get-started/) 

If you want to try crawl the data yourself\
[Google Colab (highly recommended)](https://colab.research.google.com/) or [Jupyter (local run)](https://jupyter.org/)  
## Setup
### 0. Download data from release and extract it into backend/data/steam_data/
OPTIONAL: If you want to try fetch the data yourself (takes around 2-3 days). You do not need to run these if you download the data from release. 
```
pip install -r requirements.txt
cd backend/data

#Edit .env.example into .env, add your own steam api key (more info https://steamcommunity.com/dev) 
python build_queue.py 
python fetch_data.py
python clean_data.py

#local run, I recommend to run on collab with A100 gpu instead
jupyter execute notebook.ipynb 
```

### 0.1 Download cache from release and extrat it into backend/cache/
Optional but reduces runtime for a first run query from about 2 minute to less than 10 seconds.

### 1. python environment
```
# On project root
python -m venv .venv
pip install -r requirements.txt
```

### 2 Start up Qdrant embedding database and fastapi server in docker 
```
#takes around 5 minute for first compose up
#creator allocated 8 cpu core and 8gb memory allocated for docker engine, may get away with less
cd backend
docker compose up
```

### 3. Try recommender
```
# Currently less than 10 second runtime for description similarity score only (alpha = 1.0) per query.
# Without cache, currently about 2 min runtime for description + review similarity score (alpha between 0 and 1.0 exclusive) for first query
# With cache or after first query, less than 10 second per query.

# Example input for id "portal 2" and "570940" for darksouls
#Can also open up http://127.0.0.1:8000/docs which lets you try apis in fastapi ui
curl "http://127.0.0.1:8000/query?id=portal%202"
curl "http://127.0.0.1:8000/query?id=570940"

```

### 4. Try recommender in frontend ui (still in developement)
```
cd frontend
docker compose up

#Open up http://127.0.0.1:5173/ in browser
```

## Project Structure
```
steam-game-discovery/
├── backend/
│   ├── data/
│   │   ├── steam_data/
│   │   │   ├── details/{appid}.json   
│   │   │   ├── reviews/{appid}.json 
│   │   │   ├── descriptions_embedding.parquet
│   │   │   ├── descriptions.parquet
│   │   │   ├── reviews_embedding.parquet
│   │   │   └── reviews.parquet
│   │   ├── .env.example
│   │   ├── build_queue.py
│   │   ├── clean_data.py
│   │   ├── embed.ipynb
│   │   └── fetch_data.py
│   ├── cache/
│   │   └── *.pkl # description and review cache generated when running recommender.py 
│   ├── qdrant_data/* # qdrant database data generated when running qdrant database in docker
│   ├── api.py
│   ├── qdrant_setup.py
│   ├── recommender.py
│   ├── .dockerignore
│   ├── dockerfile
│   └── docker-compose.yaml
├── frontend/
│   ├── *
│   ├── .dockerignore
│   ├── dockerfile
│   └── docker-compose.yaml
├── .gitignore
├── requirements.txt
└── README.md
```

## Scoring algorithms
The project converts texts into embedding using [bge-m3](https://huggingface.co/BAAI/bge-m3) model. When inputting a specific game vai their name or game id, it evaluates a similiarity score with every other game in the database and return the top n scorers. 
Given:
* game $A$
* game $B$
* description embedding of game $A$: $D_A$ 
* description embedding of game $B$: $D_B$ 
* i-th review embedding of game $A$: $a_i$
* i-th review embedding of game $b$: $b_i$

Description similiarity = $D = \cos(D_A, D_B)$ 

Review similiarity with centroid = $R_C = \cos(\frac{\sum a_i}{n_a}, \frac{\sum b_j}{n_b})$ 

Review similiarity with pairwise enumerate =  $R_P = \frac{\sum_i\sum_j \cos(a_i, b_j)}{n_an_b}$ 

Review similiarity with best match pair =  $R_B = \frac{\sum_i\max_j \cos(a_i, b_j) + \sum_j\max_i \cos(a_i, b_j)}{n_a + n_b}$

$Similiarity = \alpha D + (1-\alpha)R_{<C/P/B>}$