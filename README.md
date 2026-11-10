# steam-game-discovery
## Prerequisites  
[Python](https://www.python.org/downloads/release/python-31410/)
[Docker](https://www.docker.com/get-started/) 
If you want to try crawl the data yourself
[Google Colab (highly recommended)](https://colab.research.google.com/) or [Jupyter (local run)](https://jupyter.org/)  
## Setup
### 0. Download data from release and extract it into data/steam_data
OPTIONAL: If you want to try fetch the data yourself (takes around 2-3 days). You do not need to run these if you download the data from release. 
```
pip install -r requirements.txt
cd data

#Edit .env.example into .env, add your own steam api key (more info https://steamcommunity.com/dev) 
python build_queue.py 
python fetch_data.py
python clean_data.py

#local run, I recommend to run on collab with A100 gpu instead
jupyter execute notebook.ipynb 
```

### 1. python environment
```
# On project root
python -m venv .venv
pip install -r requirements.txt
```

### 2.1 Start up Qdrant embedding database in docker 
```
cd backend/qdrant
# Starting the database in docker container
docker compose -f backend/qdrant/compose.yaml up -d

# Stop the database in docker container
docker compose -f backend/qdrant/compose.yaml down

```

### 2.2 Insert embedding into database
```
# Insert embedding data into database (need docker container  up)
cd backend
python qdrant_setup.py
```

### 3. Try recommender
```
# Currently less than 5 second runtime for description similarity score only (alpha = 1.0) per query.
# Currently about 2 min runtime for description + review similarity score only (alpha between 0 and 1.0 exclusive) per query. Future plans to optimize to reduce runtime.

# Defaults to portal 2 and darksouls as query. Can edit main file for additional query. 
cd backend
python recommender.py 
```

## Project Structure
steam-game-discovery/
├── backend/
│   ├── qdrant/
│   │   └── compose.yaml   
│   ├── qdrant_setup.py
│   └── recommender.py
├── data/
│   ├── steam_data/
│   │   ├── details/{appid}.json   
│   │   ├── reviews/{appid}.json 
│   │   ├── descriptions_embedding.parquet
│   │   ├── descriptions.parquet
│   │   ├── reviews_embedding.parquet
│   │   └── reviews.parquet
│   ├── .env.example
│   ├── build_queue.py
│   ├── clean_data.py
│   ├── embed.ipynb
│   └── fetch_data.py
├── .gitignore
├── requirements.txt
└── README.md


