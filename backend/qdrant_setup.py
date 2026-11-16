import os
import pandas as pd
import numpy as np
import uuid
from pathlib import Path

import grpc
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.connection import get_channel


current_folder = Path(__file__).resolve().parent
data_folder = current_folder / "data" / "steam_data"

df_description_path = data_folder / "descriptions.parquet"
df_review_path = data_folder / "reviews.parquet"
df_description_embedding_path = data_folder / "descriptions_embedding.parquet"
df_review_embedding_path = data_folder / "reviews_embedding.parquet"

SKIP_IF_DATABSE_HAS_ITEMS = True
BATCH_SIZE = 500
REVIEW_COLS = [f"review_{i}" for i in range(1, 11)]
EMBEDDING_COLS = [f"review_{i}_embedding" for i in range(1, 11)]

def wait_for_qdrant_grpc(host="qdrant", port=6334, timeout=10.0) -> bool:
  channel = get_channel(host=host, port=port, ssl=False)
  try:
    grpc.channel_ready_future(channel).result(timeout=timeout)
    return True
  except grpc.FutureTimeoutError:
    return False

def to_f32(vec):
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32)

def iter_description_points(df, batch_size=BATCH_SIZE):
    batch = []
    for row in df.itertuples(index=False):
        batch.append(
            PointStruct(
                id=row.appid,
                vector=row.embedding, 
                payload={
                    "appid": row.appid,
                    "name": row.name,
                    "review_score": row.review_score,
                },
            )
        )
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_review_points(df, batch_size=BATCH_SIZE):
    batch = []
    for row in df.itertuples(index=False):
        appid = row.appid
        for i, (rcol, ecol) in enumerate(zip(REVIEW_COLS, EMBEDDING_COLS), start=1):
            review_text = getattr(row, rcol)
            review_embedding = getattr(row, ecol)

            if review_text is None or review_embedding is None:
                continue
            if isinstance(review_text, float) and pd.isna(review_text):
                continue

            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{appid}_review_{i}"))
            batch.append(
                PointStruct(
                    id=point_id,
                    vector=review_embedding,
                    payload={
                        "appid": appid,
                        "review_length": len(str(review_text)),
                    },
                )
            )
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def upsert_into_database(client, collection_name, point_iterator, label):
    batch_num = 0
    for batch in point_iterator:
        client.upsert(collection_name, points=batch)
        batch_num += 1
        if batch_num == 1 or batch_num % 10 == 0:
            print(f"Upserted {label} batch {batch_num}")

def main():
    print("initializing data for qdrant")
    
    if(not wait_for_qdrant_grpc()):
        raise RuntimeError("qdrant grpc channel failed to open")
    
    client = QdrantClient(host="qdrant", port=6333, grpc_port=6334, prefer_grpc=True, timeout=30.0)

    if  SKIP_IF_DATABSE_HAS_ITEMS \
        and client.collection_exists(collection_name="game_descriptions") \
        and client.collection_exists(collection_name="game_reviews") \
        and client.count(collection_name="game_descriptions", exact=False).count > 0 \
        and client.count(collection_name="game_reviews", exact=False).count > 0:
            print("skipped data initialization for qdrant")
            print(client.count(collection_name="game_descriptions", exact=False).count)
            print(client.count(collection_name="game_reviews", exact=False).count)
            return 

    #import dataset
    print("reading files")
    df_description = pd.read_parquet(df_description_path)
    df_review = pd.read_parquet(df_review_path)
    df_description_embedding = pd.read_parquet(df_description_embedding_path)
    df_description_embedding["embedding"] = df_description_embedding["embedding"].apply(to_f32)
    
    df_review_embedding = pd.read_parquet(df_review_embedding_path)
    for col in EMBEDDING_COLS:
        df_review_embedding[col] = df_review_embedding[col].apply(to_f32)
        
    df_review = df_review[df_review["review_10"].notna()].reset_index(drop=True)
    df_review_embedding = df_review_embedding[df_review_embedding["review_10_embedding"].notna()].reset_index(drop=True)

    df_review_score = (
        df_review[["appid", "review_score"]]
        .drop_duplicates(subset=["appid"])
        .reset_index(drop=True)
    )

    #merge description dataset
    desc_merged = df_description.merge(df_description_embedding, on="appid", how="inner").merge(df_review_score, on="appid", how="inner")
    desc_vector_size = len(desc_merged.iloc[0]["embedding"])
    
    #merge review dataset
    # review_long = df_review.melt(
    #     id_vars=["appid"],
    #     value_vars=[f"review_{i}" for i in range(1,11)],
    #     var_name = "review_slot",
    #     value_name = "review_text"
    # )

    # review_emb_long = df_review_embedding.melt(
    #     id_vars=["appid"],
    #     value_vars=[f"review_{i}_embedding" for i in range(1, 11)],
    #     var_name="review_slot",
    #     value_name="review_embedding"
    # )
    # review_emb_long["review_slot"] = review_emb_long["review_slot"].str.replace(
    #     "_embedding", "", regex=False
    # )
    # review_merged = review_long.merge(
    #     review_emb_long, on=["appid", "review_slot"], how="inner"
    # )

    # review_merged = review_merged.dropna(subset=["review_text", "review_embedding"])
    # review_merged["review_length"] = review_merged["review_text"].astype(str).str.len()
    # review_merged = review_merged[review_merged.groupby("appid")["review_slot"].transform("size") == 10]
    # review_vector_size = len(review_merged.iloc[0]["review_embedding"])
    
    review_merged = df_review.merge(df_review_embedding, on="appid", how="inner")
    review_vector_size = len(review_merged.iloc[0]["review_1_embedding"])
    
    #create qdrant database
    print("creating vector database")
    
    if client.collection_exists("game_descriptions"):
        client.delete_collection("game_descriptions")
    client.create_collection(
        collection_name="game_descriptions",
        vectors_config=VectorParams(size=desc_vector_size, distance=Distance.COSINE)
    )

    if client.collection_exists("game_reviews"):
        client.delete_collection("game_reviews")
    client.create_collection(
        collection_name="game_reviews",
        vectors_config=VectorParams(size=review_vector_size, distance=Distance.COSINE)
    )
    
    #upsert description embeddings
    upsert_into_database(
        client,
        "game_descriptions",
        iter_description_points(desc_merged, BATCH_SIZE),
        "description"
    )
    upsert_into_database(
            client,
            "game_reviews",
            iter_review_points(review_merged, BATCH_SIZE),
            "review"
    )
    
    # desc_points = [
    #     PointStruct(
    #         id=row["appid"],
    #         vector=row["embedding"],
    #         payload={
    #             "appid": row["appid"],
    #             "name": row["name"],
    #             "review_score": row["review_score"]
    #         }
    #     )   
    #     for _, row in desc_merged.iterrows()
    # ]

    # for i in range(0, len(desc_points), BATCH_SIZE):
    #     client.upsert("game_descriptions", points=desc_points[i : i + BATCH_SIZE])
    #     if (i == 0) or (i//BATCH_SIZE + 1) %10 == 0:
    #         print(f"Upserted description batch {i // BATCH_SIZE + 1}")

    # review_points = [
    #     PointStruct(
    #         id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{row['appid']}_{row['review_slot']}")),
    #         vector=row["review_embedding"],
    #         payload={
    #             "appid": row["appid"],
    #             "review_length": row["review_length"],
    #         },
    #     )
    #     for _, row in review_merged.iterrows()
    # ]

    # for i in range(0, len(review_points), BATCH_SIZE):
    #     client.upsert("game_reviews", points=review_points[i : i + BATCH_SIZE])
    #     if (i == 0) or (i//BATCH_SIZE + 1) %10 == 0:
    #         print(f"Upserted review batch {i // BATCH_SIZE + 1}")
    
    print("done")
            
if __name__ == "__main__":
    main()
