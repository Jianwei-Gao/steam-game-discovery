import os
import pandas as pd
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


current_folder = Path(__file__).resolve().parent
root_folder = current_folder.parent
data_folder = root_folder / "data" / "steam_data"

df_description_path = data_folder / "descriptions.parquet"
df_review_path = data_folder / "reviews.parquet"
df_description_embedding_path = data_folder / "descriptions_embedding.parquet"
df_review_embedding_path = data_folder / "reviews_embedding.parquet"

def main():
    #import dataset
    print("reading files")
    df_description = pd.read_parquet(df_description_path)
    df_review = pd.read_parquet(df_review_path)
    df_description_embedding = pd.read_parquet(df_description_embedding_path)
    df_review_embedding = pd.read_parquet(df_review_embedding_path)

    df_review_score = (
        df_review[["appid", "review_score"]]
        .drop_duplicates(subset=["appid"])
        .reset_index(drop=True)
    )

    #merge description dataset
    desc_merged = df_description.merge(df_description_embedding, on="appid", how="inner").merge(df_review_score, on="appid", how="inner")
    desc_vector_size = len(desc_merged.iloc[0]["embedding"])
    
    #merge review dataset
    review_long = df_review.melt(
        id_vars=["appid"],
        value_vars=[f"review_{i}" for i in range(1,11)],
        var_name = "review_slot",
        value_name = "review_text"
    )

    review_emb_long = df_review_embedding.melt(
        id_vars=["appid"],
        value_vars=[f"review_{i}_embedding" for i in range(1, 11)],
        var_name="review_slot",
        value_name="review_embedding"
    )
    review_emb_long["review_slot"] = review_emb_long["review_slot"].str.replace(
        "_embedding", "", regex=False
    )
    review_merged = review_long.merge(
        review_emb_long, on=["appid", "review_slot"], how="inner"
    )

    review_merged = review_merged.dropna(subset=["review_text", "review_embedding"])
    review_merged["review_length"] = review_merged["review_text"].astype(str).str.len()
    review_merged = review_merged[review_merged.groupby("appid")["review_slot"].transform("size") == 10]
    review_vector_size = len(review_merged.iloc[0]["review_embedding"])
    
    #create qdrant database
    print("creating vector database")
    client = QdrantClient(host="localhost", port=6333)
    client.recreate_collection(
        collection_name="game_descriptions",
        vectors_config=VectorParams(size=desc_vector_size, distance=Distance.COSINE)
    )
    client.recreate_collection(
        collection_name="game_reviews",
        vectors_config=VectorParams(size=review_vector_size, distance=Distance.COSINE)
    )
    client.create_payload_index(
        collection_name="game_reviews",
        field_name="appid",
        field_schema="integer"
    )
    
    BATCH_SIZE = 500
    #upsert description embeddings
    
    desc_points = [
        PointStruct(
            id=row["appid"],
            vector=row["embedding"],
            payload={
                "appid": row["appid"],
                "name": row["name"],
                "review_score": row["review_score"]
            }
        )   
        for _, row in desc_merged.iterrows()
    ]

    for i in range(0, len(desc_points), BATCH_SIZE):
        client.upsert("game_descriptions", points=desc_points[i : i + BATCH_SIZE])
        if (i == 0) or (i//BATCH_SIZE + 1) %10 == 0:
            print(f"Upserted description batch {i // BATCH_SIZE + 1}")

    review_points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{row['appid']}_{row['review_slot']}")),
            vector=row["review_embedding"],
            payload={
                "appid": row["appid"],
                "review_length": row["review_length"],
            },
        )
        for _, row in review_merged.iterrows()
    ]

    for i in range(0, len(review_points), BATCH_SIZE):
        client.upsert("game_reviews", points=review_points[i : i + BATCH_SIZE])
        if (i == 0) or (i//BATCH_SIZE + 1) %10 == 0:
            print(f"Upserted review batch {i // BATCH_SIZE + 1}")
    
    print("done")
            
if __name__ == "__main__":
    main()
