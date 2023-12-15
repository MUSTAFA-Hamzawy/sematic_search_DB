from hnswlib import Index
import pickle
import numpy as np
from typing import Dict, List, Annotated
from sklearn.cluster import MiniBatchKMeans
import time
from dataclasses import dataclass
import os

class VecDB:
    def _init_(self, file_path = "clusters_100K", new_db = True,n_clusters=100, batch_size=10000) -> None:
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        if file_path is not None:
          if new_db:
            self.file_path = file_path
            # Create a new folder called "new_db"
            os.makedirs(self.file_path, exist_ok=True)
          else:
            with open(file_path + "/objects.pkl", "rb") as file:
                loaded_obj = pickle.load(file)
            self._dict.update(loaded_obj.dict_)  # Update the attributes
            self.file_path = file_path
        else:
            self.file_path = "new_db"
            # Create a new folder called "new_db"
            os.makedirs(self.file_path, exist_ok=True)

    def retrive(self, query, top_k):
        # Predict the cluster assignment of the query vector using MiniBatch K-Means
        query_cluster = self.kmeans.predict(query.reshape(1, -1))[0]

        selected_cluster_file = f"{self.file_path}/cluster_{query_cluster}.pkl"
        with open(selected_cluster_file, 'rb') as cluster_file:
            selected_cluster_records = pickle.load(cluster_file)

        # Extract vectors from records
        selected_cluster_vectors = np.array([record["embed"] for record in selected_cluster_records])

        # Build HNSW index for the selected cluster
        hnsw_cluster = Index(space='cosine', dim=selected_cluster_vectors.shape[1])
        hnsw_cluster.init_index(max_elements=len(selected_cluster_vectors), ef_construction=200, M=16)
        hnsw_cluster.add_items(selected_cluster_vectors, np.arange(len(selected_cluster_vectors)))
        hnsw_cluster.save_index(f"hnsw_index_cluster_{query_cluster}.bin")

        # Search for similar vectors in the selected cluster
        labels, distances = hnsw_cluster.knn_query(query.reshape(1, -1), k=top_k)
        similar_vectors = selected_cluster_vectors[labels[0]]

        # Retrieve IDs of similar vectors from the original dataset
        similar_vector_ids = np.array([record["id"] for record in selected_cluster_records])[labels[0]]

        # 'similar_vectors' contains the top K similar vectors to the query vector within the selected cluster
        np.savetxt("similar_vectors.txt", similar_vectors, fmt='%.6f', delimiter='\t')

        return similar_vector_ids

    def insert_records(self, rows): 
      # : List[Dict[int, Annotated[List[float], 70]]]
      # records = [row["embed"] for row in rows]
      records = np.array([row["embed"] for row in rows])
      # records = rows
      self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,batch_size=self.batch_size, n_init="auto",random_state=0)
      cluster_assignments = self.kmeans.fit_predict(records)

      with open(self.file_path + "/objects.pkl", 'wb') as objects_file:
              pickle.dump(self, objects_file)

      # Save all clusters to separate files
      for cluster_id in range(self.n_clusters):
          cluster_indices = np.where(cluster_assignments == cluster_id)[0]
          cluster_records = [{"id": i , "embed": list(records[i])} for i in cluster_indices]

          # Specify the file path for each cluster
          cluster_file_path = f"{self.file_path}/cluster_{cluster_id}.pkl"

          # Open the file in binary write mode and dump the list of cluster records
          with open(cluster_file_path, 'wb') as cluster_file:
              pickle.dump(cluster_records, cluster_file)