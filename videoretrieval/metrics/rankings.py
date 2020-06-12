"""Copyright 2020 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Defines the loss function needed to train the model.
"""

import tensorflow as tf

def get_desired_batch_size(num_of_embeddings, embedding_shape):
    """Helper function"""

    input_bytes_per_batch = 200000 * 5000 * 700

    return 5000


def compute_ranking_metrics_wrapper(recall_at_k_list):

    def get_rank(index_matrix):
        return index_matrix[0][0]

    @tf.function
    def compute_ranking_metrics(similarites, desired_index):
        sorted_indexes = tf.argsort(similarites, direction="DESCENDING")

        index_ = tf.where(sorted_indexes == desired_index)

        rank = tf.numpy_function(get_rank, [index_matrix], tf.int32)

        recall_at_k_results = []

        for i, element in enumerate(recall_at_k_list):
            if rank <= i:
                recall_at_k_results += [1.0] * (len(recall_at_k_list) - i)
                break
            else:
                recall_at_k_results.append(0.0)

        return rank, tuple(recall_at_k_results)

    return compute_ranking_metrics

def get_ranking_metrics(
    static_embeddings, query_embeddings, recall_at_k_list):
    """Rankings for text to video lookup"""

    compute_similarities_fn = lambda embeddings_batch, desired_indicies: (
        tf.linalg.matmul(
            embeddings_batch, static_embeddings, transpose_b=True),
        desired_indicies)

    compute_ranks_fn = compute_ranking_metrics_wrapper(recall_at_k_list)

    rankings_dataset = (query_embeddings
        .batch(batch_size)
        .map(tf.function(compute_similarities_fn))
        .unbatch()
        .map(compute_ranks_fn))

    import ipdb; ipdb.set_trace();

    rankings = np.array(rankings_dataset.as_numpy_iterator())
    

@tf.function
def get_index_of_item_in_tensor(tensor_item_pair):
    tensor, item = tensor_item_pair
    return tf.where(tensor == item) 

@tf.function
def get_ranking_metrics_for_batch(
    static_embeddings, query_embeddings):

    similarities = tf.linalg.matmul(
        query_embeddings, static_embeddings, transpose_b=True)

    similarities_sorted = tf.argsort(similarities,
        direction='DESCENDING')

    ranks = 1 + tf.map_fn(
        get_index_of_item_in_tensor, 
        (similarities_sorted, tf.range(query_embeddings.shape[0])),
        dtype=tf.int64)

    mean_rank = tf.reduce_sum(ranks) / len(ranks)
    return mean_rank