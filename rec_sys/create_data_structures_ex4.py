import shelve

import numpy as np
from scipy.sparse import coo_matrix

from data_util import load_movielens_tf


def process_movielens_data(ConfigLf):
    ratings_tf, user_ids_voc, movie_ids_voc = load_movielens_tf(ConfigLf)
    movie_id_list = []
    for mid in movie_ids_voc.get_vocabulary():
        try:
            movie_id_list.append(int(mid))
        except ValueError:
            continue

    max_movie_id = max(movie_id_list) if movie_id_list else 0

    with shelve.open('rated_by_shelf.db', writeback=True) as rated_by, \
            shelve.open('user_col_shelf.db', writeback=True) as user_col:
        for entry in ratings_tf.as_numpy_iterator():
            user_id = str(entry['user_id'])
            movie_id = str(entry['movie_id'])
            rating = entry['user_rating']

            if movie_id not in rated_by:
                rated_by[movie_id] = []
            rated_by[movie_id].append(user_id)

            if user_id not in user_col:
                user_col[user_id] = {'indices': [], 'data': []}
            user_col[user_id]['indices'].append(int(movie_id))
            user_col[user_id]['data'].append(rating)

        for user_id, ratings in user_col.items():
            indices = np.array(ratings['indices'])
            data = np.array(ratings['data'])
            user_col[user_id] = coo_matrix((data, (np.zeros(len(data)), indices)),
                                           shape=(1, max_movie_id + 1)).tocsr()


if __name__ == "__main__":
    from config import ConfigLf

    process_movielens_data(ConfigLf)
