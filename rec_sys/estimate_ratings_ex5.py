import shelve

from rec_sys.cf_algorithms import centered_cosine_sim


def estimate_rating(user_id, movie_id, rated_by_shelf='rated_by_shelf.db', user_col_shelf='user_col_shelf.db'):
    with shelve.open(rated_by_shelf) as rated_by, shelve.open(user_col_shelf) as user_col:
        if str(movie_id) not in rated_by:
            return None

        users_who_rated = rated_by[str(movie_id)]

        if str(user_id) not in user_col:
            return None

        target_user_vector = user_col[str(user_id)]

        numerator = 0
        denominator = 0
        for other_user_id in users_who_rated:
            other_user_vector = user_col[other_user_id]

            sim = centered_cosine_sim(target_user_vector, other_user_vector)
            if sim > 0:
                other_user_rating = other_user_vector[0, int(movie_id)]
                numerator += sim * other_user_rating
                denominator += sim

        return numerator / denominator if denominator != 0 else None


user_movie_pairs = [
    (828, 11), (2400, 4725), (3765, 1270), (4299, 4020), (5526, 2432),
    (6063, 4525), (7045, 4100), (8160, 6300), (9682, 1212), (10277, 7355)
]

for i, (user, movie) in enumerate(user_movie_pairs, start=1):
    rating = estimate_rating(user, movie)
    print(f"Pair {i}: User {user}, Movie {movie} -> Estimated Rating: {rating}")

import tracemalloc


def track_memory(user_id, movie_id):
    tracemalloc.start()
    rating = estimate_rating(user_id, movie_id)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return rating, peak / (1024 ** 2)


# Run and report memory usage for the first 6 pairs
for i, (user, movie) in enumerate(user_movie_pairs[:6], start=1):
    rating, memory_usage = track_memory(user, movie)
    print(
        f"Pair {i}: User {user}, Movie {movie} -> Estimated Rating: {rating}, Peak Memory Usage: {memory_usage:.2f} MB")
