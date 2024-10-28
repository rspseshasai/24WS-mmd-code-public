import dataclasses

# Configurations for collaborative filtering
@dataclasses.dataclass
class ConfigCf:
    max_rows: int = int(2.5e6)
    download_dir: str = "/scratch/core/artur/movielens/"
    unzipped_dir: str = download_dir + "ml-25m/"
    dowload_url: str = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    file_path: str = download_dir + "ml-25m/ratings.csv"


# Configurations for latent factor models
@dataclasses.dataclass
class ConfigLf:
    # DS details here: https://www.tensorflow.org/datasets/catalog/movielens
    # This dataset has only split 'train'. Good explanation of the split arg:
    # https://stackabuse.com/split-train-test-and-validation-sets-with-tensorflow-datasets-tfds/


    # dataset_base_name: str = 'movielens/25m'
    # dataset_split = 'train[:1%]'   # Use only a part of the data to debug
    dataset_base_name: str = 'movielens/100k'
    dataset_split = 'train'
    split_ratios = (0.8, 0.1, 0.1)  # train, test, validation
    split_shuffle_seed = 22

    shuffle_files: bool = True
    data_dir: str = '/scratch/core/artur/movielens/'

    # Configurations for matrix factorization
    rng_seed: int = 42
    num_factors = 10

    num_epochs: int = 10

    fixed_learning_rate = None  # If None, use dynamic learning rate
    dyn_lr_initial: float = 1.0
    dyn_lr_decay_rate: float = 0.99
    dyn_lr_steps: int = 2   # Number of epochs before decaying the learning rate

    reg_param: float = 0.1
    batch_size_training: int = 128

    batch_size_predict_with_mse: int = 256
    num_records_predict_and_compare: int = 5
    num_predictions_to_show: int = 5
