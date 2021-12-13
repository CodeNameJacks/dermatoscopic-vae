### Training VAE

1. Make sure ham dataset is downloaded and extracted
2. Make sure all packages are installed (can check train_vae.py to see which or run the code and install the ones that show in the error message)
3. Assuming ham dataset is stored as `ham` on working directory and we want to train it on the `mel` class, we run `python train_vae.py 
   ham/mel --no_cache` (having issues with cache)
```bash
usage: train_vae.py [-h] [-bs BATCH_SIZE] [-hp HISTORY_PATH] [-e EPOCHS] [-xc EXPERIMENT_CHECKPOINT] [-nw NUM_WORKERS] [-c] [-nc] [-cf [CACHE_FILE]] ham_dataset_dir

positional arguments:
  ham_dataset_dir       Path to ham dataset's category ex: `ham/mel`.

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for the model (default - 32).
  -hp HISTORY_PATH, --history_path HISTORY_PATH
                        Path to where to store history in json file. Ex: checkpoints/vae/history.json (default)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train for (default - 150)
  -xc EXPERIMENT_CHECKPOINT, --experiment_checkpoint EXPERIMENT_CHECKPOINT
                        Checkpoint to save after done training ex: (default) checkpoints/vae/vae -> checkpoints/vae/vae.index, checkpoints/vae/vae.data-00000-of-00001,
                        checkpoints/vae/checkpoint.
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        Maximum number of processes to spin up when using process-based threading (default - number of cores [multiprocessing.cpu_count()]).
  -c, --cache           Caching of training data is enabled.
  -nc, --no_cache       Caching of training data is disabled (default).
  -cf [CACHE_FILE], --cache_file [CACHE_FILE]
                        File location for where to cache. Ex: /tmp/cache. If caching is enabled but directory is not provided, will cache in memory (default).

```

### Testing VAE

1. Make sure to train VAE and locate checkpoint file before testing.
2. Run `python test_vae.py checkpoints/vae/vae -hdd="ham/mel"` if trained on mel class and checkpoints were saved to `checkpoints/vae/vae`.

```bash
usage: test_vae.py [-h] -hdd HAM_DATASET_DIR [-bs BATCH_SIZE] model_weights

positional arguments:
  model_weights         Checkpoint to load the weights from. ex: checkpoints/vae/vae.data-00000-of-00001 then input checkpoints/vae/vae

optional arguments:
  -h, --help            show this help message and exit
  -hdd HAM_DATASET_DIR, --ham_dataset_dir HAM_DATASET_DIR
                        Dataset to test VAE on (only 1 class) ex: ham/mel
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of images to reconstruct
```

### Plotting Metrics

1. Make sure to train VAE and locate history.json file before plotting metrics.
2. Run `python plot_metrics.py checkpoints/vae/history.json -dfk=2`. -dfk removes first k epochs from the plot as the losses are usually very high 
   and skew the plot.

```bash
usage: plot_metrics.py [-h] [-dfk DROP_FIRST_K_EPOCHS] history_path

positional arguments:
  history_path          Checkpoint to load the weights from. ex: checkpoints/vae/vae.data-00000-of-00001 then input checkpoints/vae/vae

optional arguments:
  -h, --help            show this help message and exit
  -dfk DROP_FIRST_K_EPOCHS, --drop_first_k_epochs DROP_FIRST_K_EPOCHS
                        Number of epochs of metrics to drop from the start (since first few losses are extremely high, it can skew the graph) Ex: 2 (default).
```

### Generating Synthetic Data

1. Make sure to train VAE and locate checkpoint file before generating synthetic data.
2. Run `python generate_synthetic_data.py checkpoints/vae/vae -hdd="ham/mel" -nm=100` to generate 100 synthetic data points with a default epsilon 
   of 0.2

```bash
usage: generate_synthetic_data.py [-h] -hdd HAM_DATASET_DIR [-sd SAVE_DIR] [-e EPSILON] [-nm NUM_IMAGES] model_weights

positional arguments:
  model_weights         Checkpoint to load the weights from. ex: checkpoints/vae/vae.data-00000-of-00001 then input checkpoints/vae/vae

optional arguments:
  -h, --help            show this help message and exit
  -hdd HAM_DATASET_DIR, --ham_dataset_dir HAM_DATASET_DIR
                        Dataset to test VAE on (only 1 class) ex: ham/mel
  -sd SAVE_DIR, --save_dir SAVE_DIR
                        Directory to store the generated data
  -e EPSILON, --epsilon EPSILON
                        Epsilon value for generating noise on latent space from standard normal ex: 0.2 (default)
  -nm NUM_IMAGES, --num_images NUM_IMAGES
                        Number of images to reconstruct
```