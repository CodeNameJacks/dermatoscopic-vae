### Training VAE

1. Make sure speech commands is downloaded and extracted
2. Make sure all packages are installed (can check train_vae.py to see which or run the code and install the ones that show in the error message)
3. Assuming speech commands is stored as `speech_commands` on working directory and we want to train it on `cat`, we run `python train_vae.py speech_commands/cat --no_cache` (having issues with cache)
```bash
usage: train_vae.py [-h] [-bs BATCH_SIZE] [-nw NUM_WORKERS] [-c] [-nc] [-cf [CACHE_FILE]] speech_commands_dir

positional arguments:
  speech_commands_dir   Path to speech commands dataset's category ex: `speech_commands/cat`.

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for the model (default - 32).
  -nw NUM_WORKERS, --num_workers NUM_WORKERS
                        Maximum number of processes to spin up when using process-based threading (default - number of cores [multiprocessing.cpu_count()]).
  -c, --cache           Caching of training data is enabled.
  -nc, --no_cache       Caching of training data is disabled (default).
  -cf [CACHE_FILE], --cache_file [CACHE_FILE]
                        File location for where to cache. Ex: /tmp/cache. If caching is enabled but directory is not provided, will cache in memory (default).
```

### Known Issues

- Losses going to nan/inf and huge difference from train loss / val loss. I'm assuming this is due to the architecture, need to debug.