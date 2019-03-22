# DL_Ass2

## Hyperparameter search

### Running the script

On a remote server make sure you did the following steps:

1. Pull the project
2. Change the path on the first line of `ptb-lm.py` by the one that the command `which python` gives you (but keep the `#!`)
3. Create the `experiences` folder
4. Copy the folder of the experience you want to improve into that `experiences` folder
    - If you want to use Google Cloud SDK and `scp` for that, you might need to relax the permissions of your remote experience folder with `chmod -R 777 experiences`

And then run the following command to launch the script:

`nohup python -u random_hyperparameter_search.py BASE_XP_NAME &`

- The `BASE_XP_NAME` argument must be replaced with the name of your experience that you wish to improve. You must write the folder name of your experience that is located in the `experiences` folder.
- The `nohup` allows the script to continue running even when the ssh session is terminated. 
- The `-u` option forces the output buffer to flush frequently, allowing the output to be written into the automatically created `nohup.out` file.

### Downloading the results 

With Google Cloud SDK, run the following command:

```gcloud compute scp --recurse remote_server:training_directory_path local_directory_path```