# DL_Ass2

## Hyperparameter search

### Running the script

On a remote server make sure you did the following steps:

1. Pull the project
2. Run `chmod +x ptb-lm.py`
3. Change the path on the first line of `ptb-lm.py` by the one that the command `which python` gives you (but keep the `#!`)
4. Create the `experiences` folder
5. Copy the folder of the experience you want to improve into that `experiences` folder

And then run the following command to launch the script:

`nohup python -u random_hyperparameter_search.py BASE_XP_NAME &`

- The `BASE_XP_NAME` argument must be replaced with the name of your experience that you wish to improve. You must write the folder name of your experience that is located in the `experiences` folder.
- The `nohup` allows the script to continue running even when the ssh session is terminated. 
- The `-u` option forces the output buffer to flush frequently, allowing the output to be written into the automatically created `nohup.out` file. 