Note: Please run this on linux due to compatibility issues with some of the dependencies

# AI701 G26: An Empirical Study of Dependency Parsing on Dialectal Arabic

### Setup instructions

    git clone https://github.com/afz225/dialect_parse.git
    cd dialect_parse

Then you have to setup a conda environment with the yaml file as follows:

    conda env create -f environment.yaml

If there are any issues with certain libraries not being available just remove them from the requirements file and install them/OS-compatible alternatives using conda install or module load.

We already include the data files using `git-lfs` however if there are any issues retreiving them in the `./data` folder we make the data available on OneDrive (here: https://mbzuaiac-my.sharepoint.com/:f:/g/personal/ahmed_elshabrawy_mbzuai_ac_ae/EiOfyyPHL29Pq65wa9GENMABi2S5cwDZqGLlIJsH7gCl2g?e=x7AoJd). We request that the data not be shared publically because we have special access to this data and we do not yet have permission from the authors to share it publically.

Now, we have three best models for each dialect we explore which are Gulf, Egyptian, and Levantine. We will provide instructions to train and evaluate these models. However, we also finetune BERT on an alignment objective which we detail in the report. We provide the code we used to perform this alignment and finetuning but we do not expect the TAs to run it. Hence it is just there for reference and we will not provide instructions on how to run it (if you would like we can provide these instructions but the code takes a while to run and from our understanding you only expect inference/evaluation code for our best model). The code we run to finetune is:
1. `create_data.ipynb` to generate the data we use for finetuning
2.  `bert_finetune.py` to finetune BERT using huggingface and PyTorch. 

We already ran the finetuning and uploaded the finetuned models to HuggingFace and there is no need to manually download them the code will handle that.

### Training instructions

We provide three scripts for to train a parser using the finetuned BERT models for each dialect you can run them as follows

Gulf:

    ./train_gulf.sh
Egyptian:

    ./train_egy.sh
Levantine:

    ./train_lev.sh

These scripts will train models and place them in the `./models` folder. We already include these trained models in the folder using `git-lfs` but if that does not work for whatever reason we uploaded the models folder to OneDrive (here: https://mbzuaiac-my.sharepoint.com/personal/ahmed_elshabrawy_mbzuai_ac_ae/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fahmed%5Felshabrawy%5Fmbzuai%5Fac%5Fae%2FDocuments%2Fmodels&view=0 ) where they can be downloaded and placed in the directory manually. Just please make sure to have it in the right place which is directly in this directory as a folder called `./models`. However, there is no need to train.

### Evaluation instructions

We report on LAS and UAS in the report. To get these numbers you can run some evaluation commands to evaluate on the test files. We wrote bash scripts to run to perform these evaluations and output them to files called `best_model_eval_[dialect].log`. These files are already present so there is no need to run the scripts if it is not necessary.

If you would like to evaluate on the dev files you would need to open the bash scripts we wrote and modify the `DATA` environment variable to point to the dev file instead of the test file in the data folder.

To run the evaluations of the dialects run the following:

Gulf:

    ./eval_gulf.sh
Egyptian:

    ./eval_egy.sh
Levantine:

    ./eval_lev.sh

to view the output simply read the files outputted called `best_model_eval_[dialect].log`. They will contain the scores.

We have only included the best-performing models for each dialect because, otherwise, we exceed the storage for the repo. If you would like to view other models please reach out to any of the group members. Thank you.
