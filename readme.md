# To initialize environment

    mamba env remove -n kstu_model_education_analisys_tool

    mamba env create -f environment.yml

    mamba activate kstu_model_education_analisys_tool

    mamba env export > environment_test.yml

and use

    init.sh

# To run

    python main.py

Select R2D2 algorithm, fill parameters and press "start"

# Prototypes in Colab

The latest code demonstrated in V13 of draft, it has been tested with PPO, DQN and A2C.