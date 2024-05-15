# Bachelor-Thesis: Transformer based Information extraction and summarization of medical documents
TODO: project description here

## Developers
- Nicolas Gujer ([(nicolas.gujer@protonmail.com)](mailto:nicolas.gujer@protonmail.com))
- Jorma Steiner ()

## Disclaimer
This project is part of a bachelor thesis and is not meant to be used in a production environment. The code is not
optimized for performance and is not guaranteed to work in all environments. We won't provide any support for this
and won't be responsible for any damage caused by the usage of this project.

## Documentation
This and all the other README files are part of the documentation of the project. They are meant to give an overview
of the project and its structure. Additionally, the code is well documented with python docstrings and comments.

## Folder Structure
The project is split up into multiple folders, with each folder having a specific purpose for the project.
In each subfolder is another README that describes the details of the folder and its contents.
The following is a brief description of each folder:
- **datacorpus:** A part of this corpus was the creation of an extensive dataset for the training of a
large language model (LLM). This datacorpus consists of multiple medical sources. All the data is stored in a MongoDB
database. The datacorpus folder contains the scripts to parse the data, store it in the databse and aggregate it
into a format suitable for training a model.
- **demo:** TODO: implement :) 
- **shared:** In here are mostly helper functions that encapsulate functionality that is used in multiple parts of the
project. This is to avoid code duplication and to make the code more readable. 
- **training:** This folder contains the scripts to train a large language model (LLM) on the datacorpus. The training
is done with the Huggingface Transformers library. The training is done on a GPU, so make sure to have a compatible
GPU available. There are scripts available to train e causal language modeling (clm) LLM and multiple BERT variants 
for token classification (NER) and classification.
- **validation:** This folder contains the scripts to validate the trained model. This includes the evaluation of the
model on a test set with metrics such as Precision, Recall and the F1-Score. 

# Installation
### Dependencies
For this project a couple of dependencies are required. There won't be a guide on how to install these dependencies, 
as they are well documented on their respective websites. The following dependencies are required:
- **Python:** Most of the project is written in Python, so make sure to have Python installed on your machine. 
We personally recommend at least Python Version 3.11 as we tested the functionality of the project with this version.
But generally any version of Python 3.8 or newer should work.
- **GPU Environment:** Even tho the training, validation and inference of the models in this project can be done on a CPU,
it is highly recommended to use a GPU which will significantly speed up the process. For this project we recommend
the usage of a NVIDIA GPU with CUDA version 12.4 installed. Theoretically other GPUs should work as well, but we didn't test this
and cant guarantee that it will work. The [CUDA installation](https://docs.nvidia.com/cuda/) guide provides all the 
information you need to install CUDA on your machine.

### Python installation
All python packages that are required for this project are documented in the `requirements.txt` file. To install them
simply run the following command in the root directory of the project:
```shell
pip install -r requirements.txt
```
After this you should be able to execute the individual files in the project without any issues.

### .env file setup
For some parts of the project, environment variables are required. These are stored in file called `.env` file in the 
root directory. The following environment variables are required:
- **MONGO_URL:** The connection string to the MongoDB database. Several functionalities of the project require a 
connection to a MongoDB database to store and retrieve data. This string should be in the following format:
`mongodb://<username>:<password>@<host>:<port>`. Replace the placeholders with the actual values. For further
information you can check the [MongoDB documentation](https://www.mongodb.com/docs/manual/reference/connection-string/).
- **OPENAI_KEY:** For synthetic data generation, the OpenAI API is used. To use this API, you need an API key. 
For further information you can check the [OpenAI documentation](https://platform.openai.com/docs/api-reference).

### Tip for typical python error
If you encounter an error that looks similar to this while trying to run one of the python scripts:
```
ModuleNotFoundError: No module named 'shared'
```
You can fix this by adding the root directory of the project to the PYTHONPATH. This can be done with the following 
command. Make sure to replace the path within the angle brackets with the actual path to the root directory of the
project:
```shell
export PYTHONPATH=<path/to/root/directory>:$PYTHONPATH
```