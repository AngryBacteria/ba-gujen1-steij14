# Shared
This directory contains shared helper function that are used by multiple other directories of this project. 
This includes:
- **logger.py:** A simple logger implementation that logs to the console and to a file. The format inside the console
is multicolored indicating the levels of logs (INFO, WARNING, ERROR, DEBUG). Both the console and file logs have
a timestamp and the name of the file that the log was called from.
- **clm_model_utils.py:** Functions to interact with the Deocder LLMs are stored in here. This mainly includes the 
following functionalities:
  - Loading Model and Tokenizer for the usage in the project.
  - Patching Model and Tokenizer with special prompt/chat formats. This includes saving it into the Model/Tokenizer
  and expanding the vocabulary. If you want to add new formats you can check out this 
  [GitHub Repository](https://github.com/chujiezheng/chat_templates).
  - Uploading Model/Tokenizer to Huggingface. You have to first run `huggingface-cli login` in the console.
  - Functions to generate text / output with the Models.
  - Counting the tokens of a list of strings.
- **mongodb.py:** All required functions to interact with the MongoDB Database. 