# Folder struture
- **aggregation**: Houses scripts for merging data from different collections within the database into a standardized format for training. Each script is named following the pattern `agg_<name_of_collection>`. The final aggregation is performed by the `aggregate.py` script, which outputs the data into CSV files. The types of aggregations include:
    - **Prompts**: Aggregates data into a format suitable for fine-tuning models with specific instructions.

- **creation**: Contains scripts for parsing data from various sources into the database. This stage retains data formats as close to their original as possible. The scripts are organized into subcategories:
    - **catalog**: Parses medical catalogs, such as ICD-10GM or ATC codes.
    - **corpus**: Handles German medical corpora like bronco150 or ggponc2, parsing general and NER formats when available.
    - **english**: Processes medical corpora in languages other than German, focusing exclusively on NER parsing.
    - **utils**: Provides utilities supporting various creation scripts, including database operations, NER parsing, web scraping, and token counting.
    - **web**: Parses medical web sources, such as Wikipedia, DocCheck, or texts from Gesund Bund.


- **description**: Consists of scripts that analyze and describe the data stored in the database, such as the distribution of data types, the number of tokens, etc.
