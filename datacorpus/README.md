# Datacorpus
This directory includes scripts to:
- Create the datacorpus database.
- Aggregate data from different sources into a standardized format used for training the models.
- Describe the data stored in the database with plots and statistics.

# Folder structure
**Aggregation**: Houses scripts for merging data from different collections within the database into a standardized format 
for training. The standardized format is either one for NER. This includes two arrays, one for ner_tags, the other for
the tokens/words. The other standardized format is for instruction tuning which includes the prompt and response in
a text field. Each script is named following the pattern `agg_<name_of_collection>`. The final aggregation is performed 
by the `aggregate.py` script, which outputs the data into jsonl files. The subscripts include:
- **agg_bronco.py**: Transforms the Bronco dataset into an NER and instruction tuning format. Includes the task 
extraction and normalization.
- **agg_cardio.py**: Transforms the Cardio:DE dataset into an NER and instruction tuning format. Includes only the task 
extraction.
- **agg_catalog.py**: Transforms medical catalog data, such as icd10gm code descriptions, into an instruction tuning
format.
- **agg_ggponc.py**: Transforms the GGPONC2 dataset into an NER and instruction tuning format. Includes only the task 
extraction.
- **agg_synthetic.py**: Transforms the synthetic data from the database into the instruction tuning format. This 
includes only the task summarization.
- **prompts.py**: Contains the prompts for the instruction tuning format. The prompts are organized by the task type.

**Creation:**
Contains scripts for parsing data from various sources into the database. This stage retains data formats as close to 
their original as possible. The scripts are organized into subcategories:
- **catalog**: Parses medical catalogs into the database. Includes the ATC, ICD10GM and OPS2017 catalogs.
- **corpus**: Handles German medical corpora parsing general and NER formats when available. This includes the
Bronco, Cardio:DE, CLEF2019 eHealth, GGPONC2 and JSYNCC corpora. Additionally, the creation for the synthetic data
is also part of this category.
- **english**: Processes medical corpora in languages other than German, focusing exclusively on the NER format.
- **web**: Parses the medical web sources Wikipedia, DocCheck, or texts from GesundBund.de.

**Utils:**
Provides utilities supporting various aspects of all scripts. Mainly includes NER and webscraping utilities.


**Description:**
Consists of scripts that analyze and describe the data stored in the database, such as the distribution of data types, 
the number of tokens, number of entities and more. Description scripts are available for the following collections:
- **bronco**: Describes the Bronco collection.
- **cardio**: Describes the Cardio:DE collection.
- **ggponc**: Describes the GGPONC2 collection.
- **synthetic**: Describes the synthetic collection.
- **prompts**: Describes the prompts used for instruction tuning. For this the `prompts.jsonl` file needs to be 
available.
