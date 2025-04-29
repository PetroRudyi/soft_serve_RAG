# MM RAG by Petro Rudiy 

## Install libraries

```pip install -r requirements.txt```

## Load Data and VDB

It is available here, along with all the data you need. VDB_ani_Images unzip in the core project directory.

https://drive.google.com/drive/folders/1Kbt119Sb8jpXvOhk4LeJTXHXrdKunOQd?usp=sharing

## Instructions and Usage

RAG uses vector database ChromaDB and integrates with LLM models by OpenAI. 
Details about pricing: https://platform.openai.com/docs/pricing

## Steps for data preparation:

1) Run **dataset_web_scraping.py** - download the data from https://www.deeplearning.ai/the-batch/ with web scraping
2) Run **load_images.py** - load all images after first script run
3) Run **text_data_prepare_and_emb.py** - preparing and optimization of text information for vector DB.
4) Run **vector_db_create.py** - create VDB with text and images with ChromaDB

## Run

!!! After running all scripts from "Steps for data preparation" the data for RAG is prepared. 

For run **main.py**, use terminal and run ```streamlit run main.py```

## Notes
- OpenAI API Key and Query fields are required.
- A lot of parameters available for changes in **config.py**

## Scripts details
- **main.py** - This script handles the creation of the user interface and the logic for retrieving data to generate a request to the LLM. It collects essential inputs and settings, such as the user query, temperature, model choice, and parameters for retrieving relevant documents and images. Additionally, it displays the model’s response and allows users to save previous queries along with all associated parameters to the cache.
- **AI_bot.py** - This script stores logic and functions for interacting with the vector database, creating a query to the LLM based on the user's query, and receiving output data from the model.
- **config.py** - Stores parameters for training, UI, and user restrictions. Also contains CacheParameters - a class for saving user requests to the cache.


