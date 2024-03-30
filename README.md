# ChatGPT Trading Decision and Insights App

This is an AI app to get **real-time** trading decision for Nifty 50 stocks along with various financial aspects of that stock . The project exposes an HTTP REST endpoint to answer user queries about what the **price-action decision** for a stock for that day will be or what stocks have the particular price-action-decision for the day and also explain why the decision is made , sighting various financial aspects of the stock . It uses Pathway’s [LLM App features](https://github.com/pathwaycom/llm-app) to build real-time LLM(Large Language Model)-enabled data pipeline in Python and  leverages OpenAI API [Embeddings](https://platform.openai.com/docs/api-reference/embeddings) and [Chat Completion](https://platform.openai.com/docs/api-reference/completions) endpoints to generate AI assistant responses.

This trading prediction application , on its core is built on **machine learning classifiers** to make buy, sell, or hold predictions based on technical indicators calculated from historical OHLCV (Open, High, Low, Close, Volume) data of financial assets. The application preprocesses the data, engineers features, trains multiple classifiers, and makes predictions on new data.

The **prediction model** runs everytime the app is called and yfinance library helps in getting the daily stock prices . Then the predcited decision dataframe for each of nifty 50 stocks are combined together and compliled into a csv file which in-turn is converted to Jsonlines where each line expects to have a `doc` object as in . Now we also have another jsonl file containing the information about each of the indicators and columns and their implicatons on the decision . This jsonl file is combined to the former jsonl file to give the final jsonl file , which acts as input for the embedding process. 

- Input data in form of Jsonlines are used so as to improve the efficiency of the whole process.

## Features

- Provides stock decision and insights powered by a model with very good accuracy 
- Offers user-friendly UI with [Streamlit](https://streamlit.io/).
- Filters and presents decisions and insights on stocks based on user queries 
- Data and code reusability . 
- Extend data sources: Using Pathway's built-in connectors for JSONLines, CSV, Kafka, Redpanda, Debezium, streaming APIs, and more.

## Further Improvements

There are more things you can achieve and here are upcoming features:

- Incorporate additional data from external APIs, along with various files (such as Jsonlines, PDF, Doc, HTML, or Text format), databases like PostgreSQL or MySQL, and stream data from platforms like Kafka, Redpanda, or Debedizum.
- Merge data from these sources instantly.
- Convert any data to jsonlines.
- Beyond making data accessible UI, the LLM App allows you to relay processed data to other downstream connectors, such as BI and analytics tools. For instance, set it up to **receive alerts** upon changing stock prices for stocks of choice
- More in-depth over-view of the financial aspects of the queried stocks.
- Future prices prediction models combined with RAG can ensure to an efficient improvement.
- Can be made usable for many more stocks other than the nifty 50 stocks

  

## Code sample

It requires only few lines of code to build a real-time AI-enabled data pipeline:

```python
# Given a user question as a query 
    query, response_writer = pw.io.http.rest_connector(
        host=host,
        port=port,
        schema=QueryInputSchema,
        autocommit_duration_ms=50,
    )

    # Real-time data coming from external data sources such as jsonlines file
    stock_data = pw.io.jsonlines.read(
        "./examples/data/stock_predict_total.jsonl",
        schema=DataInputSchema,
        mode="streaming"
    )

    # Compute embeddings for each document using the OpenAI Embeddings API
    embedded_data = embeddings(context=stock_data, data_to_embed=stock_data.doc)

    # Construct an index on the generated embeddings in real-time
    index = index_embeddings(embedded_data)

    # Generate embeddings for the query from the OpenAI Embeddings API
    embedded_query = embeddings(context=query, data_to_embed=pw.this.query)

    # Build prompt using indexed data
    responses = prompt(index, embedded_query, pw.this.query)

    # Feed the prompt to ChatGPT and obtain the generated answer.
    response_writer(responses)

    # Run the pipeline
    pw.run()
```

## Use case

[Open AI GPT](https://openai.com/gpt-4) excels at answering questions, but only on topics it remembers from its training data. If you want GPT to answer questions about unfamiliar topics such as:

- Recent events after Sep 2021.
- Your non-public documents.
- Information from past conversations.
- Real-time data.
- Including discount information.

The model might not answer such queries properly. Because it is not aware of the context or historical data or it needs additional details. In this case, you can use LLM App efficiently to give context to this search or answer process.  See how LLM App [works](https://github.com/pathwaycom/llm-app#how-it-works).

For example, a typical response you can get from the OpenAI [Chat Completion endpoint](https://platform.openai.com/docs/api-reference/chat) or [ChatGPT UI](https://chat.openai.com/) interface without context is:

![chatgpt_screenshot1](/assets/chatgpt1.png)

![chatgpt_screenshot2](/assets/chatgp2.png)

As you can see, GPT responds only with suggestions on how determine decisions but it is not specific and does not provide exactly where or what decision explicitly and so on.

To help the model, we give knowledge of stock data from a reliable data source (it can also be JSON document, or data stream in Kafka) to get a more accurate answer.  There is a jsonl file with the following columns of data on the daily ohlcv data and indicators data and finally preiction data for each of the 50 stocks of nifty index for the last 10 days along with the information about each indicator and their influencce on the decision . 

After we give this knowledge to GPT through the jsonl file, look how it replies:

![sample_run](/assets/sample_run.png)

 The cool part is, the app is always aware of changes in the daily prices . If you just open the app for the day , the LLM app does magic and automatically updates the AI model's response for the day.

## How the project works

The sample project does the following procedures to achieve the above output:

1. Prepare search data:
    1. Generate: When the app runs , the [prediction model](/examples/predictionmodel/stock_decision_prediction.py) runs and computes stock decisions for the day and compiles all data including the indicators explanation jsonl file into one jsonl file ready to be given as input to the embedding process
    2. Chunk: Documents are split into short, mostly self-contained sections to be embedded.
    3. Embed: Each section is [embedded](https://platform.openai.com/docs/guides/embeddings) with the OpenAI API and retrieve the embedded result.
    4. Indexing: Constructs an index on the generated embeddings.
2. Search (once per query)
    1. Given a user question, generate an embedding for the query from the OpenAI API.
    2. Using the embeddings, retrieve the vector index by relevance to the query
3. Ask (once per query)
    1. Insert the question and the most relevant sections into a message to GPT
    2. Return GPT's answer

## How to run the project

Example only supports Unix-like systems (such as Linux, macOS, BSD). If you are a Windows user, we highly recommend leveraging Windows Subsystem for Linux (WSL) or Dockerize the app to run as a container.

### Run with Docker

1. [Set environment variables](#step-2-set-environment-variables)
2. From the project root folder, open your terminal and run `docker compose up`.
3. Navigate to `localhost:8501` on your browser when docker installion is successful.

### Prerequisites

1. Make sure that [Python](https://www.python.org/downloads/) 3.10 or above installed on your machine.
2. Download and Install [Pip](https://pip.pypa.io/en/stable/installation/) to manage project packages.
3. Create an [OpenAI](https://openai.com/) account and generate a new API Key: To access the OpenAI API, you will need to create an API Key. You can do this by logging into the [OpenAI website](https://openai.com/product) and navigating to the API Key management page.

Then, follow the easy steps to install and get started using the sample app.

### Step 1: Clone the repository

This is done with the `git clone` command followed by the URL of the repository:

```bash
git clone https://github.com/CodeAceKing382/Stocks-Insight-App
```

Next,  navigate to the project folder:

```bash
cd Stocks-Insight-App
```

### Step 2: Set environment variables

Create `.env` file in the root directory of the project, copy and paste the below config, and replace the `{OPENAI_API_KEY}` configuration value with your key. 

```bash
OPENAI_API_TOKEN={OPENAI_API_KEY}
HOST=0.0.0.0
PORT=8080
EMBEDDER_LOCATOR=text-embedding-ada-002
EMBEDDING_DIMENSION=1536
MODEL_LOCATOR=gpt-3.5-turbo
MAX_TOKENS=200
TEMPERATURE=0.0
```

### Step 3: Install the app dependencies

Install the required packages:

```bash
pip install --upgrade -r requirements.txt
```
### Step 4 (Optional): Create a new virtual environment

Create a new virtual environment in the same folder and activate that environment:

```bash
python -m venv pw-env && source pw-env/bin/activate
```

### Step 5: Run and start to use it

You start the application by navigating to `llm_app` folder and running `main.py`:

```bash
python main.py
```

When the application runs successfully, you should see output something like this:

![pathway_progress_dashboard](/assets/pathway_progress_dashboard.png)

### Step 6: Run Streamlit UI for file upload

You can run the UI separately by navigating to `cd examples/ui` and running Streamlit app
`streamlit run app.py` command. It connects to the Discounts backend API automatically and you will see the UI frontend is running http://localhost:8501/ on a browser:

  ![screenshot_ui_streamlit](/assets/streamlit_ui.png)


