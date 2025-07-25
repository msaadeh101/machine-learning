# Table of Contents
- [Machine Learning](#machine-learning)
- [Tools](#tools)
    - [Python and Libraries](#python-and-libraries)
    - [Anaconda](#anaconda)
    - [Jupyter Notebooks](#jupyter-notebooks)
    - [Data Version Control](#dvc)
    - [MLflow](#mlflow)
    - [DagsHub](#dagshub)
    - [Langchain](#langchain)

# Machine Learning

- **Machine Learning** is a subset of AI where you learn patters from data to make predictons or decisions without explicit programming. 
- You feed data to `algorithms` (linear regression, decision trees, neural networks) to produce a `model` - a serialized function that maps inputs to outputs.
- The models are trained on training data, but evaluated on `unseen` data.

- Train a model to predict engineer performance based on PRs, slack messages, incidents resolved:

```python
# Import pandas for data manipulation
import pandas as pd
# define a target label
performance = ["high", "average"]
# create a dictionary
# each developer has 5 features (pr_count, review_latency_hours, story_points, slack_messages_sent, incidents_resolved)
# each developer has 1 target label (performance)
data = {
    "pr_count": [12, 5, 18, 7, 3],
    "review_latency_hours": [4.2, 12.0, 3.5, 10.5, 16.0],
    "story_points": [35, 20, 40, 25, 15],
    "slack_messages_sent": [130, 80, 160, 90, 50],
    "incidents_resolved": [2, 0, 3, 1, 0],
    "performance": ["high", "average", "high", "average", "average"]
}
df = pd.DataFrame(data) # create a dataframe for manipulation

# train the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Encode the target label
# Convert text labels to numbers
df["performance"] = df["performance"].map({"average": 0, "high": 1})

X = df.drop("performance", axis=1) # X: features (all columns except performance) - input data
y = df["performance"] # y: Target: performance column is what we want to predict

# Train/test split into 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4 rows for training, 1 for testing

# Train a simple classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate and print performance metrics
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Define new_data for prediction
new_data = [[10, 8.0, 30, 100, 1]]  # Same format as training features
# On any new data
prediction = model.predict(new_data)
```
- `X_train` - your training features (pandas dataframe or numpy array).
- `y_train` - your training labels/target values.
- `X_test` - your test features.
- `y_test` - your test labels.

## Tools

### Python and Libraries

- `Python` (core libraries and general-purpose libraries):
    - `Numpy`: Provides n-dimensional array objects and functions for linear algebra and random number capabilities.
    - `Pandas`: Provides structure like DataFrames (tabular data) and Series (1-dimensional labeled arrays) for cleaning, transforming, and visualizing data.
    - `Matplotlib`: Library for static/animated and interactive visualizations. Options for line plots, scatter plots, bar charts, histograms.
- `Scikit-learn`: Machine learning library for supervised and unsupervised learnings including classification, regression, clustering, dimensionality reduction, and model selection.
- `TensorFlow`: Developed by Google, Strong in deep learning and supports GPU and CPU.
- `PyTorch`: Developed by Facebook's AI Research Lab (FAIR). Used for rapid prototyping for computer vision or language processing.
- `Keras`: The official high-level neural networks API for Tensorflow (or CNTK or THeano).
- `XGBoost`: Optimized gradient boosting library, implements ML algorithms under the Gradient Boosting framework. Parallel tree boosting solves data science problems efficiently.

- **Sklearn** is a python library containing algorithms for:
    - **Classification**: Identifying which category an object belongs to.
        - Use cases: Spam detection, image recognition.
        - Algorithms: Gradient boosting, Nearest Neighbors, Random Forest, Logistic reg.
    - **Regression**: Predicting a continuous-valued attiribute.
        - Use cases: Drug respons, stock price tracking.
        - Algorithms: Gradient boosting, neirest neighhbors, random forest, ridge.
    - **Clustering**: Automatic grouping of similar objects into sets
        - Use cases: Customer segmentation, grouping outcomes.
        - Algorithms: K-means, HDBSCAN, hierarchical clustering.
    - **Dimensionality Reduction**: Reducing number of random variables to consider.
        - Use cases: Visualization, increase efficiency.
        - Algorithms: PCA, feature selection, non-negative matrix factorization.
    - **Model Selection**: Improved accuracy via parameter tuning.
        - Use cases: Improved accuracy via param tuning.
        - Algorithms: Grid search, cross validation, metrics.
    - **Preprocessing**: Feature extraction and normalization.
        - Use cases: Transforming input data
        - Algorithms: preprocessing, feature extraction.

### Anaconda

- **Anaconda** is an environment and package manager. `conda` is the command line tool.
    - With anaconda, you can create isolated environments with only the packages and versions you need, and export them to any platform.

```bash
# Create an anaconda environment
conda create --name env313 python=3.13

# Activate env and enter
conda activate env313

# Install pytorch
conda install -c pytorch pytorch

# List packages from inside the env
conda list

# Deactivate the environment/exit
conda deactivate

# List all envs and locations
conda info --envs

# Show channel urls
conda list --name env313 --show-channel-urls

# export an environment as yaml
conda env export -n env313 --from-history > env313.yml

# Create from yaml file
conda env create -f env313.yml
```

### Jupyter Notebooks

- **Notebooks** are interactive environments where you can explore data, build and test models, visualize results, document experiments.
    - Best used for experimentation and data science.

- Notebooks contain: `cells` (code + markdown), `metadata` (kernel info, tags), `outputs` (visualizations, errors), `execution_count` (order of cell runs)

- MLOps using Jupyter (notebooks)
1. Explore with `Notebook`.
1. Track Runs with `MLFlow/W&B` directly from the notebook.
1. Refactor to pipeline.
1. Automate in CICD.
1. Deploy model and monitor.

```bash
# Clean outputs before committing notebook
nbstripout your_notebook.ipynb

# automate notebooks with papermill
# papermill is a python tool that lets you automate
# and parameterize jupyter notebooks
papermill train.ipynb output.ipynb -p lr 0.01
# overwrite parameter lr with 0.01

# Convert your jupyter notebook to script
juptyer nbconvert --to script your_notebook.ipynb
```

- The `ipykernel` allows interactive computing in python, for example.

### DVC

- **Data version control (DVC)** is the concept of managing and versioning extremely large volumes of data (images, audio, video, text) for your ML modeling.
    - `.dvc` files contain metadata and checksums, not actual data.
    - Always run `dvc pull` after `git pull` when collaborating.
    - Actual data files are auto-created in the `data/.gitignore`

```bash
# Install DVC
pip install dvc
# intialize DVC in your project AKA a git repo
dvc init
# Initialize with remote storage
dvc init --subdir
# Data tracking, adding .dvc files
dvc add data/dataset.csv
dvc add models/
# Add remote storage (works with ssh, s3, remote)
dvc remote add -d myremote s3://mybucket/dvcstore
# Push and pull data
dvc push # push all tracked data
dvc push data/dataset.csv
dvc pull # pull all tracked data
dvc pull data/dataset.csv

# Create a pipeline stage
dvc stage add -n preprocess \
  -d data/raw.csv \
  -o data/processed.csv \
  python preprocess.py

# Create a training stage
dvc stage add -n train \
  -d data/processed.csv \
  -d src/train.py \
  -o models/model.pkl \
  -M metrics.json \
  python src/train.py

# Run the pipeline
dvc repro # run entire pipeline
dvc repro train # run specific stage

# Show pipeline as dag graph
dvc dag

# Data versioning and branches
# Create a new experiment branch
git checkout -b experiment-new-features
dvc checkout # switch DVC files to match git branch

# After making data changes
dvc add data/new_dataset.csv
git add data/new_dataset.csv.dvc

# Compare metrics between branches and commits
dvc metrics diff # compare with previous commit
dvc metrics diff HEAD~1 # Compare with specific commit
dvc metrics diff main experiment # compare between brances

# Status and Info
dvc status # local changes
dvc status --cloud # remote changes
dvc list . --dvc-only # list dvc-tracked files
dvc get-url data/model.pkl # get remote url for file

# Show cache directory location
dvc cache dir

dvc gc # remove unused cache files
dvc gc --workspace # remove files not in current workspace
```

### MLFlow

- **MLflow** is an open-source platform from Databricks, for managing the machine learning lifecycle.
    - Provides tools for experiment tracking, model packaging, versionining, and deployment.
    - Find out your best model and how to deploy it, and compare models.
    - MLflow has a UI where you can view your experiments, paramters, metrics, tags, etc.

- Tip: Install conda forge for MLflow fof enhanced capabilities: `conda install -c conda-forge mlflow`

#### MLFlow for ML

```python
pip install mlflow
# Or with extras like sklearn, tensorflow
pip install mlflow[extras]

# Run a local version of MLflow
mlflow ui
# Or
# Create a managed MLFlow tracking server
mlflow server --host 127.0.0.1 --port 8080

# Set tracking server URI if not Databricks managed
# We are using the localhost it created
import mlfow
mlflow.set_experiment("First Experiment")
mlfow.set_tracking_uri(uri="http://<host>:<port>")

# Start an mlflow run after you train your model
with mlfow.start_run(my_run_name):
    mlflow.log_params(params)
    mlflow.log_metrics({ # Taken from python report dict
        'accuracy': report_dict['accuracy'],
        'recall_class_0': report_dict['0']['recall'],
        'recall_class_1': report_dict['1']['recall'],
        'f1_score_macro': report_dict['macro avg']['f1-score'],
    }) # Output of model training
    mlflow.sklearn.log_model(lr, "Logistic Regression") # Model name, artifact path
```

- You can then view your Model metrics, system metrics, artifacts in the UI.
    - Exports the model as `pkl` file.
    - Contains python and conda environment yamls.

- Each run in MLFlow has a `Run ID` that uniquely identifies that run, and can be registered in the **model registry**.

```python
model_name = "XGB-Smote"
run_id = input("Enter runID: ")
model_uri = f"runs:/{run_id}/{model_name}
result = mlflow.register_model(
    model_uri, model_name
)
```

- When you are testing new models, you call new ones `challenger`, since the one in production is `champion`.

- Load a model once its registered:

```python
model_version = 1
model_uri = f"models:{model_name}/{model_version}"
loaded_model = mlflow.xgboost.load_model(model_url) # Specifically the XGBoost flavor
# Go to the model registry, download it, and perform operations
y_pred = loaded_model.predict(X_test)
y_pred[:4]
```

- **MLFlow pipelines** are `yaml` defined pipelines that define your data, sql, data splits, data training, evaluation, metrics etc from **Databricks**.

- Pipeline command syntax (used in a notebook), referenced from `pipeline.yaml`:

```python
from mlflow.pipelines import Pipeline
p = Pipeline(profile="databricks")
p.clean()
p.inspect()
p.run("ingest") # pull data into local folder
p.run("split")  # split into foundation, test
p.run("transform") # add a column for example
p.run("train")
p.run("evaluate")
```

#### MLFlow for GenAI

### Cohere

- **Cohere** is an `embedding model`, it ransforms text into numerical vectors that can then be uploaded into a vector store.

- Cohere is paid, but open source `sentence-transformers` exist, which are completely free.

```python
# This creates an embedding (a list of numbers)
text = "The cat sat on the mat"
embedding = cohere.embed(text)
# Result: [0.2, -0.8, 0.1, 0.4, ...]  (1536 dimensions)
```

### Pinecone

- **Pinecone** is a vector store, a managed database for storing and searching embeddings to work alongside your LLM applications like RAG.

- Open source vector databases: `chromadb`, `qdrant`: More user friendly, `faiss`: more complicated.

- Example Flow:
1. Store documents -> Pinecone (as embeddings)
1. User asks question -> Search Pinecone for relevant docs
1. Retrieved docs -> question -> Send to LLM
1. LLM generates answer using context

### Dagshub

- **DagsHub** is the Github for ML, it provides a central hub for ML engineers to collaborate on ML projects.
    - Version control for Code, Data, Models, and Experiments.
    - Integrates with MLflow for experiment tracking.
    - Visualize ML pipelines flow. (Ingest -> Split -> Transform -> Train -> Evaluate -> Register -> Predict)
    - Supports integrating with annotation tools, to collaborate on labeling data directly.
    - Use `.dvc` files that point to your data stored on Dagshub or another data storage.

### LangChain

- **LangChain** is a framework that uses existing LLMs (chatGPT, Gemini, Claude, etc), to build applications using the pretrained models and chain sequences of operations together to create complex workflows. 
- Prompt templates, parsers, or functions are known as **Runnables**. WHen you chain Runnables together using `|` you create a **RunnableSequence**.
    - `Chains`: sequences of operations.
    - `Prompts`: templates for structuring input into language models.
    - `Memory`: components that let your app remember previous conversations. Crucial for **multi-turn chatbots**.
    - `Agents`: can use tools and make decisions about actions to take based on input. Call APIs, search DBs, perform functions.
    - `Retrievers`: help implement RAG, pulling relevant info into the LLM's knowledge.

- Example chain: 
1. Take user question -> 
1. search company docs ->
1. format the retrieved info into a prompt -> 
1. Sends it to LLM -> 
1. Parses/validates response

- [Tutorial](https://python.langchain.com/docs/tutorials/llm_chain/)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Define your components
prompt = ChatPromptTemplate.from_template("Tell me a short story about {topic}.")
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# Chain them using the pipe operator
chain = prompt | llm | output_parser

# Invoke the chain
result = chain.invoke({"topic": "a brave knight"})
print(result)
```

- Example with multiple PromptTemplates to create a **RunnableParallel**, modifying initial prompts.

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define some components
question_gen_prompt = ChatPromptTemplate.from_template("Generate a rephrased question for: {question}")
answer_gen_prompt = ChatPromptTemplate.from_template("Answer the question: {question}")
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# Run two branches in parallel
chain = RunnableParallel(
    rephrased_question=question_gen_prompt | llm | output_parser,
    original_question=RunnablePassthrough() # Pass the original input through
)

result = chain.invoke({"question": "What is the capital of France?"})
print(result)
# Example output: {'rephrased_question': 'Could you tell me the capital city of France?', 'original_question': 'What is the capital of France?'}
```

- **RunnableLambda** is any Python function wrapped into a Runnable.

```python
from langchain_core.runnables import RunnableLambda

def capitalize_string(text: str) -> str:
    return text.upper()

capitalize_runnable = RunnableLambda(capitalize_string)

chain = RunnablePassthrough() | capitalize_runnable

result = chain.invoke("hello world")
print(result) # Output: HELLO WORLD
```

## Cloud Services

### Google

#### Vertex AI

