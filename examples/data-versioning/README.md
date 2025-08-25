# Data Versioning with DVC

## Overview
**Data Version Control (DVC)** extends Git-like workflows to data and ML artifacts.  
It allows you to **track, version, and reproduce datasets and models** without storing huge binaries in Git.  
Instead, DVC stores metadata in Git and pushes large files to remote storage (e.g., ADLS, S3, GCS, Azure Blob).  

This is critical in MLOps because:
- Datasets evolve over time.
- Models must be reproducible on the exact data snapshot they were trained with.
- Production systems need reliable rollback to “last known good” data.

---

## Core Concepts
- **`dvc add`** → Track a dataset or folder, creating a `.dvc` file (pointer to data location + hash).  
- **`dvc push`** → Upload the dataset to remote storage.  
- **`dvc pull`** → Retrieve a dataset for training or debugging.  
- **`dvc.api.get_url()`** → Programmatically resolve dataset paths inside pipelines/scripts.  
- **Pipelines** → Define end-to-end ML workflows (`dvc.yaml`) with reproducibility baked in.

---

## Typical Workflow

**Research Path (Reproducibility Focused):**
Designed for the model development and experimentation phase. Primary goal is to ensure that training runs can be exactly reproduced, at any time.

`ADLS` → `ADF/Spark` → `ADLS snapshot` → `dvc add` → `Git commit` → `ML training job` → `DVC + Git (ensure exact dataset lineage)`

- **ADLS (Azure Data Lake Storage)**: Process begines with raw data stored in data lake. Source of truth for all data. Gathered from any number of means, API, database, etc.
- **ADF/Spark (Azure Data Factory/Spark)**: Automated data processing job, orchestrated by ADF. Uses compute engine like Spark to clean, transform, and feature engineer raw data.
- **ADLS Snapshot**: Creating a specific, versioned copy of the processed dataset. This locks the dataset in for the training run.
- **dvc add**: DVC command to track the specific data snapshot. This creates a metadata file that points to the actual data's location and records its hash.
- git commit: Store the metadata file that points to the data location alongside the model and training code. Now your git commit links the exact data snapshot for comprehensive reproducability.
- **ML training job**: The training job uses specific combination of code and data to train the model.
- **DVC + Git**: The result is a fully traceable lineage. By checking oiut a specific commit, you see the xact version of data that was used to train the model.

The research path prioritizes versioning everything to remove ambiguity. It answers the question, "Which exact version of the data did we use to get this result?"


**Production Path (Performance Focused):**

`ADLS` → `ADF/Spark` → `ADLS snapshot/latest` → `Inference directly from ADLS`

- **ADLS (Azure Data Lake Storage)**: Similar to the research path, raw data is the starting point.
- **ADF/Spark**: Data is processed and prepared, but the output is treated differently.
- **ADLS snapshot/latest**: Instead of creating a permanent snapshot, the production system works with the latest available data or a rolling window of recent data. The focus is on **freshness**, not immutability.
- **Inference directly from ADLS**: This is the crucial step. The model in production doesn't rely on DVC or a versioned dataset for its inputs. It's often set up to read directly from a designated production data path in the data lake. For real-time applications, this might be a *streaming source* or a *fast lookup table*. The system simply takes the latest available data, passes it to the model, and serves the prediction.

The production path sacrifices some of the strict versioning of the research path for operational efficiency. The question it answers is, "How can we get the most up-to-date prediction to our users as fast as possible?"

---

## Example: Using DVC with Azure Data Lake (ADLS)

```bash
# Track a curated dataset snapshot
dvc add abfss://datalake/curated/transactions/2025-08-23/

# Commit metadata
git add transactions.dvc .gitignore
git commit -m "Track curated transactions snapshot (2025-08-23)"

# Push data to remote
dvc remote add adls_remote azure://<container>/<path>
dvc push -r adls_remote
```

**In training code**:

```python
import dvc.api
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Resolve dataset path from DVC
data_path = dvc.api.get_url(
    "curated/transactions/",
    repo="https://github.com/org/data-versioning",
    rev="experiment-42"
)

df = spark.read.parquet(data_path)
```

## Best Practices

- **Don’t use** `dvc.api.read` for big datasets as it streams file contents (not suited for parquet/CSV tables).
- Store dataset metadata in Git, blobs in ADLS/S3/Blob.
- Track model artifacts (e.g., `model.pkl`, `model.onnx`) with DVC for reproducibility.
- Tag DVC commits with semantic labels (`v1.0-data`, `fraud-detection-march2025`).
- Combine with **MLflow**: Log the DVC commit hash or .dvc file path used in training, so you know exactly which dataset produced a model.

## Integrations
- **MLflow**: Store DVC commit hash as mlflow.log_param("dataset_commit", <hash>) for full experiment lineage.
- **Delta Lake / Hudi**: Use for scalable time-travel queries; DVC still useful for external snapshots and lightweight reproducibility.
- **Orchestration (Airflow / Kubeflow / Prefect)**: Automate `dvc pull` in training pipelines to guarantee the right dataset version is always used.
- **CI/CD**: Run `dvc pull` in CI workflows to test model training against a fixed dataset snapshot.