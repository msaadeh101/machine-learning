# MLOps Playbook

## Table of Contents

### 1. Foundation
- [1.1 MLOps Overview](#11-mlops-overview)
- [1.2 Team & Roles](#12-team--roles)
- [1.3 Maturity Assessment](#13-maturity-assessment)

### 2. Infrastructure
- [2.1 Compute & Storage](#21-compute--storage)
- [2.2 Containers & Orchestration](#22-containers--orchestration)
- [2.3 Security](#23-security)

### 3. Data & Model Management
- [3.1 Version Control Strategy](#31-version-control-strategy)
- [3.2 Data Pipelines](#32-data-pipelines)
- [3.3 Model Registry](#33-model-registry)

### 4. ML Pipelines
- [4.1 Training Automation](#41-training-automation)
- [4.2 Model Validation](#42-model-validation)
- [4.3 Deployment Strategies](#43-deployment-strategies)

### 5. Testing & Quality
- [5.1 Testing Framework](#51-testing-framework)
- [5.2 Data Quality](#52-data-quality)
- [5.3 Performance Testing](#53-performance-testing)

### 6. Operations
- [6.1 Monitoring & Alerting](#61-monitoring--alerting)
- [6.2 CI/CD Automation](#62-cicd-automation)
- [6.3 Incident Response](#63-incident-response)

### 7. Tools & Practices
- [7.1 Technology Stack](#71-technology-stack)
- [7.2 Best Practices](#72-best-practices)
- [7.3 Case Studies](#73-case-studies)

### Appendices
- [A. Tool Comparison](#a-tool-comparison)
- [B. Templates & Examples](#b-templates--examples)


## 1.1 MLOps Overview

- The MLOps Lifecycle is a continuous loop that can be broken down into several key stages. It is an iterative cycle of experimentation, deployment and improvement.

### Experimentation and Development

- This is the initial phase where data scientists and ML engineers explore data, train models, and experiment with different algorithms. The focus is rapid iteration and tracking experiments.

**Key Activities**:
- **Data Exploration and Feature Engineering**: Prepare and transform raw data into features suitable for model training.
- **Model Training and Evaluation**: Building and training various models, then evaluating performance based on metrics like accuracy, precision, and recall. Includes cross-validation and train/validation/test splits.
- **Experiment Tracking**: Logging parameters (data based), hyperparameters (model based), metrics, artifacts (models, datasets), and environment details (libraries, runtime) for each experiment, ensuring reproducability and a clear history.

### CICD for ML

- Once a model is ready for deployment, the CI/CD pipeline automates the process of getting it into production.

**CI for ML**:
- Differs from Traditional CI. Involves testing code as well as data and the trained model. A robust CI pipeline automatically retrains the models when new data becomes available or underlying code changes.

**CD for ML**:
- This stage automates deployment of trained model to production (dev, stage, etc) as a web service, batch processing job or to an edge device.

### Operations and Monitoring

After a model is deployed, MLOps focuses on ensuring its performance and stability over time.

- **Model Monitoring**: a critical part of MLOps. It involves monitoring the models performance on live data, looking for signs of *Model Drift* or *Data Drift*:
    - **Model drift**: AKA Concept drift. The relationship between input features and target variable has changed.
    - **Data drift**: The statistical properties of the incoming data have changed, causing teh model to perform poorly.

- **Retraining and Redeployment**: When monitoring indicates a decline in performance, the model needs to be retrained on new data. This triggers the entire MLOps pipeline from experimentation to deployment.


## 1.2 Team & Roles

MLOps is inherently cross-disciplinary, it bridges the gap between different teams and stakeholders.

**Key Roles**:
- **Data Scientists**:
    - Focus on research, data exploration, feature engineering, and model development.
    - Define business problems as ML tasks and evaluate models with appropriate metrics.
- **ML Engineers**:
    - Productionize models: Optimize, refactor, and package for scalable deployment.
    - Build pipelines for training, evaluation, and serving.
    - Bridge handoff between research (data scientists) and operations (DevOps).
- **DevOps / MLOps Engineers**:
    - Ensure reliable infrastructure for training and serving (Cloud, K8, CI/CD).
    - Manage deployment automation, versioning, monitoring and scalability.
    - Apply DevOps principles of automation, reproducability, observability to ML systems.
- **Data Engineers**:
    - Build and Maintain data pipelines to provide clean, reliable, and timely data.
    - Handle data ingestion, transformation and storage at scale (ETL/Data lake)
- **Product Managers / Business Stakeholders**:
    - Align model development with business objectives and KPIs.
    - Prioritize experiements and balance trade-offs between accuracy and efficiency.

| **ML Lifecycle Phase**                   | **Data Scientist** | **ML Engineer**                 | **DevOps / MLOps Engineer**     | **Data Engineer**               | **Product Manager / Business**  |
| ---------------------------------------- | ------------------ | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| **Data Collection & Ingestion**          | C (consulted)      | I (informed)                    | I (informed)                    | R (responsible) A (accountable) | C (consulted)                   |
| **Data Exploration & Feature Eng.**      | R (responsible)    | C (consulted)                   | I (informed)                    | R (responsible)                 | C (consulted)                   |
| **Model Experimentation & Training**     | R (responsible)    | A (accountable)                 | I (informed)                    | C (consulted)                   | C (consulted)                   |
| **Model Evaluation & Validation**        | R (responsible)    | A (accountable)                 | I (informed)                    | C (consulted)                   | C (consulted)                   |
| **Model Packaging & Optimization**       | C (consulted)      | R (responsible) A (accountable) | C (consulted)                   | I (informed)                    | I (informed)                    |
| **Deployment (CI/CD, Serving)**          | I (informed)       | R (responsible)                 | A (accountable) R (responsible) | I (informed)                    | I (informed)                    |
| **Monitoring & Logging**                 | I (informed)       | R (responsible)                 | A (accountable) R (responsible) | I (informed)                    | C (consulted)                   |
| **Model Retraining & Versioning**        | R (responsible)    | A (accountable)                 | C (consulted)                   | R (responsible)                 | I (informed)                    |
| **Business Alignment & Success Metrics** | C (consulted)      | C (consulted)                   | I (informed)                    | I (informed)                    | A (accountable) R (responsible) |


## 1.3 Maturity Assessment

### Level 0: Manual Process (No MLOps)

This is the starting point for most orgs. ML models are developed and deployed in an ad-hoc, manual fashion.

- **Process**: Data scientists train models on their local machines or notebooks. Model artifacts are manually passed to a separate team for deployment, which is often a one-off event.

- **Pain Points**: Lack of reproducibility, no version control for data or models, no monitoring, and significant delays in deploying and updating models. This approach is brittle and not scalable.


### Level 1: ML Pipeline Automation

At this level, some parts of the ML pipeline are automated. The focus is on automating the model training and deployment process with a *repeatable pipeline*.

- **Process**: A *CI/CD pipeline* is introduced, automating the training and deployment of the model. When a new model is trained, the pipeline automatically packages and deploys it.

- **Key Enablers**: Use of a centralized ML platform (e.g., `Kubeflow`, `MLflow`) and version control for model code and data. This level reduces manual errors and improves deployment speed.

### Level 2: CI/CD

This level extends the existing CI/CD to the entire ML Lifecycle. It's about creating a fully integrated, automated system that responds to changes in code, data or environment.

- **Process**: The pipeline automatically triggers on code commits or data changes. Automated tests are run on the data, code, and model before deployment. *A/B testing* and *canary deployments* are often implemented to safely roll out new models.

- **Key Enablers**: Robust data and model validation tests, automated monitoring for data and concept drift, and integrated CI/CD tools. This level ensures a high degree of reliability and reproducibility.

### Level 3: Automated and Monitored Pipelines

This level focuses on the operational aspects of MLOps. The pipeline is not only automated, but now continuously monitored for performance and triggered for retraining. 


- **Process**: The system automatically monitors model performance in production. When a decline in performance is detected (e.g., due to data or concept drift), the system automatically triggers a new training run and, if successful, a redeployment of the new model. This creates a fully autonomous feedback loop.

- **Key Enablers**: Advanced monitoring dashboards, alerting systems, and automated retraining triggers. This level reduces the need for constant human intervention and ensures models stay relevant over time.


### Level 4: Self-Healing and Governance

This is the *pinnacle* of MLOps maturity. The system is not only automated but also *self-correcting* and fully integrated with robust governance and security practices.

- **Process**: The platform includes self-healing capabilities, automatically recovering from pipeline failures or model serving issues. There are integrated governance features for model lineage, auditing, and compliance.

- **Key Enablers**: Advanced platform orchestration, automated security scans, and a comprehensive audit trail for every model and data version. This level is characteristic of highly regulated industries or large-scale, mission-critical ML applications.

- **Relevant tools**:
    - **Workflow Orchestration and Automation**: Kubeflow pipelines, Airflow, Dagster, Argo.
    - **Model Monitoring and Detection Drift**: WhyLabs, Prometheus/Grafana, Fiddler AI.
    - **Experiment Tracking and Registry**: MLFlow, W&B,, Vertex AI Model Registry.
    - **CI/CD**: Jenkins, Github Actions, Tekton, DVC (for versioning datasets).
    - **Serving and Deployment**: KFServing, Seldon Core, BentoML, Azure ML Endpoints.
    - **Alerting and Incidents**: Grafana Alerting, Opsgenie, CloudWatch.

## 2.1 Compute & Storage

Idenfifying the proper compute and storage solutions is fundamental to an MLOps platform's performance and cost effectiveness. The requirements for ML workloads differ significantly from traditional approaches, especially relating to data volume and computational intensity. The main purpose is to provide scalable, reliable resources for training, serving and storing ML workloads and data.

**Compute**:
- **CPUs**: The standard for data preprocessing, ETL (Extract, Transform, Load) and serving simple models. Versatile and cost-effective for general-purpose tasks.
- **GPUs**: Essential for *deep learning* and other types of parallilizable workloads. GPUs accelerate *matrix multiplication* and other operations for training large models, thus drastically reducing training time. Cloud providers offer GPU optimized instances (think `NVIDIA`).
- **TPUs**: Tensor Processing Units are Google's custom-built ASICs (Application-Specific Integrated Circuits) designed specifically for machine learning, especially for training large-scale models. High performant, but less flexible than GPUs.

- **Best Practices**:
    - Use autoscaling clusters for unpredictable workloads.
    - Leverage Spot instances for non-critical training jobs!

**Storage**:
- **Object Storage**: Cloud services (S3, GCS, Blob) are the standard for storing raw and processed data. They are highly scalable, durable and cost-effective. Ideal for massive datasets used in ML.
- **Network File Systems (NFS)**: Used for shared storage that can be mounted on multiple instances, often for faster access to data during training, or for sharing model artifacts within a cluster.
- **Data WareHouses / Lakes**: Essential for managing large, structured, and unstructured datasets. A data lake stores raw data in its native format, while a data warehouse stores structured, processed data for analytics/reporting. A **feature store** is a specialized data store that serves pre-computed features to both training and serving pipelines, ensuring consistency.

- **Best Practices**:
    - Always separate Raw, Curated and Production-ready datasets (Medallion architecture)
    - Raw Data: Object storage like S3, GCS, Azure Blob, Data Lakes.
    - Feature Storage: Feature stores like Feast, Tecton, Vertex AI Store.
    - Model Storage: Model registries like MLflow, Vertex AI, SafeMaker, Azure ML.
    - Artifact Storage: Versioned storage using DVC, Git, MLFlow artifacts.
    - Logging/Monitoring: Time-series DBs (Prometheus, InfluxDB, ElasticSearch)

**On-Prem Considerations**:
    - Hardware procurement times
    - GPU Clusters for training workloads.
    - Network bandwidth for data transfers.
    - Compliance Requirements (GDPR, HIPAA, SOC2)

## 2.2 Containers & Orchestration

*Containers* and *orchestration* tools are the backbone of modern, scalable MLOps infrastructure. They provide a reproducible and portable environment for running ML workflows.

**Containers**: A container packages an application and all its dependencies (code, runtime, libraries) into a single, isolated unit (e.g Docker). Using containers for ML pipelines solves the *"it works on my machine"* problem by ensuring that the exact same environment used for development is used for production. This is critical for reproducibility, as ML models are sensitive to library versions and environment configurations.

- **Best Practices for Containers**:
    - Use lightweight base images (e.g. `python:slim`)
    - Pin dependency versions.
    - Include custom CA certs, GPU drivers and runtime libs wherever necessary.
    - **ALWAYS** scan your images for vulnerabilities.



**Orchestration**: Orchestration tools manage the lifecycle of containers at scale.
- **Kubernetes (K8s)**: The industry standard for container orchestration. Kubernetes automates the deployment, scaling, and management of containerized applications. For MLOps, it's used to manage compute clusters for model training and to run model-serving microservices.

- **Kubeflow**: An open-source project *built on Kubernetes clusters*, specifically designed to make ML pipelines portable and scalable. It provides a platform for running Jupyter notebooks, training models, and deploying them to production. Installed using Kustomize, Helm, or Manifests from Github.

- **Best Practices for Orchestration**:
    - If using the same cluster, isolate environments with namespaces.
    - ALWAYS define resource requests and limits for pods.
    - Apply autoscaling preferences: HPA/VPA for inference, and the cluster autoscaler for training.
    - Treat the infrastructure as code (Terraform, OpenTofu, Helm, Kustomize)

## 2.3 Security

*Security* in MLOps is a multi-layered concern that extends beyond traditional application security to include data, models, and the ML platform itself. All aspects of security must be considered, including using service accounts for automation, Vnet isolation, encryption and more.

**Data Security**: Protecting sensitive data. Mask or anonymize sensitive features, monitor for *data exfiltration*.
- **Encryption**: Data should be encrypted both in transit (TLS) and at rest (e.g. encrypted storage buckets). Keys can be customer managed or cloud managed. Use database encryption (TDE, field-level encryption). Use T-SQL to control Row and Column level security.
- **Access Control**: Use Role-Based Access Control (**RBAC**) to restrict who can access, modify or delete a dataset or model. This prevents unauthorized access to sensitive information.


**Model Security**: The models themselves must be secured. Control access and protect the endpoints with API rate limiting, model encryption and obfuscation. 
- **Model Tampering**: Ensure the integrity of models from training to deployment to prevent malicious actors from injecting backdoors. Use digital signatures or hash checks to verify model artifacts. Use model *watermarking* and *logging* to detect misuse.
- **Adversarial Attacks**: Subtle changes to input data can cause the model to make incorrect predictions. MLOps security involves monitoring and mitigating these attacks (e.g. evasion, poisoning).

**Infrastructure Security**: The MLOps pipeline itself, including the surrounding infrastructure, must be secured.
- **Network Security**: Secure VPC/Virtual Networks using a Firewall. Consider *private endpoints* for sensitive services to prevent exposure to the internet. Utilize network policies in Kubernetes.
- **Secrets Management**: Use Hashicorp Vault or cloud Key Vault service to store secrets including API keys, credentials and other sensitive information. Github and other Enterprise git services can hold secrets securely for each repository. NEVER hardcode a secret. Rotate credentials regularly.
- **Image Scanning**: Automatically scan a container image for vulnerabilities (Aqua, cloud-native scanners)
- **Least Privelege**: Apply this principle to all services and users at every step of the process, to ensure everyone has only the permissions they need for the task.
- **Audit and Compliance**: Centralized logging either in ELK, Splunk to observe the entire environment. Automate compliance checks with Cloud tools or others like OPA. Consider data retention policies and right to deletion compliance (GDPR). Keep up with regular penetration testing and up-to-date incident response procedures.


## 3.1 Version Control Strategy

In MLOps, *version control* goes beyond the code to include the data and the models, which are often large binary files. This is essential for collaboration, reproducability, auditing, and rolling back to previous versions if issues arise.

**Sample Repository Structure**:

```txt
ml-project/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model training code
│   └── serving/           # Model serving code
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── dockerfiles/           # Container definitions
├── .dvcignore             # DVC ignore patterns
├── dvc.yaml               # DVC pipeline definition
└── requirements.txt       # Dependencies
```

**Code Versioning**:
- Use Git as the foundation for all ML code (Notebooks, scripts, configs, pipeline definitions).
- Adopt *branching strategies* (**GitFlow**, trunk-based) aligned with your team's CI/CD practices.
- Store pipeline definitions alongside application code (EVERYTHING should be code!).

**Data Versioning**:
- Datasets change over time, and a model trained on one version of data will produce different results from one trained on another. We need to track these changes. Two popular options are:
1. **Git LFS (Large File Storage)**: An extension for Git that handles klarge files by storing pointers in the repository.
1. **Data Version Control (DVC)**: **DVC** is specifically built for MLOps. It works alongside Git to version large datasets and ML models. It creates a `.dvc` file that points to the actual data in remote locations. In addition, DVC allows for data stages integrated with python scripts to manipulate data.
- Establish policies for *Raw data* (immutable), *Curated Data* (versioned and auditable), *Production-ready* (stable contracts/schema). 

- Use `.dvcignore` to exclude temporary files.

**Model Versioning**:
- The *trained model* is the most important artifact of an ML pipeline. Model *binries* are stored as pickle, joblib, or ONNX types for example. Each time a model is trained, it should be treated as a new version. Version metadata includes: Code commit hash, data version, training environment, and hyperparameters and metrics.
- **MD5/SSHA Hashes**: basic way to track models by using the hash of the file.
- **Model Registry**: Think of a model registry like a docker container registry. More info in 3.3. For exmample, DVC Studio model registy is enabled on top of Git.

- Model Version naming convention: 
`model-name-v{major}.{minor}.{patch}-{experiment-id}-fraud-detection-v2.1.0-exp-4a3f2b1`


## 3.2 Data Pipelines

Data pipelines automate data ingestion, transformation, validation, and delivery to training and inference systems. A reliable and automated data pipeline ensures the model always has access to fresh, clean, and validated data. Consider *batch* (`Spark`, `Flink`, `Beam`) vs *stream* (`Kafka`, `Spark Streaming`, `Flink`) processing patterns.

**General Pipeline Stages**:
1. **Ingestion**: This is the process of collecting and loading raw data from various sources (databases, APIs, streaming) into a data lake. Be minful of data types, structured vs unstructured data, etc.
1. **Validation**: Before data is used for training, it must be validated. Checking for schema changes, missing values, outliers. Any unexpected changes in data can break the pipeline. Vaslidation tools like Great Expectations, Deequ, and TFX Data Validation can be useful.
1. **Transformation and Feature Engineering**: This stage involves cleaning the data, handling missing values, and transforming raw data into features suitable for model training. This includes aggregations, encoding, scaling (StandardScaler, MinMaxScaler, RobustScaler) and temporal features.
1. **Orchestration**: DVC pipelines are lightweight and integrate easily into existing workflows. Tools like Apache Airflow, Prefect or Dagster can be used to orchestrate Data pipelines as well. They define a sequence of tasks (DAGS - Directed Acyclic Graphs) that run automatically with clear scheduling, monitoring and error handling capabilities.

- Use *DVC Pipelines* for lightweight reproducable ML workflows, or use *Airflow* or *Kubeflow* when you need more robust pipelines.

- Pipeline monitoring should be in place to check for data freshness, and monitor pipeline health (CPU, Memory, I/O metrics).

## 3.3 Model Registry

The *Model Registry* is a centralized, version-controlled repo for managing the lifecycle of trained ML models. It is a critical component for promoting collaboration, reproducibility, and governance.

- **Centralized Storage**: A Model Registry provides a single source of truth for all models in an organization. This prevents model proliferation and confusion, where different teams might be using different versions of the same model.

- **Version and Stage Management**: It tracks the version history of each model, from experimentation to production. Models can be assigned different stages (e.g., `Staging`, `Production`, `Archived`), providing a clear state for each model and controlling its readiness for deployment. This is crucial for managing the model release process.

- **Metadata and Provenance**: A robust registry stores crucial metadata about each model version, including the training data used, the hyperparameters, the code commit hash, and the performance metrics. This *provenance data* is invaluable for auditing, debugging, and reproducing results.

**Deployment Integration**: The Model Registry integrates directly with the CI/CD pipeline. When a new model version is approved for production, the pipeline can automatically retrieve it from the registry and deploy it to the serving environment. Define a **model promotion workflow** that can validate models and promote from the registry, including *A/B testing* integration patterns.