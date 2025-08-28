# MLOps: Deployments

**Deployment** is the crucial stage where ML models are integrated into production systems to serve predictions. Not only getting a model into an application, but managing the entire lifecycle and ensuring its reliable, scalable, and maintainable in a real-world environment. There is an emphasis on performance for a robust large scale system.

## ML Deployment Pillars of Success

- **Automation**: 
    - Add **CT (Continuous Training)** as well as Model validation to the general CI/CD pipeline.
    - Use Terraform, OpenTofu, Pulumi, Helm, Argo to help spin up environments.
    - Use blue-green or canary releases for model updates using GitOps.
- **Scalability**:
    - Workloads must handle traffic spikes, data size spikes, varying inference load, and number of concurrent experiments.
    - Use distributed training (Azure ML, Pytorch DDP, Spark MLLib) to scale across GPUs and VM clusters.
- **Monitoring**:
    - Monitor system metrics: CPU, Memory, GPU utilization, request latency, throughput.
    - ML Specific metrics including accuracy drift, data drift, feature distro changes, fairness metrics.
    - Use Prometheus and Grafana or Azure Monitor to set up drift anomaly alerts.
- **Version Control**:
    - Feast or Delta Lake for feature versioning adn point-in-time extraction.
    - Combination of Git and DVC for source code, infrastructure, datasets, features.
    - Use Model Registries (MLFlow, SageMaker) and container registries for docker containers.
- **Reproducability**:
    - Use Docker images and pinned python package versions.
    - Keep training/validation datasets immutable and version controlled.
    - Codify transformations in notebooks-to-pipelines (Papermill, Kedro, Prefect + DVC)
- **Observability**:
    - Track schema evolution, missing values, and outliers (Great Expectations, Soda are tools for this).
    - Track per-feature attribution (SHAP, Integrated Gradients), prediction uncertainty, bias/fairness.
    - Pipeline observability using distributed tracing (Open Telemetry, Jaeger) for ML pipelines across Spark, Kafka.

## Kubernetes in ML

**Kubernetes** as a container orchestration platform is the de facto standard for ML model serving, as it covers the scalability, flexibility and extensibility required by the ML pillars of success. There are various ML serving frameworks (Seldon, KServe, BentoML) built for Kubernetes that simplify model serving features like multi-model deployment, serverless scaling and scaling to zero, A/B testing, pre/post-processing pipelines, etc.

### KServe

**KServe** is the dominant choice today for K8 native ML frameworks, part of Kubeflow and now the CNCF sandbox project. It provides model serving on Kubernetes with autoscaling, GPU/accelerator support, and canary rollouts.
- Supports many backends (TorchServe, TensorFlow Serving, ONNX Runtime, XGBoost, Skleran servers, and custom inference services via docker image)
- Provides **Kubernetes Native CRDs** (`InferenceService`) to define model servers (predictor, transformer, explainer). 
- Integreate with `Iter8` or `Argo` for automated metrics based promotions (A/B metrics based rollouts)

KServe supports two modes:
- **Serverless (Knative)**: Automatic request-based autoscaling (scale to zero, canary, revision-based traffic routing)
- **Raw (no Knative)**: Traditional deployment based, predictable resources for high-throughput and long-running inference.

Supports Autoscaling and scale-to-zero:

```yaml
metadata:
  annotations:
    autoscaling.knative.dev/minScale: "0"
    autoscaling.knative.dev/maxScale: "10"
    autoscaling.knative.dev/target: "1"  # target concurrency
```

If Kserve is installed with Istio or Kourier, there will be a Gateway/Ingress. (often a cluster-local svc).

Making a prediciton Request to a model built on K8 with KServe:

1. cURL
```bash
# Example using the Istio ingress gateway hostname and path:
curl -v -H "Content-Type: application/json" \
  -d '{"instances":[[5.1,3.5,1.4,0.2]]}' \
  http://<INGRESS_HOST>/v1/models/some-model:predict
```
2. Python Client (requests)
```python
import requests, json
# Kserves standard prediciton API format of array of samples
url = "http://<INGRESS_HOST>/v1/models/some-model:predict"
payload = {"instances": [[5.1,3.5,1.4,0.2]]}
r = requests.post(url, json=payload, timeout=10)
print(r.status_code, r.json())
```

**Note**: KServe is configured to expose the inferenceservice via Knative revision URL.

- Additionally, Use Pod Security Contstaints to block anti-patterns.

```bash
kubectl label ns ml-prod pod-security.kubernetes.io/enforce=restricted
kubectl label ns ml-prod pod-security.kubernetes.io/enforce-version=latest
```


#### KServe Quick Recipe

Use example manifests in `./manifests/`:

1. Install **KServe** in cluster (serverless mode requires Knative + optional Istio). (See KServe install docs). KServe
2. Create namespace `ml-prod` with labels for sidecar injection and pod-security.
3. Create minimal RBAC + kserve-model-sa service account.
4. Deploy `InferenceService` manifest.
5. Add `canaryTrafficPercent: 10` to test a new version.
6. Monitor latency and error metrics; if healthy, promote canary.
7. Enforce `PeerAuthentication` and `AuthorizationPolicy` to restrict callers.

## Cloud Provider Ecosystem

- Leverage Managed Kubernetes services to abstract much of the complexity away.

- Leverage serverless compute options like Lambda, Cloud Functions, Azure Functions for cost-effective, infrequent, burstable inferences (integrate with API gateways).