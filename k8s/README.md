# Deploying Quiver to Kubernetes

This directory contains Kubernetes manifests and Helm charts for deploying Quiver to a Kubernetes cluster.

## Prerequisites

- Kubernetes cluster (v1.19+)
- kubectl configured to communicate with your cluster
- Helm v3 (for Helm chart deployment)
- Docker (for building the container image)

## Quick Start with kubectl

1. **Build and push the Docker image**

   ```bash
   # From the root of the repository
   docker build -t your-registry/quiver:latest .
   docker push your-registry/quiver:latest
   ```

2. **Update the image reference**

   Edit `deployment.yaml` and replace `${REGISTRY}/quiver:${TAG}` with your actual image reference.

3. **Create a namespace**

   ```bash
   kubectl create namespace quiver
   ```

4. **Apply the Kubernetes manifests**

   ```bash
   kubectl apply -f k8s/pvc.yaml -n quiver
   kubectl apply -f k8s/configmap.yaml -n quiver
   kubectl apply -f k8s/deployment.yaml -n quiver
   kubectl apply -f k8s/service.yaml -n quiver
   kubectl apply -f k8s/ingress.yaml -n quiver
   kubectl apply -f k8s/hpa.yaml -n quiver
   ```

5. **Verify the deployment**

   ```bash
   kubectl get pods -n quiver
   kubectl get svc -n quiver
   ```

## Deploying with Helm

1. **Build and push the Docker image**

   ```bash
   # From the root of the repository
   docker build -t your-registry/quiver:latest .
   docker push your-registry/quiver:latest
   ```

2. **Install the Helm chart**

   ```bash
   # From the root of the repository
   helm install quiver ./helm/quiver \
     --namespace quiver \
     --create-namespace \
     --set image.repository=your-registry/quiver \
     --set image.tag=latest
   ```

3. **Verify the deployment**

   ```bash
   kubectl get pods -n quiver
   kubectl get svc -n quiver
   ```

## Configuration

### Environment Variables

The Quiver container accepts the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QUIVER_PORT` | HTTP server port | `8080` |
| `QUIVER_DIMENSION` | Vector dimension | `1536` |
| `QUIVER_MAX_ELEMENTS` | Maximum number of vectors | `1000000` |
| `QUIVER_HNSW_M` | HNSW hyperparameter M | `16` |
| `QUIVER_HNSW_EF_CONSTRUCT` | HNSW hyperparameter efConstruction | `200` |
| `QUIVER_HNSW_EF_SEARCH` | HNSW hyperparameter efSearch | `100` |
| `QUIVER_BATCH_SIZE` | Number of vectors to batch before insertion | `100` |
| `QUIVER_PERSIST_INTERVAL` | How often to persist index to disk | `5m` |
| `QUIVER_BACKUP_INTERVAL` | How often to create backups | `1h` |
| `QUIVER_MAX_BACKUPS` | Maximum number of backups to keep | `5` |
| `QUIVER_BACKUP_COMPRESSION` | Whether to compress backups | `true` |
| `QUIVER_LOG_LEVEL` | Logging level (debug, info, warn, error) | `info` |

### Helm Values

When deploying with Helm, you can customize the deployment by overriding values in `values.yaml`. For example:

```bash
helm install quiver ./helm/quiver \
  --namespace quiver \
  --create-namespace \
  --set image.repository=your-registry/quiver \
  --set image.tag=latest \
  --set resources.requests.memory=1Gi \
  --set config.dimension=768
```

## Monitoring

Quiver exposes Prometheus metrics at the `/metrics` endpoint. You can configure Prometheus to scrape these metrics by using the ServiceMonitor resource (if you have the Prometheus Operator installed):

```bash
helm install quiver ./helm/quiver \
  --namespace quiver \
  --create-namespace \
  --set metrics.serviceMonitor.enabled=true
```

## Scaling

Quiver can be scaled horizontally for read-heavy workloads. The Horizontal Pod Autoscaler (HPA) is configured to scale based on CPU and memory usage:

```bash
helm install quiver ./helm/quiver \
  --namespace quiver \
  --create-namespace \
  --set autoscaling.enabled=true \
  --set autoscaling.minReplicas=2 \
  --set autoscaling.maxReplicas=10
```

## Persistence

By default, Quiver uses a PersistentVolumeClaim to store data. You can customize the storage class and size:

```bash
helm install quiver ./helm/quiver \
  --namespace quiver \
  --create-namespace \
  --set persistence.storageClass=standard \
  --set persistence.size=20Gi
```

## Troubleshooting

### Checking Logs

```bash
kubectl logs -f deployment/quiver -n quiver
```

### Checking Pod Status

```bash
kubectl describe pod -l app=quiver -n quiver
```

### Checking PVC Status

```bash
kubectl get pvc -n quiver
kubectl describe pvc quiver-pvc -n quiver
```
