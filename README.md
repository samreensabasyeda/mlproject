# Named Entity Recognition - System Design Architecture

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│  (Web App / Mobile App / API Clients / Third-party Services)   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY                                │
│  • Authentication & Authorization (JWT/OAuth)                   │
│  • Rate Limiting & Throttling                                   │
│  • Request Routing                                              │
│  • SSL/TLS Termination                                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOAD BALANCER                                │
│  • Distributes traffic across multiple instances               │
│  • Health checks                                                │
│  • Auto-scaling triggers                                        │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────────┬──────────────┬──────────────┐
│   ML Service │   ML Service │   ML Service │
│   Instance 1 │   Instance 2 │   Instance N │
└──────┬───────┴──────┬───────┴──────┬───────┘
       │              │              │
       └──────────────┼──────────────┘
                      ▼
         ┌────────────────────────┐
         │   MODEL REPOSITORY     │
         │  • Model Versioning    │
         │  • A/B Testing         │
         │  • Rollback Capability │
         └────────────────────────┘
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
┌──────────────┬──────────────┬──────────────┐
│  PostgreSQL  │    Redis     │   S3/Blob    │
│  (Metadata)  │   (Cache)    │   (Storage)  │
└──────────────┴──────────────┴──────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  MONITORING & LOGGING  │
         │  • Prometheus          │
         │  • Grafana             │
         │  • ELK Stack           │
         │  • MLflow              │
         └────────────────────────┘
```

---

## 2. Component Details

### 2.1 API Gateway (Kong / AWS API Gateway / NGINX)

**Responsibilities:**
- Authentication and authorization
- Request validation
- Rate limiting (e.g., 1000 requests/hour per user)
- API versioning (v1, v2)
- Request/response transformation
- CORS handling

**Endpoints:**
```
POST /api/v1/ner/predict
  Body: {"text": "sentence to analyze"}
  Response: {"entities": [...], "confidence": [...]}

POST /api/v1/ner/batch
  Body: {"texts": ["sentence 1", "sentence 2", ...]}
  
GET /api/v1/ner/health
  Response: {"status": "healthy", "version": "2.0"}

GET /api/v1/ner/metrics
  Response: {"latency_p95": 120, "throughput": 500}
```

### 2.2 ML Service Layer (FastAPI / Flask)

**Service Structure:**
```python
app/
├── main.py              # FastAPI application
├── models/
│   ├── model_loader.py  # Load and cache models
│   └── predictor.py     # Prediction logic
├── preprocessing/
│   ├── tokenizer.py
│   └── encoder.py
├── postprocessing/
│   └── formatter.py
├── monitoring/
│   └── metrics.py
└── config/
    └── settings.py
```

**Key Features:**
- Asynchronous request handling
- Connection pooling
- Model caching in memory
- Batch processing support
- Graceful degradation

**Docker Container:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.3 Model Repository (MLflow / DVC)

**Features:**
- Version control for models
- Model metadata storage
- Experiment tracking
- Model lineage
- A/B testing support

**Model Versioning Strategy:**
```
models/
├── ner-bilstm/
│   ├── v1.0.0/
│   │   ├── model.h5
│   │   ├── metadata.json
│   │   └── performance.json
│   ├── v1.1.0/
│   └── v2.0.0/
```

### 2.4 Data Storage

**PostgreSQL:**
- User information
- Request logs
- Model metadata
- Performance metrics

**Redis:**
- Caching frequent predictions
- Session management
- Rate limiting counters
- Real-time metrics

**S3/Blob Storage:**
- Model artifacts
- Training data
- Logs archive
- Backup data

---

## 3. Canary Deployment Strategy

### 3.1 Deployment Process

```
┌─────────────────────────────────────────────────┐
│  STEP 1: Deploy New Model (v2.0) to 5% Traffic │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
      ┌────────────────────────────┐
      │  95% → Model v1.0          │
      │   5% → Model v2.0 (Canary) │
      └────────────┬───────────────┘
                   │
                   ▼
      ┌────────────────────────────┐
      │  Monitor for 24-48 hours:  │
      │  • Error rates             │
      │  • Latency (p50, p95, p99) │
      │  • Model accuracy          │
      │  • User feedback           │
      └────────────┬───────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    Success?             Failure?
         │                   │
         ▼                   ▼
┌────────────────┐    ┌──────────────┐
│ Increase to    │    │  ROLLBACK    │
│ 25% → 50% →    │    │  to v1.0     │
│ 75% → 100%     │    │              │
└────────────────┘    └──────────────┘
```

### 3.2 Canary Configuration (Kubernetes)

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ner-service
spec:
  selector:
    app: ner-predictor
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ner-stable
spec:
  replicas: 9  # 95% of traffic
  selector:
    matchLabels:
      app: ner-predictor
      version: v1.0
  template:
    metadata:
      labels:
        app: ner-predictor
        version: v1.0
    spec:
      containers:
      - name: ner
        image: ner-service:v1.0
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ner-canary
spec:
  replicas: 1  # 5% of traffic
  selector:
    matchLabels:
      app: ner-predictor
      version: v2.0
  template:
    metadata:
      labels:
        app: ner-predictor
        version: v2.0
    spec:
      containers:
      - name: ner
        image: ner-service:v2.0
```

### 3.3 Success Criteria for Canary

| Metric | Threshold | Action if Exceeded |
|--------|-----------|-------------------|
| Error Rate | < 1% | Continue rollout |
| P95 Latency | < 200ms | Investigate |
| P99 Latency | < 500ms | Hold rollout |
| F1-Score Drop | < 2% | Investigate |
| Memory Usage | < 80% | Optimize |

---

## 4. ML Model Monitoring Strategy

### 4.1 Monitoring Layers

```
┌─────────────────────────────────────────────────┐
│        LAYER 1: INFRASTRUCTURE MONITORING       │
│  • CPU, Memory, Disk usage                      │
│  • Network I/O                                  │
│  • Container health                             │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│        LAYER 2: APPLICATION MONITORING          │
│  • Request rate, latency (p50, p95, p99)        │
│  • Error rates (4xx, 5xx)                       │
│  • Throughput (requests/sec)                    │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│        LAYER 3: MODEL PERFORMANCE               │
│  • Prediction accuracy                          │
│  • F1-score, precision, recall                  │
│  • Confusion matrix                             │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│        LAYER 4: DATA QUALITY                    │
│  • Input data distribution                      │
│  • Missing values, outliers                     │
│  • Data drift detection                         │
│  • Prediction confidence distribution           │
└─────────────────────────────────────────────────┘
```

### 4.2 Key Metrics to Track

**Model Metrics:**
- Entity-level F1-score (daily)
- Per-tag performance (B-PER, I-PER, B-geo, etc.)
- Prediction confidence distribution
- Low confidence predictions (< 0.5)

**Data Drift Metrics:**
- Input text length distribution
- Vocabulary overlap with training data
- Unknown word ratio
- Entity type distribution

**System Metrics:**
- Prediction latency (p50, p95, p99)
- Throughput (predictions/second)
- Error rates
- Model loading time

### 4.3 Alert Configuration

```yaml
alerts:
  - name: HighErrorRate
    condition: error_rate > 5%
    duration: 5m
    severity: critical
    action: page_on_call
    
  - name: HighLatency
    condition: p95_latency > 300ms
    duration: 10m
    severity: warning
    action: notify_team
    
  - name: ModelPerformanceDrop
    condition: f1_score < 0.85
    duration: 1h
    severity: critical
    action: page_on_call
    
  - name: DataDrift
    condition: vocabulary_overlap < 70%
    duration: 1d
    severity: warning
    action: notify_data_team
```

### 4.4 Dashboard Components

**Real-time Dashboard (Grafana):**
1. Request rate and latency graph
2. Error rate over time
3. Model version distribution
4. Cache hit rate
5. Resource utilization

**ML Performance Dashboard:**
1. F1-score trends
2. Confusion matrix heatmap
3. Per-entity type performance
4. Prediction confidence histogram
5. Daily/weekly performance comparison

---

## 5. Load and Stress Testing

### 5.1 Testing Strategy

**Load Testing:** Simulate expected production load
- Normal: 100 requests/second
- Peak: 500 requests/second
- Duration: 1 hour

**Stress Testing:** Find system breaking point
- Start: 100 requests/second
- Increment: +100 every 5 minutes
- Continue until: error rate > 10% or latency > 1s

**Spike Testing:** Sudden traffic surges
- Baseline: 100 requests/second
- Spike: 1000 requests/second for 2 minutes
- Return to baseline

### 5.2 Testing Tool (Locust)

```python
from locust import HttpUser, task, between

class NERUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def predict_single(self):
        self.client.post("/api/v1/ner/predict", json={
            "text": "John lives in New York"
        })
    
    @task(1)
    def predict_batch(self):
        self.client.post("/api/v1/ner/batch", json={
            "texts": [
                "Apple Inc is in California",
                "Barack Obama was president",
                "Google acquired YouTube"
            ]
        })
    
    @task(1)
    def health_check(self):
        self.client.get("/api/v1/ner/health")
```

**Run Command:**
```bash
locust -f load_test.py --host=http://localhost:8000 \
       --users=500 --spawn-rate=10 --run-time=1h
```

### 5.3 Performance Benchmarks

| Metric | Target | Acceptable | Poor |
|--------|--------|-----------|------|
| P50 Latency | < 50ms | < 100ms | > 100ms |
| P95 Latency | < 150ms | < 300ms | > 300ms |
| P99 Latency | < 300ms | < 500ms | > 500ms |
| Throughput | > 500 rps | > 200 rps | < 200 rps |
| Error Rate | < 0.1% | < 1% | > 1% |

---

## 6. ML Training Tracking & Audit

### 6.1 Training Pipeline

```
┌───────────────┐
│  Data Source  │
│  (S3/Database)│
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Data Pipeline │ ← DVC (Data Version Control)
│ • Validation  │
│ • Transform   │
└───────┬───────┘
        │
        ▼
┌───────────────────┐
│  Training Job     │
│  • Hyperparameters│ ← Logged to MLflow
│  • Model training │
│  • Evaluation     │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Model Registry   │
│  • Version: 2.0   │
│  • Metrics: {...} │ ← MLflow Model Registry
│  • Artifacts      │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  CI/CD Pipeline   │
│  • Tests          │
│  • Deploy         │
└───────────────────┘
```

### 6.2 Experiment Tracking with MLflow

```python
import mlflow
import mlflow.keras

# Start MLflow run
with mlflow.start_run(run_name="ner-bilstm-v2"):
    
    # Log parameters
    mlflow.log_param("embedding_dim", 100)
    mlflow.log_param("lstm_units", 100)
    mlflow.log_param("dropout", 0.1)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("max_epochs", 15)
    
    # Train model
    history = model.fit(...)
    
    # Log metrics
    mlflow.log_metric("train_loss", history.history['loss'][-1])
    mlflow.log_metric("val_loss", history.history['val_loss'][-1])
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    
    # Log model
    mlflow.keras.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("training_history.png")
    
    # Log dataset info
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("val_samples", len(X_val))
    mlflow.log_param("test_samples", len(X_test))
```

### 6.3 Audit Trail

**Track Everything:**
1. Who triggered the training
2. When it was triggered
3. What data was used (version)
4. What code was used (git commit)
5. What hyperparameters were used
6. What results were achieved
7. Where the model was deployed

**Audit Database Schema:**
```sql
CREATE TABLE model_audit (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50),
    trained_by VARCHAR(100),
    training_started_at TIMESTAMP,
    training_completed_at TIMESTAMP,
    data_version VARCHAR(50),
    git_commit VARCHAR(40),
    hyperparameters JSONB,
    metrics JSONB,
    deployed_at TIMESTAMP,
    deployed_by VARCHAR(100),
    status VARCHAR(20)
);
```

---

## 7. CI/CD Pipeline

### 7.1 Continuous Integration

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ --cov=app --cov-report=xml
      
      - name: Lint code
        run: |
          pip install flake8
          flake8 app/ --max-line-length=100
      
      - name: Type checking
        run: |
          pip install mypy
          mypy app/
```

### 7.2 Continuous Deployment

```yaml
# .github/workflows/cd.yml
name: CD Pipeline

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: |
          docker build -t ner-service:${{ github.sha }} .
      
      - name: Run integration tests
        run: |
          docker-compose up -d
          pytest tests/integration/
          docker-compose down
      
      - name: Push to registry
        run: |
          docker push ner-service:${{ github.sha }}
      
      - name: Deploy to staging
        run: |
          kubectl set image deployment/ner-service \
            ner=ner-service:${{ github.sha }} \
            --namespace=staging
      
      - name: Run smoke tests
        run: |
          pytest tests/smoke/
      
      - name: Deploy to production (canary)
        if: success()
        run: |
          kubectl apply -f k8s/canary-deployment.yaml
```

---

## 8. Security Considerations

### 8.1 Security Measures

**Authentication & Authorization:**
- JWT tokens with expiration
- Role-based access control (RBAC)
- API key management

**Data Security:**
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- PII data masking in logs

**Network Security:**
- VPC isolation
- Security groups / firewall rules
- DDoS protection

**Model Security:**
- Model watermarking
- Input validation
- Output sanitization
- Rate limiting to prevent abuse

---

## 9. Cost Optimization

### 9.1 Strategies

**Compute:**
- Auto-scaling based on load
- Spot instances for training
- Right-sizing containers

**Storage:**
- Lifecycle policies for S3
- Compression for logs
- Archive old model versions

**Caching:**
- Cache frequent predictions in Redis
- CDN for static content
- Model caching in memory

**Monitoring:**
- Optimize log retention
- Sample metrics for high-volume data
- Use log aggregation

---

## 10. Disaster Recovery

### 10.1 Backup Strategy

**Daily Backups:**
- Database (automated snapshots)
- Model artifacts
- Configuration files

**Weekly Backups:**
- Full system backup
- Training data archives

**Retention:**
- Daily: 7 days
- Weekly: 4 weeks
- Monthly: 12 months

### 10.2 Recovery Plan

**RTO (Recovery Time Objective):** < 1 hour
**RPO (Recovery Point Objective):** < 24 hours

**Steps:**
1. Identify failure point
2. Switch to backup region
3. Restore from latest backup
4. Verify system functionality
5. Update DNS if needed

---

## Summary

This system design provides:
- **Scalability:** Auto-scaling, load balancing
- **Reliability:** Multi-region, redundancy, monitoring
- **Performance:** Caching, optimization, load testing
- **Maintainability:** CI/CD, version control, documentation
- **Security:** Authentication, encryption, auditing
- **Cost-effective:** Resource optimization, caching

The architecture is production-ready and can handle millions of requests while maintaining model quality and system stability.
