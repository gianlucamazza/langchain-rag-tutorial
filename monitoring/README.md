# Monitoring Configuration

This directory contains monitoring configuration files for the LangChain RAG Tutorial application.

## Files

### prometheus.yml

Prometheus configuration for scraping metrics from:

- The RAG application (port 8000)
- Redis cache
- Container metrics
- Self-monitoring

### grafana-dashboard.json

Pre-configured Grafana dashboard showing:

- Request rate
- Response time percentiles (P50, P95, P99)
- Success rate
- Error rate by endpoint
- Latency trends

## Setup

### Using Docker Compose

The monitoring stack is already configured in `docker-compose.yml`:

```bash
# Start with monitoring enabled
docker-compose --profile monitoring up -d

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### Manual Setup

#### 1. Start Prometheus

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

#### 2. Start Grafana

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

#### 3. Configure Grafana

1. Login to <http://localhost:3000> (admin/admin)
2. Add Prometheus data source:
   - URL: <http://prometheus:9090>
   - Access: Server (default)
3. Import dashboard:
   - Dashboard → Import
   - Upload `grafana-dashboard.json`

## Metrics Collected

### Application Metrics

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency histogram
- `rag_query_duration_seconds` - RAG query processing time
- `vector_search_duration_seconds` - Vector similarity search time
- `llm_generation_duration_seconds` - LLM generation time
- `cache_hit_rate` - Redis cache hit rate

### Infrastructure Metrics

- Container CPU/memory usage
- Redis operations/sec
- Network I/O
- Disk I/O

## Alerting (Optional)

To enable alerts, uncomment the alerting section in `prometheus.yml` and create alert rules:

```yaml
# alerts/rag_alerts.yml
groups:
  - name: rag_application
    interval: 30s
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"
```

## Customization

### Adding Custom Metrics

To expose custom metrics from your application:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
rag_queries = Counter('rag_queries_total', 'Total RAG queries')
query_duration = Histogram('rag_query_duration_seconds', 'RAG query duration')
active_connections = Gauge('active_connections', 'Active WebSocket connections')

# Use in your code
@query_duration.time()
def process_query(query: str):
    rag_queries.inc()
    # ... your code
```

### Dashboard Customization

Edit `grafana-dashboard.json` or create panels directly in Grafana UI:

1. Common visualizations:
   - Time series for trends
   - Stat panels for current values
   - Gauge for percentages
   - Heatmap for distribution

2. Useful PromQL queries:

   ```promql
   # Request rate by endpoint
   rate(http_requests_total[5m])
   
   # Error percentage
   100 * (rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]))
   
   # Average response time
   rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])
   ```

## Troubleshooting

### Prometheus not scraping metrics

1. Check targets: <http://localhost:9090/targets>
2. Verify application is exposing `/metrics` endpoint
3. Check network connectivity between containers

### Grafana dashboard shows "No data"

1. Verify Prometheus datasource is configured correctly
2. Check time range in dashboard
3. Ensure metrics are being collected (check Prometheus UI)

### High memory usage

1. Reduce retention time in `prometheus.yml`
2. Decrease scrape frequency
3. Use remote storage for long-term metrics

## Best Practices

1. **Retention**: Keep 15-30 days locally, use remote storage for historical data
2. **Scrape Interval**: 10-15s for most metrics, 30s for infrastructure
3. **Cardinality**: Avoid high-cardinality labels (user IDs, timestamps)
4. **Security**: Enable authentication in production
5. **Backup**: Export important dashboards regularly

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/)
