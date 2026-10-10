# Monitoring examples (standalone)

This directory is a **sketch**, not part of the running tutorial stack.

`docker-compose.yml` starts Jupyter and the FastAPI template only. There is no Redis
service, no `monitoring` compose profile, and the FastAPI template does **not** expose
`/metrics`. These files are here if you want to wire Prometheus/Grafana yourself later.

## Files

### prometheus.yml

Example scrape config aimed at a hypothetical app on port 8000 plus Prometheus itself.
The `langchain-rag-app` job's `/metrics` target will stay down until you add an exporter
to the template.

### grafana-dashboard.json

Example dashboard JSON (request rate, latency percentiles, error rate). Import it only
after Prometheus is actually scraping something.

## Manual setup (optional)

These commands are independent of this repo's compose file:

```bash
# Prometheus
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v "$(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml" \
  prom/prometheus

# Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

Then in Grafana (http://localhost:3000, admin/admin): add a Prometheus datasource
(URL `http://host.docker.internal:9090` or the Prometheus container name on a shared
network) and import `grafana-dashboard.json`.

## What is not included

- No Redis cache or `cache_hit_rate`
- No compose `--profile monitoring`
- No application `/metrics` endpoint
- No committed SLOs or alert routing

If you add metrics later, point Prometheus at that exporter and treat these files as a
starting template, not as a working production monitor.
