# Deployment

What has to be true before this application faces real users, and how to undo a
bad release.

## Before you deploy: the four things that will hurt you

**1. `AUTH_ENABLED=false` means every visitor shares one identity.** Quotas,
history, and workspaces all key off the user ID. With auth disabled that ID is
the literal string `local` for everyone — so one visitor can read another's
conversations and exhaust everyone's budget. Set `AUTH_ENABLED=true` before the
application is reachable from the internet, and check `/readyz` reports it.

**2. Rate limiting is per-process.** `services/quota_service.py` keeps its
sliding window in memory. With `--workers 2` a client gets twice the configured
limit; behind three replicas, six times. For a single small instance this is
fine. Beyond that, move the window to Redis — the interface to change is
`check_rate_limit`, and nothing else needs to know.

**3. The cost ceiling is enforced *before* a run, not during one.** A user under
their limit can start a run that takes them well past it. `REQUEST_TOKEN_BUDGET`
bounds how far past — the true worst case is `DAILY_COST_LIMIT_USD` plus one
full request budget. Size both together.

**4. Docker socket access is host root.** `docker-compose.yml` mounts
`/var/run/docker.sock` so the app can launch sandbox containers. Anyone who can
execute code in the app container can then start a privileged container and own
the host. That is acceptable locally and **not** acceptable in production. In a
real deployment, run checks on separate sandbox workers (a queue the app writes
to and isolated workers read from), or use a runtime built for this — gVisor,
Kata, Fly Machines, or a per-run Firecracker VM.

## Required configuration

| Variable | Production value | Why |
| --- | --- | --- |
| `OPENAI_API_KEY` | from the secret manager | never in an image or a repo |
| `FLASK_SECRET_KEY` | 32+ random bytes | session signing |
| `AUTH_ENABLED` | `true` | see above |
| `CONVERSATION_STORE` | `sqlite` (or your SQL store) | `memory` loses everything on restart |
| `DATABASE_URL` | a path on a **persistent volume** | see below |
| `WORKSPACE_ROOT` | a path on a persistent volume | clones survive a restart |
| `SANDBOX_MODE` | `docker` | `auto` silently degrades isolation |
| `LOG_LEVEL` | `INFO` | `DEBUG` logs request bodies |
| `DAILY_COST_LIMIT_USD` | deliberate number | the only ceiling on spend |

Generate a secret key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Storage

SQLite on a persistent volume is genuinely durable and appropriate for a single
instance. It does **not** work across replicas — concurrent writers over a
network filesystem corrupt the database. If you scale horizontally, move to
Postgres: keep `SqlConversationStore` as the reference, change `_connect` and
the parameter style (`?` → `%s`), and point `create_store` at it. The rest of the
application only knows the `ConversationStore` interface.

Both volumes matter: `DATABASE_URL` holds conversations and usage records,
`WORKSPACE_ROOT` holds cloned repositories. Losing the second is recoverable —
users re-clone. Losing the first loses history and billing data.

## Deploy

```bash
docker build -t ai-coding-workspace:$(git rev-parse --short HEAD) .

docker run -d --name acw \
  -p 8000:8000 \
  --env-file .env.production \
  -v acw-data:/data \
  --restart unless-stopped \
  ai-coding-workspace:$(git rev-parse --short HEAD)
```

Then confirm:

```bash
curl -fsS https://your-host/healthz     # process is alive
curl -fsS https://your-host/readyz      # every dependency is usable
```

`/readyz` returns 503 with a per-check breakdown when something is wrong. Wire
it to your load balancer so a broken instance leaves the pool instead of serving
errors.

## HTTPS and the reverse proxy

Terminate TLS at the proxy. Two SSE-specific settings are not optional — without
them the stream buffers and the UI appears frozen until the run finishes:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;

    proxy_buffering off;          # or SSE arrives in one lump at the end
    proxy_read_timeout 900s;      # a long investigation is not an idle connection

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Also send HSTS, `X-Content-Type-Options: nosniff`, and a CSP. The UI loads
Marked, DOMPurify, and Highlight.js from jsDelivr, so either allow that origin
or vendor the three files into `static/` and drop the CDN entirely (the app
degrades to escaped plain text if they fail to load, so it never renders unsafe
HTML either way).

## Network egress

The application needs outbound access to exactly three places: the OpenAI API,
`github.com`, and your CDN if you kept it. Sandbox containers need **none** —
they already run with `--network none`. Restrict egress to that list; it is the
cheapest defence against a compromised dependency phoning home.

## Observability

Logs are one JSON object per line on stdout, each carrying a `request_id` that
is also returned in the `X-Request-ID` header. When a user reports a failure,
that header is the search key.

Worth alerting on:

| Signal | Why |
| --- | --- |
| `/readyz` failing | an instance cannot serve |
| 5xx rate | unhandled errors |
| daily spend vs `DAILY_COST_LIMIT_USD` | runaway cost |
| `termination_reason` = `token_budget` rising | budget too small, or the prompt regressed |
| `cache_hit_rate` falling | the cacheable prefix broke — check `READ_LOOP_PROMPT_CACHE_KEY` |
| `provider_error` rate | upstream trouble |
| sandbox `isolated: false` in production | `SANDBOX_MODE` is wrong |

## Rollback

Releases are immutable images tagged by commit SHA, so rolling back is running
the previous tag. **Do this first and diagnose afterwards** — a rollback is
cheap and an outage is not.

```bash
# 1. What is running, and what ran before it?
docker ps --filter name=acw --format '{{.Image}}'
docker images ai-coding-workspace --format '{{.Tag}}\t{{.CreatedAt}}' | head

# 2. Swap back to the last known-good tag.
docker stop acw && docker rm acw
docker run -d --name acw -p 8000:8000 \
  --env-file .env.production -v acw-data:/data --restart unless-stopped \
  ai-coding-workspace:<previous-sha>

# 3. Verify.
curl -fsS https://your-host/readyz
```

The data volume is deliberately untouched — conversations and usage records
survive a rollback.

**If the bad release migrated the database**, rolling the image back is not
enough: restore the volume from backup first, then start the old image. Take a
volume snapshot before any release that changes the schema.

```bash
docker run --rm -v acw-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/acw-data-$(date +%Y%m%d-%H%M).tar.gz /data
```

## Release checklist

- [ ] `pytest -q` green
- [ ] image builds and the container smoke test passes
- [ ] staging `/readyz` returns 200
- [ ] one read-only run on staging returns a cited answer
- [ ] `AUTH_ENABLED=true`, `SANDBOX_MODE=docker`, real `FLASK_SECRET_KEY`
- [ ] data volume snapshot taken
- [ ] previous image tag written down where the on-call person can find it
