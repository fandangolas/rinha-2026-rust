# syntax=docker/dockerfile:1

# ---- Stage 1: build the index builder tool --------------------------------
FROM golang:1.23-alpine AS builder-tool

WORKDIR /src
COPY go.mod go.sum ./
COPY api/ ./api/

WORKDIR /src/api/cmd/buildindex
RUN go build -o /buildindex .

# ---- Stage 2: build the IVF index from reference data --------------------
FROM alpine:3.20 AS indexer

# Install curl to download reference data files.
RUN apk add --no-cache curl

COPY --from=builder-tool /buildindex /buildindex

# Download reference data.
# Set REFERENCES_URL, MCC_RISK_URL, NORM_URL as build args pointing to
# wherever the challenge hosts the data files.
ARG REFERENCES_URL
ARG MCC_RISK_URL
ARG NORM_URL

RUN mkdir -p /data && \
    curl -fsSL "${REFERENCES_URL}"  -o /data/references.json.gz && \
    curl -fsSL "${MCC_RISK_URL}"    -o /data/mcc_risk.json      && \
    curl -fsSL "${NORM_URL}"        -o /data/normalization.json

# Build the IVF index.
# Tune -centroids and -probes here; they are also overridable at runtime
# via the IVF_PROBES env var without rebuilding.
RUN /buildindex \
      -in   /data/references.json.gz \
      -out  /data/index.ivf.bin      \
      -centroids 1000                \
      -sample    0.1                 \
      -iters     20                  \
      -probes    20

# ---- Stage 3: build the API binary ----------------------------------------
FROM golang:1.23-alpine AS api-builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY api/ ./api/
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /api ./api/cmd/server

# ---- Stage 4: minimal runtime image --------------------------------------
FROM alpine:3.20

COPY --from=api-builder /api          /api
COPY --from=indexer     /data/index.ivf.bin    /data/index.ivf.bin
COPY --from=indexer     /data/mcc_risk.json    /data/mcc_risk.json
COPY --from=indexer     /data/normalization.json /data/normalization.json

ENV PORT=9999
ENV INDEX_PATH=/data/index.ivf.bin
ENV MCC_RISK_PATH=/data/mcc_risk.json
ENV NORM_PATH=/data/normalization.json
# Switch to "hnsw" once the HNSW implementation is ready.
ENV SEARCHER=ivf
# Number of clusters to probe per query. Higher = better recall, more latency.
ENV IVF_PROBES=20

# Tune GC: rely entirely on GOMEMLIMIT rather than GOGC heap-doubling heuristic.
# Set GOMEMLIMIT to ~80% of the container's memory budget.
ENV GOGC=off
ENV GOMEMLIMIT=120MiB

EXPOSE 9999
ENTRYPOINT ["/api"]
