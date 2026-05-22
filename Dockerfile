# Stage 1: compile Rust API + buildindex binary (fully offline via vendored deps),
# then download reference data and build the IVF binary index.
FROM rust:slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY rust-api/ .
RUN cargo build --release --locked --offline --bins && \
    mkdir -p /data && \
    curl -fsSL "https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz" \
         -o /data/references.json.gz && \
    curl -fsSL "https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/mcc_risk.json" \
         -o /data/mcc_risk.json && \
    curl -fsSL "https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/normalization.json" \
         -o /data/normalization.json && \
    ./target/release/buildindex \
        -in  /data/references.json.gz \
        -out /data/index.ivf.bin \
        -centroids 5000 \
        -sample    0.1 \
        -iters     20 \
        -probes    15

# Stage 2: minimal runtime image
FROM debian:bookworm-slim
# wget is used by the Docker healthcheck (GET /ready).
# debian:bookworm-slim has no bash or curl, so wget is the smallest option.
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/rinha-api /usr/local/bin/rinha-api
COPY --from=builder /data/index.ivf.bin            /data/index.ivf.bin
COPY --from=builder /data/index.raw_f32.bin        /data/index.raw_f32.bin
COPY --from=builder /data/mcc_risk.json            /data/mcc_risk.json
COPY --from=builder /data/normalization.json       /data/normalization.json
ENV INDEX_PATH=/data/index.ivf.bin
ENV IVF_PROBES=15
ENV TOKIO_WORKER_THREADS=2
EXPOSE 9999
CMD ["rinha-api"]
