FROM rust:alpine AS builder
RUN apk add --no-cache musl-dev
RUN rustup default nightly

WORKDIR /opt/eevee
ADD src src
ADD examples examples
ADD benches benches
ADD Cargo.toml Cargo.lock .
RUN cat Cargo.toml


ARG TARGET
ARG BUILD_OPT="-r"
RUN cargo build ${BUILD_OPT} --example ${TARGET}

FROM alpine:latest
WORKDIR /opt/eevee

ARG TARGET
COPY --from=builder /opt/eevee/target/release/examples/${TARGET} /opt/eevee/run

VOLUME /opt/eevee/output
ENTRYPOINT ["/opt/eevee/run"]
