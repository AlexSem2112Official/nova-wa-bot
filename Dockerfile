FROM ghcr.io/ggml-org/llama.cpp:server
USER root
ENTRYPOINT []
RUN apt-get update && apt-get install -y curl ca-certificates git \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*
WORKDIR /bot
COPY package.json ./
RUN npm install --omit=dev
COPY . .
ENV LLAMA_BIN=/app/llama-server
CMD ["node", "index.js"]
