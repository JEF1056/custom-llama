FROM node:22-alpine
WORKDIR /app

# Install dependencies at build time so they're cached in the image layer.
# Docker Watch rebuilds this image (re-running npm ci) when package*.json changes.
# npm ci is faster and more reliable than npm install in container environments —
# it installs directly from package-lock.json without dep resolution.
COPY package*.json ./
RUN npm install --no-audit --no-fund

# Copy source — Docker Watch syncs incremental changes without rebuilding.
COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"]
