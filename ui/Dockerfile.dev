FROM node:22-alpine
WORKDIR /app

# Install dependencies at build time so they're cached in the image layer.
# Docker Watch rebuilds this image (re-running npm install) when package*.json changes.
COPY package*.json ./
RUN npm install

# Copy source — Docker Watch syncs incremental changes without rebuilding.
COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev"]
