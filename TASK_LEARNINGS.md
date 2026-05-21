# Task Learnings

## Task 2/24: Dockerfile dependency flow

- **No Dockerfile changes needed** for new Python dependencies. The `mcp-search-server/Dockerfile` already copies `requirements.txt` in the builder stage (`COPY requirements.txt .` → `pip install --prefix=/install -r requirements.txt`) and copies installed packages to the runtime stage (`COPY --from=builder /install /usr/local`).
- **To include new deps**: run `docker compose build mcp-search-server`. Docker's layer caching will invalidate the `pip install` layer when `requirements.txt` changes, reinstalling all dependencies including the new ones.
- **Dependencies added in Task 1/24**: `openpyxl>=3.1.0` and `python-pptx>=0.6.23` — both will be picked up automatically on next build.
