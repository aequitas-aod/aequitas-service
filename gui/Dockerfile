# Containerises the Streamlit demo in gui/.
#
# Build context must be the repository root (not gui/), because
# requirements.txt installs the local `framework` package (`-e ./framework`)
# and the app also reads from tools_catalog/ and framework/ at runtime.
#
# Build (from the repository root):
#   docker build -f gui/Dockerfile -t aequitas-gui .
# Run:
#   docker run --rm -p 8501:8501 --env-file .env aequitas-gui

FROM python:3.10-slim

WORKDIR /app

# build-essential covers C-extension dependencies (e.g. pulled in by
# mlserver/holisticai) that don't ship a prebuilt wheel for this image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependency manifests first for better layer caching. `framework` must be
# present before `pip install`, since requirements.txt installs it as a
# local editable package (`-e ./framework`).
COPY requirements.txt ./requirements.txt
COPY framework ./framework

RUN pip install --no-cache-dir -r requirements.txt

# Application code. tools_catalog/ is read at runtime by the GUI (tool
# suggestions shown alongside each operation).
COPY tools_catalog ./tools_catalog
COPY gui ./gui

EXPOSE 8501

CMD ["streamlit", "run", "gui/Main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
