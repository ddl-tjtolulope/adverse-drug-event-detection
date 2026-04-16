# Domino Compute Environment Setup — ADE Detection Workshop

Make the changes below to the Domino compute environment for the workshop.
Leave all other entries as Default.

### Name:
```
ADE-Detection-Workshop
```

### Base Environment / Image:

Select:
```
Start from a custom base image
```
From:
```
quay.io/domino/domino-standard-environment:ubuntu22-py3.10-r4.5-domino6.1-standard
```

### Description:
```
ADE Detection Workshop Environment
6.1 Domino Standard Environment Py3.10 R4.5
Ubuntu 22.04 | Python 3.10 | R 4.5
Jupyter, JupyterLab, VSCode, RStudio
```

### Visibility:
```
Globally Accessible
```

### Additional Dockerfile Instructions:
```
# Install uWSGI pre-requisites
USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add missing library for uWSGI
ARG LDFLAGS=-fno-lto
ENV LDFLAGS=-fno-lto
ENV PYTHONPATH="${PYTHONPATH}:/mnt/code:/mnt"

# Install required packages
RUN pip install --no-cache-dir \
    Flask Flask-Compress Flask-Cors uwsgi six prometheus-client \
    ydata_profiling streamlit st-pages streamlit-extras \
    xgboost scikit-learn pandas numpy boto3 \
    mlflow flytekitplugins-domino
```

### Pluggable Workspace Tools:
```
jupyter:
  title: "Jupyter (Python, R, Julia)"
  iconUrl: "/assets/images/workspace-logos/Jupyter.svg"
  start: [ "/opt/domino/workspaces/jupyter/start" ]
  supportedFileExtensions: [ ".ipynb" ]
  httpProxy:
    port: 8888
    rewrite: false
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    requireSubdomain: false
jupyterlab:
  title: "JupyterLab"
  iconUrl: "/assets/images/workspace-logos/jupyterlab.svg"
  start: [ "/opt/domino/workspaces/jupyterlab/start" ]
  httpProxy:
    internalPath: "/{{ownerUsername}}/{{projectName}}/{{sessionPathComponent}}/{{runId}}/{{#if pathToOpen}}tree/{{pathToOpen}}{{/if}}"
    port: 8888
    rewrite: false
    requireSubdomain: false
vscode:
  title: "vscode"
  iconUrl: "/assets/images/workspace-logos/vscode.svg"
  start: [ "/opt/domino/workspaces/vscode/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
rstudio:
  title: "RStudio"
  iconUrl: "/assets/images/workspace-logos/Rstudio.svg"
  start: [ "/opt/domino/workspaces/rstudio/start" ]
  httpProxy:
    port: 8888
    requireSubdomain: false
```

Build It
