# Start from Drake's official image (includes pydrake, solvers, etc.)
FROM robotlocomotion/drake:latest

# Set a dedicated workspace directory
WORKDIR /workspace

# ---- System utilities & Python tooling ----
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git \
        curl \
        build-essential \
        python3-venv \
        python3-pip \
        python3-dev \
        vim \
    && rm -rf /var/lib/apt/lists/*

# ---- Create and use a virtual environment ----
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Upgrade pip
RUN pip install --upgrade pip

# ---- Core Python deps for this project (you can trim/adjust) ----
# Note: pydrake is already available from the base image; DO NOT pip install it.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Make sure Jupyter kernel is available in notebooks
RUN python -m ipykernel install --name=mpc_spacecraft --display-name "Python (mpc_spacecraft)"

# ---- (Optional) Locale & time zone setup, if you care ----
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Expose Meshcat default port
EXPOSE 7000

# Default command
CMD [ "bash" ]
