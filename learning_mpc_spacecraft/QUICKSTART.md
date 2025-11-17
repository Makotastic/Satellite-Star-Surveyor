# Quick Start Guide

This guide will help you get the Learning-Augmented MPC spacecraft project up and running quickly.

## Prerequisites

- Docker and Docker Compose installed
- OR VSCode with Remote-Containers extension

## Option 1: Using Docker Compose (Recommended)

### 1. Build and Start the Container

```bash
cd learning_mpc_spacecraft
docker-compose up -d
```

### 2. Enter the Container

```bash
docker-compose exec mpc_spacecraft bash
```

### 3. Install the Package

```bash
pip install -e .
```

### 4. Run Tests (Optional)

```bash
pytest tests/
```

### 5. Start Jupyter Notebook (Optional)

```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open the URL shown in your browser.

### 6. Stop the Container

```bash
docker-compose down
```

## Option 2: Using VSCode DevContainer

### 1. Open in VSCode

```bash
cd learning_mpc_spacecraft
code .
```

### 2. Reopen in Container

- Press `F1` or `Ctrl+Shift+P`
- Select "Dev Containers: Reopen in Container"
- Wait for the container to build and start

### 3. The environment is ready!

All dependencies are automatically installed via the `postCreateCommand`.

## Quick Commands

### Build the Docker image
```bash
docker-compose build
```

### Start the container in background
```bash
docker-compose up -d
```

### View container logs
```bash
docker-compose logs -f
```

### Execute commands in the container
```bash
docker-compose exec mpc_spacecraft python -c "import pydrake; print('Drake is ready!')"
```

### Stop and remove the container
```bash
docker-compose down
```

### Remove everything including volumes
```bash
docker-compose down -v
```

## Accessing Services

- **Meshcat Visualization**: http://localhost:7000
- **Jupyter Notebook**: http://localhost:8888

## Next Steps

1. Explore the notebooks in `notebooks/`
2. Run example scripts in `src/mpc_spacecraft/scripts/`
3. Check out the project structure in `README.md`
4. Start implementing Week 1 tasks from `full_project_plan.md`

## Troubleshooting

### Port already in use
If ports 7000 or 8888 are already in use, modify `docker-compose.yml`:
```yaml
ports:
  - "7001:7000"  # Change host port
  - "8889:8888"  # Change host port
```

### Permission issues
If you encounter permission issues with mounted volumes:
```bash
docker-compose exec mpc_spacecraft chown -R root:root /workspace
```

### Rebuild after changes
If you modify the Dockerfile:
```bash
docker-compose build --no-cache
docker-compose up -d