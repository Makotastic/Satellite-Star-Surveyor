# Docker Build Optimization Guide

This guide explains the optimizations implemented to speed up Docker builds for the Learning-Augmented MPC spacecraft project.

## Implemented Optimizations

### 1. BuildKit Syntax (Line 1)
```dockerfile
# syntax=docker/dockerfile:1.4
```
**Benefit:** Enables advanced BuildKit features including parallel builds and better caching.

### 2. Cache Mounts for apt-get
```dockerfile
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked
```
**Benefit:** Reuses apt package cache across builds, avoiding re-downloading packages.

### 3. Cache Mounts for pip
```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip
```
**Benefit:** Reuses pip download cache, significantly faster for Python package installation.

### 4. Layer Optimization
- System packages installed first (rarely change)
- Virtual environment setup (rarely changes)
- requirements.txt copied separately (better caching)
- Python packages installed last (changes occasionally)

**Benefit:** Only rebuilds layers that actually changed.

### 5. .dockerignore File
Excludes unnecessary files from build context:
- Python cache files
- Git history
- IDE configurations
- Large data files
- Documentation (except README)

**Benefit:** Faster context transfer to Docker daemon.

## How to Use

### Enable BuildKit

**Linux/Mac:**
```bash
export DOCKER_BUILDKIT=1
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
```

**Windows PowerShell:**
```powershell
$env:DOCKER_BUILDKIT=1
```

**Windows CMD:**
```cmd
set DOCKER_BUILDKIT=1
```

### Build with Optimizations

```bash
# First build (will be slower)
docker-compose build

# Subsequent builds (much faster with cache)
docker-compose build

# Force rebuild without cache (when needed)
docker-compose build --no-cache
```

## Expected Build Times

| Scenario | Without Optimization | With Optimization |
|----------|---------------------|-------------------|
| First build | 10-20 minutes | 8-15 minutes |
| Rebuild (no changes) | 5-10 minutes | 10-30 seconds |
| Rebuild (code changes) | 5-10 minutes | Instant* |
| Rebuild (requirements.txt change) | 5-10 minutes | 1-3 minutes |

*Code changes don't require rebuild due to volume mounts

## Advanced Optimizations

### Pre-pull Base Image
```bash
docker pull robotlocomotion/drake:latest
```
**When:** Before first build or when base image updates.

### Parallel Package Installation
Already implemented in Dockerfile with pip cache mounts.

### Multi-stage Builds (Future)
For production deployments, consider multi-stage builds to reduce final image size:

```dockerfile
# Stage 1: Builder
FROM robotlocomotion/drake:latest AS builder
# ... install everything ...

# Stage 2: Runtime
FROM robotlocomotion/drake:latest
COPY --from=builder /opt/venv /opt/venv
```

## Troubleshooting

### Cache Not Working
```bash
# Clear Docker build cache
docker builder prune

# Rebuild
docker-compose build
```

### BuildKit Not Enabled
Check if BuildKit is enabled:
```bash
docker buildx version
```

If not available, update Docker to latest version.

### Slow Network
If downloading packages is slow:
1. Use a closer mirror for apt packages
2. Consider using a pip cache server
3. Pre-download large packages

## Monitoring Build Performance

### View Build Cache Usage
```bash
docker system df
```

### View Build History
```bash
docker history learning_mpc_spacecraft:latest
```

### Analyze Build Time
```bash
time docker-compose build
```

## Best Practices

1. **Don't disable cache unless necessary** - Cache is your friend
2. **Keep .dockerignore updated** - Exclude unnecessary files
3. **Order Dockerfile commands wisely** - Least changing first
4. **Use volume mounts for development** - No rebuild needed for code changes
5. **Clean up periodically** - Remove unused images and cache

```bash
# Clean up unused Docker resources
docker system prune -a
```

## Development Workflow

For fastest iteration:

```bash
# Build once
docker-compose up -d

# Make code changes
# (No rebuild needed - uses volume mount)

# Only rebuild when changing:
# - Dockerfile
# - requirements.txt
# - System dependencies
docker-compose build
```

## Comparison: Before vs After

### Before Optimization
- No BuildKit features
- No cache mounts
- No .dockerignore
- Inefficient layer ordering
- **Result:** 5-10 minute rebuilds

### After Optimization
- BuildKit enabled
- Cache mounts for apt and pip
- Comprehensive .dockerignore
- Optimized layer ordering
- **Result:** 10-30 second rebuilds (with cache)

## Additional Resources

- [Docker BuildKit Documentation](https://docs.docker.com/build/buildkit/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Layer Caching](https://docs.docker.com/build/cache/)