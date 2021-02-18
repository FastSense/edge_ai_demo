# Overview

Contains utilities for quickly and conveniently launching the docker container [ros_ai](https://hub.docker.com/r/fastsense/ros_ai)

## Work with container

In order to start a container:

> **Hint** If there is no container in the system, it will be loaded automatically.

```bash
docker-compose up -d
```

To open a bash session inside a container:

```bash
docker-compose exec ros_ai bash
```
