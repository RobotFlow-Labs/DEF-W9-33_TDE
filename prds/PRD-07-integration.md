# PRD-07: Integration

> Status: TODO
> Module: 33_TDE
> Depends on: PRD-06

## Objective
Docker serving infrastructure, ROS2 node, ANIMA registry integration.

## Components

### 1. Docker Serving (Dockerfile.serve)
- Base: ghcr.io/robotflow-labs/anima-serve:jazzy
- Install module + weights download at runtime
- FastAPI endpoints: /health, /ready, /info, /predict
- GPU detection + weight format priority (TRT > ONNX > safetensors)

### 2. Docker Compose (docker-compose.serve.yml)
- Profiles: serve, ros2, api, test
- Environment: ANIMA_MODULE_NAME=project_tde
- Port: 8080 (configurable via ANIMA_SERVE_PORT)
- network_mode: host (for ROS2 DDS)

### 3. ROS2 Node (src/anima_tde/serve.py)
- AnimaNode subclass
- setup_inference: load model weights
- process: run detection on input image
- Publishes: detection results as ROS2 messages

### 4. ANIMA Module YAML (anima_module.yaml)
- Module identity, version, capabilities
- API schema, Docker config, ROS2 topics

## Acceptance Criteria
- `docker compose --profile api up` starts and /health returns 200
- /predict accepts image, returns bounding boxes
- anima_module.yaml validates against schema
- ROS2 node publishes detections (if rclpy available)
