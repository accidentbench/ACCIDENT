# CARLA Scenario Generation

This repository provides an automated framework for synthesizing traffic datasets, specifically focusing on **unusual traffic events and accidents**. By leveraging the [CARLA Simulator](https://carla.readthedocs.io/en/latest/start_introduction/) and its Python API, users can generate diverse, annotated datasets using custom simulation configurations.

---

## 🚀 Key Features

* **Automated Dataset Synthesis:** Generate comprehensive annotations including 2D/3D bounding boxes, object classes, instance segmentation (contours), and LiDAR point clouds.
* **Dynamic Scenario Configuration:** Define complex traffic scenarios via YAML configuration files without modifying the core codebase.
* **Environmental Variability:** Automatically iterate through various maps, weather conditions, sensor configurations, and actor behaviors.
* **Robust Execution Pipeline:** Sequentially runs multiple scenarios with automated exception handling and simulator restarts to ensure clean state transitions.
* **Post-Processing Tools:** Provided utilities for generating traffic accident videos from captured frames and visualizing ground-truth annotations.

---

## 💻 System Requirements

| Requirement     | Minimum Specification                                                                           |
|:----------------|:------------------------------------------------------------------------------------------------|
| **Storage**     | 80–90 GB (CARLA image + additional maps)                                                        |
| **GPU**         | NVIDIA GPU with at least 6GB VRAM                                                               |
| **Software**    | [Docker](https://www.docker.com/) & [Docker Compose](https://docs.docker.com/compose/)          |
| **GPU Drivers** | [NVIDIA Container Toolkit](https://docs.docker.com/engine/containers/resource_constraints/#gpu) |

---

## 🛠 Run Application

The architecture consists of two main docker containers:
1.  **CARLA Simulator:** The CARLA server build from an official image.
2.  **Python Client:** The controller logic that manages the simulation, captures frames, generates annotations, and handles the orchestration of scenarios.

### 1. Development Mode (Interactive)
Use this mode for debugging or visual verification. It launches the CARLA window and a `pygame` overlay showing real-time annotations.
```bash
docker compose up --build
```

### 2. Production Mode (Headless)
Optimized for data generation. It runs in windowless mode to save resources.
```bash
docker compose -f docker-compose.yml up --build
```

### 3. Manual / Scenario Setup
Launches the CARLA simulator alone. This is ideal when using helper notebooks to define new spawn points or scenario parameters.
```bash
docker compose -f docker-compose.manual.yml up --build
```

---

## 🔄 Workflow & General Flow

1. **Scenario Definition:** Select or create a configuration in `src/client/scenarios`. Use `notebooks/1.0-Create-new-scenario.ipynb` while the simulator is running to extract coordinates or actor paths. Refer to `EXAMPLE_SCENARIO.yaml` for syntax details.
2. **Configuration:** Update the `.env` file to specify which scenarios to run and adjust global runtime settings.
3. **Orchestration (`main.py`):**
    - **ScenarioMaker:** Generates a grid of scenario variants (e.g., same event, but different weather/sensors).
    - **CarlaScenarioRunner:** Executes each variant independently, restarting the CARLA container between runs to prevent memory leaks.
4. **Simulation & Synthesis (`CarlaSynthesizer`):**
    - Operates in **Synchronous Mode** for precise frame-to-annotation alignment.
    - Spawns the actors (vehicles, pedestrians, etc.) and configures the sensor suite.
    - Utilizes **Hooks** (`src/client/hooks.py`) to trigger unusual events (e.g., a vehicle with specific route, pedestrians crossing highways or sudden braking).
    - Captures and saves data (RGB, LiDAR, Collisions) at a specified frequency.
    - **CarlaAnnotator** formats data (default: **Ultralytics/YOLO**, optional: **COCO**).
5. **Output:** Results are stored in the `runs/out` directory.
6. **Post-Processing:** Use `src/notebooks/2.0-Generate-videos-from-frames-manually.ipynb` to compile captured frames into video files.

---

## 🔌 Local CARLA Installation (Optional)

To use a local CARLA installation instead of the Docker container:
1.  Remove the `carla-simulator` service from the `docker-compose` configuration.
2.  Set `USE_DOCKER=False` in your `.env` file.
3.  Set `CARLA_HOST_NAME=127.0.0.1`.
4.  Ensure your local simulator is running before starting the client.

---

## ⚠️ Known Issues

**Memory Deallocation (v0.9.15):**
CARLA 0.9.15 has a known bug where GPU memory is not properly deallocated after repeated object spawning or client reconnections. This leads to `malloc` crashes.
* **Solution:** The system is designed to restart the CARLA container after every scenario. For this to function, the client container must have access to the **host docker socket**.

---

## 🤝 Contributing

We welcome improvements! Please follow these steps:
1.  Create a branch: `[your_name]/[feature_description]`.
2.  Implement changes.
3.  **Linting & Quality Assurance:**
    ```bash
    # Manual checks
    black --check .
    isort --check .
    flake8 .

    # Or auto-format using tox
    tox run -e lint
    ```
4.  Open a Pull Request against the `dev` branch.
