# CARLA Scenario Generation
This project aims to automatically synthesize traffic datasets with unusual events using the [CARLA Simulator](https://carla.readthedocs.io/en/latest/start_introduction/) and its Python API.

## Key features
* Generate bounding boxes with classes of simulated objects, collision information, and possibly object contours and lidar data.
* Define arbitrary simulation scenario without a need to change code.
* Simulate different conditions like map, weather, sensor location, and specific actor behavior automatically.
* Run multiple scenarios sequentially, robustly handle exceptions, and restart the simulation after each scenario to ensure the same conditions. 
* Generate traffic accident videos from captured frames and visualize annotations.

## Requirements
* 80-90GB of free space minimum (CARLA image + Extra maps).
* GPU with min 6GB VRAM.
* [Docker](https://www.docker.com/).
* [Docker Compose](https://docs.docker.com/compose/).
* To access GPU from Docker follow the official [documentation](https://docs.docker.com/engine/containers/resource_constraints/#gpu). 


## Run Application
Application is divided into 2 containers: CARLA simulator build from official image and Python Client that controls the simulation and produces frame captures, annotations, and videos.
The client also controls the simulator container, restarts it after each simulated scenario, and stops it after all scenarios are completed.

### Development 
Run the following command to build and start the application in the interactive / development mode.
```bash
docker compose up --build
```
The docker-compose.yml is automatically overridden with variables from docker-compose.override.yml, which specify the interactive behavior (opens CARLA window and pygame window with annotations in the main display).

### Production 
Run the following command to build and run in production / windowless mode (no overrides).
```bash
docker compose -f docker-compose.yml up --build
```

### CARLA Only
Run this command to build and start CARLA simulator only without the client application. This is useful when you want to set up a new scenario (via helper notebooks etc.).
```bash
docker compose -f docker-compose.manual.tml up --build
```

## General flow
1. Update runtime settings in __.env__ file.
2. Prepare or select a scenario in __src/client/scenarios__. When preparing new scenario, run the CARLA only build and use __notebooks/1.0-Create-new-scenario.ipynb__ to retrieve information from the simulator and put it into the new scenario config. See the __src/client/scenarios/EXAMPLE_SCENARIO.yaml__ for explanation.
3. Client's __main.py__ script runs over multiple scenario configs. For each scenario a _ScenariosMaker_ is called which creates multiple variants in a grid (mainly different weather or sensor settings). Then a _CarlaScenarioRunner_ is called to run different variants independently. Carla simulation is restarted after each variant.
4. This runs a single scenario variant using _CarlaSynthesizer_:
   - It runs a simulation in the [synchronous mode](https://carla.readthedocs.io/en/latest/adv_traffic_manager/#synchronous-mode).
   - Optionally, creates a main actor - EGO car cases.
   - Setups sensors, vehicles, and pedestrians.
   - Manually spawns or destroy additional actors (vehicles, pedestrian, and props) using hooks in the configuration file to creates unusual situations (e.g. a person running on a highway). Adds autopilot or manual controllers to them or collision sensors to capture their collisions with other actors or the environment. To see the available hooks or to create a new hook, use __src/client/hooks.py__.
   - Using different sensors, it projects bounding boxes, segmentations into the 2D camera space, captures collision data, or saves LiDAR information.
   - The sensor data (RGB images, lidar or collision data, images with projected annotations) are saved for frames in specified frequency (every n-th frame). Use _CarlaAnnotator_ to save the data in a specific format (ultralytics as default, coco). 
5. Simulated data are saved in __runs/out__ directory.
6. Use notebooks available in __src/notebooks__ to process simulated data or to create a synthetic dataset as ACCIDENT.  For instance, use _2.0-Generate-videos-from-frames-manually.ipynb_ to generate videos from captured images (works for Ultralytics-style annotations). 

### Scenario options
- See __src/client/scenarios/EXAMPLE_SCENARIO.yaml__ to view all supported options.


## Local CARLA installation (optional)
It is possible to exchange the _carla-simulator_ container for a locally running simulator program.
In such case:
* Remove __carla-simulator__ from the docker compose config.
* Change __CARLA_HOST_NAME__ environment variable for a correct localhost IP address (127.0.0.1).
* Change __USE_DOCKER__ environment variable to False.

Follow the official [documentation](https://carla.readthedocs.io/en/latest/start_quickstart/).

## Know issues
In the 0.9.15, Carla has a bug where GPU does not deallocate memory properly, leading to a crash due to malloc. This is mainly an issue when running many iteration (creating new objects, clients, etc.).
- https://github.com/carla-simulator/carla/issues/6068

Because of this, the Carla has to be restarted after each scenario/iteration, which requires the client application to have access to the host docker socket.

## Contributing (Pull Request)

Follow these steps to add new feature to the repository:
1. Create a new branch with name `[YOUR_NAME]/[SHORT_FEATURE_DESCRIPTION]`.
2. Add your changes to the branch.
3. Run quality checks using `black`, `isort`, `flake8`:
   ```bash
   black --check .  # checks code formatting according to PEP 8 standard
   isort --check .  # checks if the package imports are sorted alphabetically and separated into sections by type
   flake8 .  # checks if the code is valid
   ```
   Code can be formated automatically using [tox](https://tox.wiki/en/4.27.0/):
   ```bash
   tox run -e lint
   ```
4. Create a Pull Request to branch `dev`.
