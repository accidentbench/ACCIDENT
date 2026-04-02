# CARLA Versatile Scenario Generation
This project aims to automatically synthesize traffic datasets with unusual events using the [CARLA Simulator](https://carla.readthedocs.io/en/latest/start_introduction/) and its Python API.

### Key features
* Generate bounding boxes with classes of simulated objects, collision information, and possibly object contours and lidar data.
* Define arbitrary simulation scenario without a need to change code.
* Simulate different conditions like map, weather, sensor location, and specific actor behavior automatically.
* Run multiple scenarios sequentially, robustly handle exceptions, and restart the simulation after each scenario to ensure the same conditions.

## Requirements
* 80-90GB of free space minimum (CARLA image + Extra maps).
* GPU with min 6GB VRAM.
* [Docker](https://www.docker.com/).
* [Docker Compose](https://docs.docker.com/compose/).
* To access GPU from Docker follow the official [documentation](https://docs.docker.com/engine/containers/resource_constraints/#gpu). 



## Run Application

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

### Carla Only
Run this command to build and start CARLA simulator only without the client application. This is useful when you want to set up a new scenario (via helper notebooks etc.).
```bash
docker compose -f docker-compose.manual.tml up --build
```

## General flow
1. Update runtime settings in __.env__ file.
2. Prepare or select a scenario in __src/client/scenarios__.
3. Client's __main.py__ script functions over multiple scenario configs. For each scenario a _ScenariosMaker_ is called which creates multiple variants in a grid (mainly different weather or sensor settings). Then a _CarlaScenarioRunner_ is called to run different variants independently. Carla simulation is restarted after each variant.
4. Run single scenario variant using _CarlaSynthesizer_:
   - Run simulation in the [synchronous mode](https://carla.readthedocs.io/en/latest/adv_traffic_manager/#synchronous-mode).
   - Create main actor (optional) - EGO car cases.
   - Setup sensors, vehicles, and pedestrians.
   - Manually spawn or destroy additional actors (vehicles, pedestrian, and props) using hooks in the configuration file to create unusual situations (e.g. a person running on a highway). Add simple controllers to them or collision sensors to capture their collisions with other actors or the environment. 
   - Project bounding box, collision or lidar information into the sensor plane, handle edge-cases and capture it.
   - Save the sensor data (RGB images, lidar or collision data, images with projected annotations) for frames in specified frequency (every n-th frame). Use _CarlaAnnotator_ to save the data in Ultralytics (preferred) or COCO format.
5. Simulated data are saved in __out__ directory.
6. Other useful scripts are available in __src/notebooks__ and __src/scripts__ directories. For instance, use _generate_videos_from_images.ipynb_ to generate videos from captured images (works for Ultralytics-style annotations). 

### Scenario options
- See __src/client/scenarios/EXAMPLE_SCENARIO.yaml__ to view all supported options.


## Local CARLA installation (optional)
It is possible to exchange the _carla-simulator_ container for a locally running simulator instance.
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
