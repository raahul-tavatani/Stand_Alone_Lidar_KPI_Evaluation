# utils/environment.py
import carla
from utils.sensor_utils import spawn_lidar

def setup_environment(use_mesh=False, headless=False):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    print("[INFO] Loading map: Proving_Ground")
    world = client.load_world('Proving_Ground')

    # Headless rendering
    if headless:
        settings = world.get_settings()
        settings.no_rendering_mode = True
        world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # === Spawn LiDAR at origin ===
    #lidar_location = carla.Location(x=0, y=0, z=1.7)
    #lidar = spawn_lidar(world, blueprint_library, lidar_location)

    return client, world#, lidar
