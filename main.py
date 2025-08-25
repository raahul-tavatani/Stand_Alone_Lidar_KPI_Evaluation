# main.py (add the import + branch)
import argparse
from utils.environment import setup_environment
from utils.sensor_utils import spawn_lidar
from tests import test_multidomain
from tests import test_radial_sep1 
import carla

def main():
    parser = argparse.ArgumentParser(description="DIN 91471 LiDAR Validation Suite")
    parser.add_argument('--prop', choices=['plane', 'mesh'], required=False)
    parser.add_argument('--test', choices=['angular_sep','radial_sep1','radial_sep2','false_alarm','multi_domain'],
                        required=True)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    client, world = setup_environment(use_mesh=(args.prop=='mesh'), headless=args.headless)

    # Sync mode @ 0.1 s
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    settings.no_rendering_mode = args.headless
    world.apply_settings(settings)

    # Spawn LiDAR at origin (0,0,1.7)
    blueprint_library = world.get_blueprint_library()
    lidar_location = carla.Location(x=0, y=0, z=1.7)
    lidar = spawn_lidar(world, blueprint_library, lidar_location)

    if args.test == 'radial_sep1':
        test_radial_sep1.run(client, world, lidar)
    elif args.test == 'multi_domain':
        test_multidomain.run(client, world, lidar)
    # (you can add other tests similarly)

if __name__ == '__main__':
    main()
