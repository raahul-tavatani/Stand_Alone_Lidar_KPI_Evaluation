import argparse
from utils.environment import setup_environment
from utils.sensor_utils import spawn_lidar
from tests import test_multidomain
import carla

def main():
    parser = argparse.ArgumentParser(description="DIN 91471 LiDAR Validation Suite")
    parser.add_argument('--prop', choices=['plane', 'mesh'], required=False,
                        help='Choose the prop to use: "plane" or "mesh"')
    parser.add_argument('--test', choices=[
        'angular_sep', 'radial_sep1', 'radial_sep2', 'false_alarm', 'multi_domain'
    ], required=True, help='Select the test to run')
    parser.add_argument('--headless', action='store_true', help='Run without rendering')
    args = parser.parse_args()

    # === Set up environment ===
    client, world= setup_environment(
        use_mesh=(args.prop == 'mesh'),
        headless=args.headless
    )

    # === Apply synchronous mode and fixed time step ===
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1  # Match sensor_tick!
        settings.no_rendering_mode = args.headless
        world.apply_settings(settings)
        print("[INFO] Synchronous mode enabled (Δt = 0.1s)")
    except RuntimeError as e:
        print(f"[ERROR] Failed to set synchronous mode: {e}")
        return

    # === Spawn standalone LiDAR ===
    blueprint_library = world.get_blueprint_library()
    lidar_location = carla.Location(x=0, y=0, z=1.7)  # Place it at origin, 2m above ground
    lidar = spawn_lidar(world, blueprint_library, lidar_location)

    # === Run selected test ===
    if args.test == 'multi_domain':
        test_multidomain.run(client, world, lidar)

if __name__ == '__main__':
    main()
