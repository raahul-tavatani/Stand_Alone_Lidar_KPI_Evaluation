import carla
import time

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.load_world('Proving_Ground')
    time.sleep(2)

    blueprint_library = world.get_blueprint_library()
    prop_bp = blueprint_library.find('static.prop.planea')
    if not prop_bp:
        print("Blueprint 'static.prop.planea' not found! Available static props:")
        for bp in blueprint_library.filter('static.prop.*'):
            print(bp.id)
        return

    spawn_location = carla.Location(x=5.0, y=0.0, z=2.0)
    spec_location = carla.Location(x=0.0, y=0.0, z=2.0)
    spawn_rotation = carla.Rotation(pitch=90.0, yaw=0.0, roll=60.0)
    transform = carla.Transform(spawn_location, spawn_rotation)

    try:
        prop_actor = world.spawn_actor(prop_bp, transform)
        print(f"Spawned prop: {prop_actor.id} at {transform.location}, rotation {transform.rotation}")
    except RuntimeError as e:
        print(f"Failed to spawn prop: {e}")
        return

    # Set spectator rotation to face prop correctly (adjust yaw to 180 or 270 as needed)
    spectator_rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=00.0)
    spectator_transform = carla.Transform(spec_location, spectator_rotation)

    spectator = world.get_spectator()
    spectator.set_transform(spectator_transform)
    print("Spectator moved and rotated to face prop correctly.")

    print("Prop spawned and will remain visible. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting script.")

if __name__ == '__main__':
    main()
