# external/spot/test_spot_model.py

import time

from spot_model import build_spot_robot_diagram


def main():
    # Build the robot and visualize the default pose
    model = build_spot_robot_diagram()

    # If Meshcat was started inside build_spot_robot_diagram, it will already
    # have printed the URL in the terminal. We can also print it explicitly:
    if model.meshcat is not None:
        try:
            print("Meshcat URL:", model.meshcat.web_url())
        except Exception:
            # Older Drake versions may not have web_url(); StartMeshcat prints it anyway.
            pass

    print("Spot model built and published to Meshcat.")
    print("Leave this script running and open the Meshcat URL in your browser.")
    print("Press Ctrl+C here to quit.")

    # Just sleep forever so the process (and Meshcat connection) stays alive.
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
