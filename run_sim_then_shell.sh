#!/usr/bin/env bash

# Run the standing simulation; on Ctrl+C (SIGINT) or completion, drop to a shell
# instead of exiting the container.

set +e

drop_to_shell() {
  echo "Stopping simulation; opening a shell..."
  exec bash
}

trap drop_to_shell INT TERM

python3 spot_standing_sim.py
status=$?
echo "Simulation exited with status ${status}; opening a shell..."
exec bash
