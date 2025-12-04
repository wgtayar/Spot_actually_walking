<h1 align="center">🌿🌿🌿 BASIL: Spot Submodule 🌿🌿🌿</h1>
<h1 align="center">Better Actuation of Spot Ignited at LAU</h1>

### Quadruped Locomotion and Control Playground

This repository is the **Spot submodule** of the 🌿BASIL🌿 project.

`BASIL / Spot` is a Drake and Python playground for experimenting with advanced control of Boston Dynamics’ Spot robot model.  
It builds on the MIT Underactuated Robotics Spot examples and adds:

- Standing PD controllers with disturbance tests  
- Joint-space and full-state LQR regulators  
- MPC Control of Spot  
- Kino-dynamic jumping and multi-step trot demos with passive knees  
- A clean structure for extending behaviors and controllers  

All of this runs in simulation (no real robot required).

---

## Quick Start

### 1. Clone the Repository (with All Submodules)

**Recommended (via 🌿BASIL🌿 meta-repo):**

```bash
git clone --recurse-submodules git@github.com:wgtayar/basil.git basil
cd basil/external/spot
```

This gives you both the Spot playground and the matching `external/underactuated` dependency.

**Standalone usage (advanced):**

```bash
git clone git@github.com:wgtayar/Spot_actually_walking.git basil-spot
cd basil-spot
```

If you use this repo standalone, you are responsible for making the `underactuated` Python package available yourself (for example by installing your own clone of the MIT Underactuated repo with `pip install -e /path/to/underactuated --no-deps`) and adjusting the Docker paths accordingly.

---

### 2. If You Already Cloned BUT Without Submodules

If directories from `underactuated` or other externals look empty:

```bash
# From BASIL:
cd basil
git submodule update --init --recursive

# Standalone clone:
git submodule update --init --recursive
```

This guarantees that the required underactuated code and its submodules are present.

---

### 3. Docker quickstart (from 🌿BASIL🌿)

When used inside the 🌿BASIL🌿 meta-repo (recommended), the Docker setup assumes you are in:

```bash
cd basil/external/spot
```

From there, Docker Compose will build the image using the BASIL root as the build context (so it can see `external/underactuated` and the rest of the workspace).

Build the image:

```bash
docker compose build spot-sim
```

#### Option A (recommended): Run sim, then land in a shell

Run the standing sim with Meshcat on port 7000. Press `Ctrl+C` to stop the sim and drop into a shell in the same container (at `/workspace/basil/external/spot`). Type `exit` to leave the container.

```bash
docker compose run --rm --service-ports spot-sim
```

Open <http://localhost:7000> in your browser for Meshcat.

#### Option B: Keep container up in the background

Start the service detached:

```bash
docker compose up -d
```

Open <http://localhost:7000> for Meshcat, then exec into the container to run scripts:

```bash
docker compose exec spot-sim bash
```

Stop everything when done:

```bash
docker compose down
```

> **Note:** The `docker-compose.yml` and `Dockerfile` are wired to the BASIL layout, with `external/underactuated` available at `/workspace/basil/external/underactuated`. If you use this Spot repo standalone, you will need to update the `build.context`, paths in the Dockerfile, and `PYTHONPATH` to match your own directory structure.

---

### 4. Working with branches

The container always sees whatever branch is checked out on your host because the repo is mounted into `/workspace/basil`. Typical flow (from BASIL):

```bash
git clone --recurse-submodules git@github.com:wgtayar/basil.git basil
cd basil/external/spot
git checkout <branch-name>
docker compose run --rm --service-ports spot-sim
```

If you switch branches and the dependencies change, rebuild the image:

```bash
docker compose build spot-sim
```

---

### 5. Running a First Demo (Local Python) (NOT Recommended)

If you are not using Docker and have a Python environment set up (for example via `basil/requirements.txt` plus `pip install -e external/underactuated --no-deps`):

```bash
cd basil/external/spot    # or standalone clone

# Stand in place with the nominal PD controller
python3 spot_standing_sim.py

# Stand with a disturbance injected at t = 15 s, magnitude = 0.6 rad
python3 spot_standing_sim.py   --disturbance_time 15.0   --disturbance_magnitude 0.6
```

Other scripts can be run in the same way:

```bash
python3 spot_multi_step_playback.py
# or any other example script, see headers and comments in the files
```

---

## Notes and Best Practices

- Ensure both this repo and the `underactuated` dependency are initialized (`git submodule update --init --recursive` at the 🌿BASIL🌿 root is safest).  
- Use Docker for a fully reproducible environment, or `basil/requirements.txt` + `pip install -e external/underactuated --no-deps` for a lighter local install.  
- Treat branches as experiments: one branch per major controller or trajectory change.  

---

*Enjoy BASIL and making Spot actually walk like a pro 🌿*
