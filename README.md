# AI Evolution Habitat - Simulation project

An interactive evolutionary simulation built using Pygame, NumPy, and OpenCV. This project simulates an environment where two teams of neural-network-controlled agents (Blue: Tank/Melee and Yellow: Range) evolve over generations to optimize their survival instincts, locate food, avoid hunters, utilize shelters, and compete with the opposing team.

## Features

- **Neural Network-Controlled Agents:** Agents make decisions (movement speed and direction) using a simple neural network feed-forward system, initialized with survival instincts/heuristics.
- **Genetic Algorithm & Evolution:** Best performing agents pass their neural network weights (brains) to the next generation with random mutations.
- **Team Dynamics:** 
  - **Blue Team (Tank/Melee):** High health, damage mitigation, short range.
  - **Yellow Team (Range):** Normal health, high damage, long range.
- **Interactive Environment:**
  - **Food:** Recharges health and increases fitness score.
  - **Hunters:** Chase agents, eat them to restore their own health, and can be attacked and defeated by agents.
  - **Shelters:** Safe zones where agents can hide temporarily from hunters and enemies.
- **Video Export:** Automatically records and exports the simulation sessions as high-quality `.mp4` video files.

## Requirements

Ensure you have Python installed, along with the following dependencies:

```bash
pip install pygame numpy opencv-python
```

## How to Run

Simply run the main script:

```bash
python simulation.py
```

### Controls
- Press **N** to manually skip to the next generation.
- Press **Q** or close the window to quit and save the video.
