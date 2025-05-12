### Installation

* **Using pip:**

  ```bash
  pip install -r requirements.txt
  ```

* **Using conda:**

  ```bash
  conda env create -f environment.yml
  ```

---

### File Overview

* **`wind.py`**
  Simulates real-time wind effects.
  *(Run with: `mjpython wind.py`)*

* **`wind_re.py`**
  Records the wind simulation for later playback or analysis.
  *(Run with: `mjpython wind_re.py`)*

* **`acrobot_env.py`**
  Custom environment wrapper. You can modify the reward function and other environment settings here (e.g., energy cost).

* **`agent.py`**
  Sample script to train a model using the environment.
  *(Run with: `python agent.py`)*

* **`eva.py`**
  Script to visualize and evaluate the best-performing policy.
  *(Run with: `mjpython eva.py`)*

