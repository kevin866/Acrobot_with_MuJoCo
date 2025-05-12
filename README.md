

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

### Files Overview

* **`wind.py`**
  Simulates real-time wind effects.

* **`wind_re.py`**
  Records the wind simulation for later playback or analysis.

* **`acrobot_env.py`**
  Custom environment wrapper. You can modify the reward function and other environment settings here (e.g., cost for energy).

* **`agent.py`**
  Sample script to train a model using the environment.

* **`eva.py`**
  Script to visualize and evaluate the best-performing policy.


