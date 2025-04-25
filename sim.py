from dm_control import suite
from dm_control import viewer

# Load Acrobot swing-up task
env = suite.load(domain_name="acrobot", task_name="swingup")

# Launch the interactive viewer
viewer.launch(env)
action = viewer.get_keyboard_action()

