import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
df1 = pd.read_csv('PPO_Acrobot_current.csv')
df2 = pd.read_csv('PPO_effort_current.csv')

# Apply moving average with window size 1024
window_size = 5
df1['Smoothed'] = df1['Value'].rolling(window=window_size).mean()
df2['Smoothed'] = df2['Value'].rolling(window=window_size).mean()
print(df1.shape)
# Plot

plt.style.use('dark_background')
plt.figure(figsize=(10, 6))

plt.plot(df1['Step'], df1['Smoothed'], label='PPO (smoothed)')
plt.plot(df2['Step'], df2['Smoothed'], label='PPO with effort (smoothed)')

# Add labels and legend
plt.xlabel('Time Steps')
plt.ylabel('Reward')
plt.title('Reward of PPO vs PPO with effort')
plt.legend()
plt.grid(True)
plt.show()


# # Load the CSV files
# df1 = pd.read_csv('PPO_Acrobot_current.csv')
# df2 = pd.read_csv('PPO_effort_current.csv')

# # Print first few rows to see what the data looks like
# print(df1.head())
# print(df2.head())

# # Plot example: assume both files have 'x' and 'y' columns
# plt.plot(df1['Step'], df1['Value'], label='PPO', color='cyan', linewidth=2)
# plt.plot(df2['Step'], df2['Value'], label='PPO with effort', color='magenta', linewidth=2)

# # Add labels and legend
# plt.xlabel('Time Steps')
# plt.ylabel('Reward')
# plt.title('Reward of PPO vs PPO with effort')
# plt.legend()
# plt.grid(True)
# plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# # Load CSV files
# df1 = pd.read_csv('PPO_Acrobot_current_epi.csv')
# df2 = pd.read_csv('PPO_effort_currrent_epi.csv')

# # Apply moving average with window size 1024
# window_size = 5
# df1['Smoothed'] = df1['Value'].rolling(window=window_size).mean()
# df2['Smoothed'] = df2['Value'].rolling(window=window_size).mean()
# print(df1.shape)
# # Plot
# # plt.style.use('dark_background')
# plt.figure(figsize=(10, 6))

# plt.plot(df1['Step'], df1['Smoothed'], label='PPO (smoothed)')
# plt.plot(df2['Step'], df2['Smoothed'], label='PPO with effort (smoothed)')

# plt.xlabel('Time Steps')
# plt.ylabel('Mean Episode Length')
# plt.title('Mean Episode Length: PPO vs PPO with Effort')
# plt.legend()
# plt.grid(True)
# plt.show()
