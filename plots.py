import pandas as pd
import matplotlib.pyplot as plt

# # Load the CSV files
# df1 = pd.read_csv('PPO.csv')
# df2 = pd.read_csv('PPO_effort.csv')

# # Print first few rows to see what the data looks like
# print(df1.head())
# print(df2.head())

# # Plot example: assume both files have 'x' and 'y' columns
# plt.plot(df1['Step'], df1['Value'], label='PPO')
# plt.plot(df2['Step'], df2['Value'], label='PPO with effort')

# # Add labels and legend
# plt.xlabel('Time Steps')
# plt.ylabel('Explained Variance')
# plt.title('Explained Variance of PPO vs PPO with effort')
# plt.legend()
# plt.grid(True)
# plt.show()

# Load the CSV files
df1 = pd.read_csv('PPO_entropy_loss.csv')
df2 = pd.read_csv('PPO_effort_entropy_loss.csv')

# Print first few rows to see what the data looks like
print(df1.head())
print(df2.head())

# Plot example: assume both files have 'x' and 'y' columns
plt.plot(df1['Step'], df1['Value'], label='PPO')
plt.plot(df2['Step'], df2['Value'], label='PPO with effort')

# Add labels and legend
plt.xlabel('Time Steps')
plt.ylabel('Entropy Loss')
plt.title('Entropy Loss of PPO vs PPO with effort')
plt.legend()
plt.grid(True)
plt.show()