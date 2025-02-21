import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Define matrix size ranges
m_values = [32, 64, 128, 256, 512]  # Varying number of rows
n_values = [4, 8, 16, 32]  # Varying number of columns

execution_times = {}

# Run C program for different values of m (fixed n = 4)
execution_times['m_variation'] = []
for m in m_values:
    cmd = ["mpirun", "-np", "4", "./TSQR_Q3", str(m), "4"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "Execution Time" in line:
            time_taken = float(line.split(":")[-1].strip().split()[0])
            execution_times['m_variation'].append(time_taken)

# Run C program for different values of n (fixed m = 256)
execution_times['n_variation'] = []
for n in n_values:
    cmd = ["mpirun", "-np", "4", "./TSQR_Q3", "256", str(n)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "Execution Time" in line:
            time_taken = float(line.split(":")[-1].strip().split()[0])
            execution_times['n_variation'].append(time_taken)

# Plot results
plt.figure()
plt.plot(m_values, execution_times['m_variation'], marker='o', label='Varying m, fixed n=4')
plt.xlabel('Number of Rows (m)')
plt.ylabel('Execution Time (s)')
plt.title('Scaling with Increasing m')
plt.legend()
plt.grid()
plt.savefig("MScaling.png")

plt.figure()
plt.plot(n_values, execution_times['n_variation'], marker='o', label='Varying n, fixed m=256')
plt.xlabel('Number of Columns (n)')
plt.ylabel('Execution Time (s)')
plt.title('Scaling with Increasing n')
plt.legend()
plt.grid()
plt.savefig("NScaling.png")

