import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Create 100 synthetic real delivery times
normal_times = np.random.normal(loc=31.5, scale=6, size=97)

# Keep ordinary deliveries in a realistic range
normal_times = np.clip(normal_times, 18, 48)

# Add 3 unusually long deliveries
delivery_times = np.concatenate([normal_times, [68, 71, 75]])

data = {
    "OrderID": range(1, 101),
    "DeliveryTime(min)": delivery_times
}

# Save CSV
pd.DataFrame(data).to_csv("delivery_times.csv", index=False)

# Load CSV
df = pd.read_csv("delivery_times.csv")

# Mean and standard deviation of real data
mean_time = df["DeliveryTime(min)"].mean()
std_time = df["DeliveryTime(min)"].std()

print(
    f"Real delivery times -- Mean: {mean_time:.2f} min "
    f"Std: {std_time:.2f} min"
)

# Simulate 100 expected delivery times
simulated_times = np.random.normal(
    loc=mean_time,
    scale=std_time,
    size=100
)

print(
    f"Simulated 100 values from Normal"
    f"(mean={mean_time:.2f}, std={std_time:.2f})"
)

# Missing values
missing_values = df["DeliveryTime(min)"].isnull().sum()
print("Missing values:", missing_values)

# Percentiles
q1 = df["DeliveryTime(min)"].quantile(0.25)
median = df["DeliveryTime(min)"].quantile(0.50)
q3 = df["DeliveryTime(min)"].quantile(0.75)

print(
    f"Percentiles -- 25th: {q1:.2f} "
    f"50th: {median:.2f} "
    f"75th: {q3:.2f}"
)

# IQR method
IQR = q3 - q1
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR

print(
    f"IQR outlier bounds: "
    f"{lower_bound:.2f} - {upper_bound:.2f}"
)

outliers = df[
    (df["DeliveryTime(min)"] < lower_bound) |
    (df["DeliveryTime(min)"] > upper_bound)
]

print("Outliers detected:", len(outliers), "deliveries")
print(outliers[["OrderID", "DeliveryTime(min)"]])

percentage = (len(outliers) / len(df)) * 100
print(f"Percentage flagged as outliers: {percentage:.1f}%")

# Overlapping histogram
plt.figure(figsize=(8, 5))

plt.hist(
    df["DeliveryTime(min)"],
    bins=15,
    alpha=0.6,
    label="Real Delivery Times"
)

plt.hist(
    simulated_times,
    bins=15,
    alpha=0.6,
    label="Simulated Normal Times"
)

plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.title("Real vs Simulated Delivery Times")
plt.legend()
plt.tight_layout()
plt.savefig("delivery_comparison_histogram.png")
plt.close()

# Boxplot
plt.figure(figsize=(6, 5))
plt.boxplot(df["DeliveryTime(min)"])
plt.ylabel("Delivery Time (minutes)")
plt.title("Delivery Time Boxplot")
plt.tight_layout()
plt.savefig("delivery_boxplot.png")
plt.close()

print(
    "Saved delivery_comparison_histogram.png, "
    "delivery_boxplot.png"
)