import matplotlib.pyplot as plt

def plot_rainfall_temperature(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['Date'], df['Rainfall'], label='Rainfall (mm)', color='blue', alpha=0.6)
    plt.plot(df['Date'], df['Temperature'], label='Temperature (°C)', color='red', alpha=0.6)
    plt.title('Rainfall and Temperature Over Time')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_avg_rainfall(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['Date'], df['AvgRainfall'], label='7-Day Avg Rainfall', color='green')
    plt.axhline(y=5, color='r', linestyle='--', label='Drought Threshold')
    plt.title('7-Day Rolling Average Rainfall')
    plt.xlabel('Date')
    plt.ylabel('Avg Rainfall (mm)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drought_comparison(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['Date'], df['Drought'], label='Actual Drought', linestyle='-', color='black')
    plt.plot(df['Date'], df['Predicted_Drought'], label='Predicted Drought', linestyle='--', color='orange')
    plt.title('Actual vs Predicted Drought Conditions')
    plt.xlabel('Date')
    plt.ylabel('Drought (1 = Yes, 0 = No)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
