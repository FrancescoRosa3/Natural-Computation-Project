import matplotlib.pyplot as plt

def plot_history(history_lap_time, history_speed, history_damage):
    for lap in history_lap_time:
        print(lap)
        plt.figure(figsize=(1, 2))
        plt.subplot(121)

        minutes, seconds = divmod(history_lap_time[lap], 60)
        dsecond = int((seconds%1)*100)
        title = f"Lap {lap}, lap time {int(minutes)}:{int(seconds)}:{dsecond}"
        plt.title(title)

        plt.plot(range(len(history_speed[lap])), history_speed[lap])
        plt.xlabel("Step")
        plt.ylabel("Speed")

        plt.subplot(122)
        plt.plot(range(len(history_damage[lap])), history_damage[lap])
        plt.xlabel("Step")
        plt.ylabel("Damage")

        plt.show()