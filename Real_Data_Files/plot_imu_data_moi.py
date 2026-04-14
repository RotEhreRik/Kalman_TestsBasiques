import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
 
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df
 
def animate(path, interval=1):
    df = load_csv(path)
    t  = df['time_s'].values
    ax = df['ax'].values;  ay = df['ay'].values;  az = df['az'].values
    gx = df['gx'].values;  gy = df['gy'].values;  gz = df['gz'].values
    mx = df['mx'].values;  my = df['my'].values;  mz = df['mz'].values
    n  = len(t)
 
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    # fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle('IMU raw sensors', color='#ccccee', fontsize=13)
 
    COLORS = ('#f472b6', '#7c73e6', '#4ade80')
    # BG, GRID, TICK, TEXT = '#1a1a2e', '#333355', '#aaaacc', '#ccccee'
    BG, GRID, TICK, TEXT = '#FFFFFF', '#333355', '#aaaacc', '#ccccee'

    panels = [
        (ax1, 'acceleration (m/s²)', "ax''", "ay''", "az''"),
        (ax2, 'angular rate (°/s)',   'gx',   'gy',   'gz'),
        (ax3, 'magnetic field (µT)',  'mx',   'my',   'mz'),
    ]
 
    for panel, ylabel, lx, ly, lz in panels:
        panel.set_facecolor(BG)
        panel.tick_params(colors=TICK)
        panel.xaxis.label.set_color(TICK)
        panel.yaxis.label.set_color(TICK)
        panel.set_ylabel(ylabel, fontsize=9)
        for spine in panel.spines.values():
            spine.set_edgecolor(GRID)
        panel.grid(True, color=GRID, linewidth=0.5)
 
    ax3.set_xlabel('time (s)', color=TICK)
 
    def ylims(a, b, c):
        lo  = min(a.min(), b.min(), c.min())
        hi  = max(a.max(), b.max(), c.max())
        pad = 0.1 * max(hi - lo, 1e-3)
        return lo - pad, hi + pad
 
    ax1.set_ylim(*ylims(ax, ay, az))
    ax2.set_ylim(*ylims(gx, gy, gz))
    ax3.set_ylim(*ylims(mx, my, mz))
 
    # Fixed x axis over full time range — no scrolling needed
    for panel in (ax1, ax2, ax3):
        panel.set_xlim(t[0], t[-1])
 
    def make_lines(panel, lx, ly, lz):
        lnx, = panel.plot([], [], color=COLORS[0], linewidth=1.4, label=lx)
        lny, = panel.plot([], [], color=COLORS[1], linewidth=1.4, label=ly)
        lnz, = panel.plot([], [], color=COLORS[2], linewidth=1.4, label=lz)
        panel.legend(facecolor=BG, edgecolor=GRID, labelcolor=TICK,
                     fontsize=8, loc='upper right')
        return lnx, lny, lnz
 
    lax, lay, laz = make_lines(ax1, "ax''", "ay''", "az''")
    lgx, lgy, lgz = make_lines(ax2, 'gx', 'gy', 'gz')
    lmx, lmy, lmz = make_lines(ax3, 'mx', 'my', 'mz')
 
    time_text = ax1.text(0.01, 0.95, '', transform=ax1.transAxes,
                         color=TEXT, fontsize=9)
 
    all_lines = (lax, lay, laz, lgx, lgy, lgz, lmx, lmy, lmz)
 
    def init():
        for ln in all_lines:
            ln.set_data([], [])
        time_text.set_text('')
        return (*all_lines, time_text)
 
    def update(frame):
        ts = t[:frame+1]
        lax.set_data(ts, ax[:frame+1])
        lay.set_data(ts, ay[:frame+1])
        laz.set_data(ts, az[:frame+1])
        lgx.set_data(ts, gx[:frame+1])
        lgy.set_data(ts, gy[:frame+1])
        lgz.set_data(ts, gz[:frame+1])
        lmx.set_data(ts, mx[:frame+1])
        lmy.set_data(ts, my[:frame+1])
        lmz.set_data(ts, mz[:frame+1])
        time_text.set_text(f't = {t[frame]:.3f} s   frame {frame+1}/{n}')
        return (*all_lines, time_text)
 
    ani = animation.FuncAnimation(
        fig, update, frames=n, init_func=init,
        interval=interval, blit=True,
        cache_frame_data=False, repeat=False
    )
 
    plt.tight_layout()
    plt.show()
    return ani
 
if __name__ == '__main__':

    filename = "imu_data_move_y.csv"
    # filename = "imu_data_static.csv"
    # filename = "imu_data_rotation.csv"

    parser = argparse.ArgumentParser(description='Animate all IMU sensors from CSV')
    parser.add_argument('csv', nargs='?', default=filename,
                        help=f'Path to CSV file (default: {filename})')
    parser.add_argument('--interval', type=int, default=1,
                        help='Frame interval in ms (default: 1 = fastest)')
    args = parser.parse_args()
 
    animate(args.csv, interval=args.interval)