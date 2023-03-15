'''
A collection of functions for visulaizing solutions to the 
neural field model with synaptic depression.
'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from tqdm import tqdm
from plotting_styles import U_style, Q_style, solution_styles, threshold_style

def make_animation(file_name,
                   ts,
                   xs,
                   us_list,
                   theta,
                   *,
                   us_labels=None,
                   x_window=(-15, 60),
                   y_window=(-0.1, 1.1),
                   U_style=U_style,
                   Q_style=Q_style,
                   sol_styles=solution_styles,
                   frames=200,
                   fps=24,
                   animation_interval=200,
                   title=None):

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    line_us = [ax.plot(xs, us[0][0], **U_style, **sol_style)[0]
               for us, sol_style in zip(us_list, sol_styles)]
    line_qs = [ax.plot(xs, us[0][1], **Q_style, **sol_style)[0]
               for us, sol_style in zip(us_list, sol_styles)]

    ax.plot(xs, theta + 0*xs, **threshold_style, label='$\\theta$')

    ax.plot([], [], 'k', **U_style, label='$u$')
    ax.plot([], [], 'k', **Q_style, label='$q$')
    if us_labels is not None:
        assert len(us_labels) == len(us_list)
        for style, label in zip(sol_styles, us_labels):
            ax.plot([], [], 'o', **style, label=label)

    ax.set_xlim(*x_window)
    ax.set_ylim(*y_window)
    ax.legend(loc='right')

    if title is not None:
        ax.set_title(title)

    frames_array = np.arange(0, len(ts), len(ts)//frames)

    def animate(i):
        for us, line_u, line_q in zip(us_list, line_us, line_qs):
            line_u.set_ydata(us[i][0])
            line_q.set_ydata(us[i][1])
        return line_us

    # Init only required for blitting to give a clean slate.
    def init():
        for us, line_u, line_q in zip(us_list, line_us, line_qs):
            line_u.set_ydata(us[0])
        return line_us

    print(f'frames = {frames}')

    if file_name is not None:
        anim = animation.FuncAnimation(fig,
                                       animate,
                                       frames=tqdm(frames_array),
                                       # tqdm(frames_array),
                                       init_func=init,
                                       interval=animation_interval,
                                       blit=True)

        # anim.save(file_name, writer='imagemagick', fps=fps)
        # anim.save(file_name, writer='ffmpeg', fps=fps)
        anim.save(file_name, writer='ffmpeg', fps=fps, extra_args=['-loglevel', 'debug'])
        plt.close()

    else:
        plt.ion()
        for i in frames_array:
            animate(i)
            plt.pause(1/fps)
