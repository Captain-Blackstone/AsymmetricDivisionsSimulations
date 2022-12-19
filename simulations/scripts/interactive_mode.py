import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import atexit


def non_blocking_pause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


class Drawer:
    """
    Class that draws the plots in the interactive mode.
    """
    def __init__(self, simulation_thread):
        self.simulation_thread = simulation_thread
        self.update_time = 100  # number of steps between figure updates
        self.resolution = 10  # number of steps between data collection events
        self.plot_how_many = 1000  # number of points present on the plot at each time point
        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1)
        plt.show(block=False)
        atexit.register(plt.close)
        for i, title in enumerate(["Population size", "Mean damage", "Asymmetry"]):
            self.ax[i].set_title(title, fontsize=10)

        data_dicts = [
            {"ax_num": 0, "color": "blue", "alpha": 1,
             "update_function": lambda: self.simulation_thread.chemostat.N},
            {"ax_num": 1, "color": "green", "alpha": 1,
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).mean()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 1, "color": "green", "alpha": 0.5,
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).max()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 1, "color": "green", "alpha": 0.5,
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).min()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 2, "color": "red", "alpha": 1,
             "update_function":
                 lambda: np.array([cell.asymmetry for cell in self.simulation_thread.chemostat.cells]).mean()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 2, "color": "red", "alpha": 0.5,
             "update_function":
                 lambda: np.array([cell.asymmetry for cell in self.simulation_thread.chemostat.cells]).max()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 2, "color": "red", "alpha": 0.5,
             "update_function":
                 lambda: np.array([cell.asymmetry for cell in self.simulation_thread.chemostat.cells]).min()
                 if self.simulation_thread.chemostat.N else 0},
        ]

        self.plots = [Plot(self,
                           self.plot_how_many,
                           self.ax[data_dict["ax_num"]],
                           data_dict["color"],
                           data_dict["alpha"],
                           data_dict["update_function"]) for data_dict in data_dicts]

        plt.get_current_fig_manager().full_screen_toggle()

    def draw_step(self, step_number):
        """
        Update all the Plots of the Drawer.
        Update the data only each resolution time_step,
        Update the plot only each update_time time_step.
        :param step_number:
        :return:
        """
        # Collect data each self.resolution steps
        if step_number % self.resolution == 0:
            for plot in self.plots:
                plot.collect_data(step_number)
        # Update figure each self.update_time steps
        if step_number % self.update_time == 0:
            for plot in self.plots:
                plot.update_data()
            for plot in self.plots:
                plot.update_plot()
            # self.fig.canvas.draw() # seems like this line is not needed. I will delete it later if nothing goes wrong.
            non_blocking_pause(0.01)


class Plot:
    """
    Helper class for a Drawer class.
    A single Plot object can store and update data it needs to plot and plot it on a relevant axis.
    """
    def __init__(self,
                 drawer: Drawer,
                 plot_how_many: int,
                 ax: plt.Axes,
                 color: str,
                 alpha: str,
                 update_function):
        self.drawer, self.plot_how_many = drawer, plot_how_many
        self.ax, self.color, self.alpha = ax, color, alpha
        self.update_function = update_function
        self.xdata, self.ydata = [], []
        self.layer, = self.ax.plot(self.xdata, self.ydata, color=self.color, alpha=self.alpha)

    def collect_data(self, step_num: int):
        """
        Update xdata and ydata lists.
        xdata is updated by appending the step_num input value into the xdata list.
        ydata is updated by calling update_function of the object.
        :param step_num:
        :return:
        """
        self.xdata.append(step_num)
        self.ydata.append(self.update_function())
        self.xdata = self.xdata[-self.plot_how_many:]
        self.ydata = self.ydata[-self.plot_how_many:]

    def update_data(self):
        """
        put the current xdata and ydata on the plot
        :return:
        """
        self.layer.set_ydata(self.ydata)
        self.layer.set_xdata(self.xdata)

    def update_plot(self):
        """
        rescale the axis
        :return:
        """
        self.ax.relim()
        self.ax.autoscale_view(tight=True)