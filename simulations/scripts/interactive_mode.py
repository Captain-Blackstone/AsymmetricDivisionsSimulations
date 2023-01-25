import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import atexit
from itertools import chain

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
        self.update_time = 400  # number of steps between figure updates
        self.resolution = 40  # number of steps between data collection events
        self.plot_how_many = 250  # number of points present on the plot at each time point
        plt.ion()
        n_plots = 1
        damage_plot = False
        if self.simulation_thread.chemostat.cells[0].damage_accumulation_linear_component > 0 or \
                self.simulation_thread.chemostat.cells[0].damage_accumulation_exponential_component > 0 or \
                len(set([cell.damage for cell in self.simulation_thread.chemostat.cells])) > 1:
            damage_plot = True

        asymmetry_plot = False
        if self.simulation_thread.chemostat.cells[0].asymmetry_mutation_rate > 0 and \
                self.simulation_thread.chemostat.cells[0].asymmetry_mutation_step > 0 or \
                len(set([cell.asymmetry for cell in self.simulation_thread.chemostat.cells])) > 1:
            asymmetry_plot = True
        repair_plot = False
        if self.simulation_thread.chemostat.cells[0].repair_mutation_rate > 0 and \
                self.simulation_thread.chemostat.cells[0].repair_mutation_step > 0 or \
                len(set([cell.damage_repair_intensity for cell in self.simulation_thread.chemostat.cells])) > 1:
            repair_plot = True

        self.fig, self.ax = plt.subplots(n_plots + damage_plot + asymmetry_plot + repair_plot, 1)
        if n_plots + damage_plot + asymmetry_plot + repair_plot == 1:
            self.ax = [self.ax]
        plt.show(block=False)
        atexit.register(plt.close)
        line_data_dicts = [
            {"ax_num": 0, "color": "blue", "alpha": 1, "label": "Population size",
             "update_function": lambda: self.simulation_thread.chemostat.N},
            {"ax_num": 1, "color": "grey", "alpha": 1, "label": "Mean damage",
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).mean()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 1, "color": "grey", "alpha": 0.5,
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).max()
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": 1, "color": "grey", "alpha": 0.5,
             "update_function":
                 lambda: np.array([cell.damage for cell in self.simulation_thread.chemostat.cells]).min()
                 if self.simulation_thread.chemostat.N else 0}]
        frequency_data_dicts = [
            {"ax_num": int(n_plots + damage_plot), "color": "green", "label": "Asymmetry", "max": 1,
             "update_function":
                 lambda: np.array([round(cell.asymmetry, 5) for cell in self.simulation_thread.chemostat.cells])
                 if self.simulation_thread.chemostat.N else 0},
            {"ax_num": int(n_plots + damage_plot + asymmetry_plot), "color": "red", "label": "Repair",
             "max": self.simulation_thread.chemostat.cells[0].damage_accumulation_linear_component,
             "update_function":
                 lambda: np.array([round(cell.damage_repair_intensity, 24)
                                   for cell in self.simulation_thread.chemostat.cells])
                 if self.simulation_thread.chemostat.N else 0}
        ]
        if not damage_plot:
            line_data_dicts = line_data_dicts[:1]
        if not repair_plot:
            frequency_data_dicts.pop(1)
        if not asymmetry_plot:
            frequency_data_dicts.pop(0)
        self.plots = [LinePlot(self,
                           self.plot_how_many,
                           self.ax[data_dict["ax_num"]],
                           data_dict["color"],
                           data_dict["alpha"],
                           data_dict["update_function"], data_dict.get("label")) for data_dict in line_data_dicts]
        self.plots.extend([
            FrequencyPlot(self,
                     self.plot_how_many,
                     self.ax[data_dict["ax_num"]],
                     data_dict["color"],
                     data_dict["label"],
                     data_dict["max"],
                     data_dict["update_function"]) for data_dict in frequency_data_dicts
        ])
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
                 update_function, ylabel=None):
        self.drawer, self.plot_how_many = drawer, plot_how_many
        self.ax, self.color = ax, color
        self.update_function = update_function
        self.xdata, self.ydata = [], []
        if ylabel is not None:
            self.ax.set_ylabel(ylabel, fontsize=10)

    def collect_data(self, step_num: int):
        """
        Update ydata list.
        ydata is updated by calling update_function of the object.
        :param step_num:
        :return:
        """
        self.xdata.append(step_num)
        self.xdata = self.xdata[-self.plot_how_many:]
        self.ydata.append(self.update_function())
        self.ydata = self.ydata[-self.plot_how_many:]

    def update_data(self):
        pass

    def update_plot(self):
        """
        rescale the axis
        :return:
        """
        self.ax.relim()
        self.ax.autoscale_view(tight=True)


class LinePlot(Plot):
    def __init__(self,
                 drawer: Drawer,
                 plot_how_many: int,
                 ax: plt.Axes,
                 color: str,
                 alpha: str,
                 update_function, ylabel=None):
        super().__init__(drawer, plot_how_many, ax, color, update_function, ylabel)
        self.alpha = alpha
        self.layer, = self.ax.plot(self.xdata, self.ydata, color=self.color, alpha=self.alpha)


    def update_data(self):
        """
        put the current xdata and ydata on the plot
        :return:
        """
        self.layer.set_ydata(self.ydata)
        self.layer.set_xdata(self.xdata)


class FrequencyPlot(Plot):
    def __init__(self, drawer: Drawer,
                 plot_how_many: int,
                 ax: plt.Axes,
                 color: str, label: str, max: float,
                 update_function):
        super().__init__(drawer, plot_how_many, ax, color, update_function)
        self.drawer, self.plot_how_many = drawer, plot_how_many
        self.ax, self.color, self.label, self.max = ax, color, label, max
        self.update_function = update_function
        self.xdata, self.ydata = [], []

    def update_data(self):
        """
        put the current xdata and ydata on the plot
        :return:
        """
        self.ax.clear()
        bottom = np.zeros_like(self.xdata).astype(float)
        for batch in sorted(list(set(list(chain(*self.ydata)))), reverse=True):
            rect_heights = [(el == batch).sum() / len(el) for el in self.ydata]
            self.ax.fill_between(self.xdata, bottom, bottom + rect_heights, color=self.color,
                                 alpha=min((self.max/20 + batch)/(1.05*self.max), 1),
                                 label=batch)
            self.ax.plot(self.xdata, bottom + rect_heights, color="black", alpha=0.5)

            bottom += np.array(rect_heights)
        self.ax.set_ylabel(self.label)
        self.ax.set_ylim(0, 1)
        self.ax.legend(loc="upper left")
