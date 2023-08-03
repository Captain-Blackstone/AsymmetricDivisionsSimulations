from interactive_mode import non_blocking_pause
import matplotlib.pyplot as plt
import numpy as np
import atexit


class Drawer:
    """
    Class that draws the plots in the interactive mode.
    """
    def __init__(self, simulation_thread):
        self.simulation = simulation_thread
        self.update_time = 500  # number of steps between figure updates
        self.resolution = 150  # number of steps between data collection events
        self.plot_how_many = 1000  # number of points present on the plot at each time point
        self.timeline = []
        plt.ion()
        self.fig, self.ax = plt.subplots(3, 2)
        self.fig.canvas.manager.full_screen_toggle()
        plt.show(block=False)
        atexit.register(plt.close)
        line_data_dicts = [
            {"ax_num": [0, 0], "color": "blue", "alpha": 1, "label": "Population size",
             "update_function": lambda: self.simulation.matrix.sum()},
            {"ax_num": [1, 0], "color": "green", "alpha": 1, "label": "Mean damage",
             "update_function":
                 lambda: (self.simulation.rhos * self.simulation.matrix/self.simulation.matrix.sum()).sum() if self.simulation.matrix.sum() > 0 else 0},
            {"ax_num": [2, 0], "color": "orange", "alpha": 1, "label": "Nutrient concentration",
             "update_function":
                 lambda: self.simulation.ksi
             },

        ]
        dist_data_dicts = [
            {"ax_num": [0, 1], "color": "blue", "alpha": 1, "label": "Nutrient dist",
             "update_function": lambda: self.simulation.matrix.sum(axis=1)},
            {"ax_num": [1, 1], "color": "green", "alpha": 1, "label": "Damage dist",
             "update_function":
                 lambda: np.trim_zeros(self.simulation.matrix.sum(axis=0))},
            # {"ax_num": [2, 1], "color": "green", "alpha": 1, "label": "Rho dist",
            #  "update_function":
            #      lambda: self.damage_update_func()}
        ]
        matrix_data_dicts = [
            {"ax_num": [2, 1], "label": "Matrix",
             "update_function":
                 lambda: np.log(self.simulation.matrix)}
        ]



        self.plots = [LinePlot(self,
                               self.plot_how_many,
                               self.ax[*data_dict["ax_num"]],
                               data_dict["color"],
                               data_dict["alpha"],
                               data_dict["update_function"],
                               data_dict.get("label")) for data_dict in line_data_dicts]

        self.plots += [DistPlot(self,
                                self.ax[*data_dict["ax_num"]],
                                data_dict["color"],
                                data_dict["alpha"],
                                data_dict["update_function"],
                                data_dict.get("label")) for data_dict in dist_data_dicts]
        self.plots += [MatrixPlot(self,
                                 self.ax[*data_dict["ax_num"]],
                                data_dict["update_function"],
                                data_dict.get("label")) for data_dict in matrix_data_dicts]

    def damage_update_func(self):
        rhos, counts = self.simulation.rhos.flatten(), self.simulation.matrix.flatten()
        rhos, counts = rhos[counts > 0], counts[counts > 0]
        rhos_unique = sorted(np.unique(rhos))
        res = [counts[rhos == rho].sum() for rho in rhos_unique]
        return dict(x=rhos_unique, y=res)

    def draw_step(self, step_number: int, time_step_duration: float) -> None:
        """
        Update all the Plots of the Drawer.
        Update the data only each resolution time_step,
        Update the plot only each update_time time_step.
        :param step_number:
        :param time_step_duration:
        :return:
        """
        # Collect data each self.resolution steps
        if step_number % self.resolution == 0:
            for plot in self.plots:
                plot.collect_data(time_step_duration)
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
                 ax: plt.Axes,
                 color: str,
                 alpha: float,
                 update_function, ylabel=None):
        self.drawer = drawer
        self.ax, self.color = ax, color
        self.update_function = update_function
        self.xdata, self.ydata = [], []
        if ylabel is not None:
            self.ax.set_ylabel(ylabel, fontsize=10)
        self.alpha = alpha
        self.layer, = self.ax.plot(self.xdata, self.ydata, color=self.color, alpha=self.alpha, marker="*", linestyle='--')

    def collect_data(self, time_step_duration: float) -> None:
        pass

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


class LinePlot(Plot):
    def __init__(self,
                 drawer: Drawer,
                 plot_how_many: int,
                 ax: plt.Axes,
                 color: str,
                 alpha: float,
                 update_function, ylabel=None):
        super().__init__(drawer, ax, color, alpha, update_function, ylabel)
        self.plot_how_many = plot_how_many
        self.ydata = []
        self.xdata = list(np.arange(len(self.ydata)))

    def collect_data(self, time_step_duration: float) -> None:
        """
           Update ydata list.
           ydata is updated by calling update_function of the object.
           :param time_step_duration:
           :return:
           """
        if self.xdata:
            self.xdata.append(self.drawer.simulation.time)
        else:
            self.xdata.append(self.drawer.simulation.time)
        self.xdata = self.xdata[-self.plot_how_many:]
        self.ydata.append(self.update_function())
        self.ydata = self.ydata[-self.plot_how_many:]


class DistPlot(Plot):
    def __init__(self,
                 drawer: Drawer,
                 ax: plt.Axes,
                 color: str,
                 alpha: str,
                 update_function, ylabel=None):
        super().__init__(drawer, ax, color, alpha, update_function, ylabel)
        self.ydata = []
        self.xdata = list(np.arange(len(self.ydata)))

        self.layer, = self.ax.plot(self.xdata, self.ydata, color=self.color, alpha=self.alpha)

    def collect_data(self, time_step_duration: float) -> None:
        """
        put the current xdata and ydata on the plot
        :return:
        """
        res = self.update_function()
        if isinstance(res, dict):
            self.ydata = res["y"]
            self.xdata = res["x"]
        else:
            self.ydata = res
            self.xdata = list(np.arange(len(self.ydata)))


class MatrixPlot:
    def __init__(self,
                 drawer: Drawer,
                 ax: plt.Axes,
                 update_function, ylabel=None):
        self.drawer = drawer
        self.ax = ax
        self.update_function = update_function
        self.ydata = []
        if ylabel is not None:
            self.ax.set_ylabel(ylabel, fontsize=10)

    def collect_data(self, time_step_duration):
        self.ydata = self.update_function()

    def update_data(self):
        """
        put the current xdata and ydata on the plot
        :return:
        """
        self.ax.clear()
        self.ax.imshow(self.ydata, interpolation="none", aspect="auto")

    def update_plot(self):
        pass
