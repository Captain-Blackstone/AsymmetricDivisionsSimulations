from interactive_mode import non_blocking_pause
import matplotlib.pyplot as plt
import numpy as np
import atexit
from scipy.signal import argrelmin, argrelmax
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import copy

class Drawer:
    """
    Class that draws the plots in the interactive mode.
    """
    def __init__(self, simulation_thread):
        self.simulation = simulation_thread
        self.update_time = 3000  # number of steps between figure updates
        self.resolution = 1000  # number of steps between data collection events
        self.plot_how_many = 100  # number of points present on the plot at each time point
        self.timeline = []
        # plt.ion()
        self.fig, self.ax = plt.subplots(3, 2, figsize=(15, 10))
        # self.fig.canvas.manager.full_screen_toggle()
        # plt.show(block=False)
        atexit.register(plt.close)
        line_data_dicts = [
            {"ax_num": [0, 0], "color": "blue", "alpha": 1, "label": "Population size",
             "update_function": lambda: self.simulation.matrix.sum()},
            {"ax_num": [1, 0], "color": "green", "alpha": 1, "label": "Mean damage",
             "update_function":
                 lambda: (self.simulation.rhos * self.simulation.matrix/self.simulation.matrix.sum()).sum() if self.simulation.matrix.sum() > 0 else 0},
            {"ax_num": [2, 0], "color": "orange", "alpha": 1, "label": "Nutrient concentration",
             "update_function":
                 lambda: self.simulation.phi
             },

        ]
        dist_data_dicts = [
            {"ax_num": [0, 1], "color": "blue", "alpha": 1, "label": "Nutrient dist",
             "update_function": lambda: self.simulation.matrix.sum(axis=1)},
            {"ax_num": [1, 1], "color": "green", "alpha": 1, "label": "Damage dist",
             "update_function":
                 lambda: np.trim_zeros(self.simulation.history.burst_dist_history[-1]) if self.simulation.history.burst_dist_history else self.simulation.matrix.sum(axis=0)},
            # {"ax_num": [2, 1], "color": "green", "alpha": 1, "label": "Rho dist",
            #  "update_function":
            #      lambda: self.damage_update_func()}
        ]
        matrix_data_dicts = [
            {"ax_num": [2, 1], "label": "Matrix",
             "update_function":
                 lambda: self.simulation.matrix}
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
        # self.reg1, = self.ax[0][0].plot([0, 0], [0, 0], color="red")
        # self.reg2, = self.ax[0][0].plot([0, 0], [0, 0], color="red")
        self.window = Window(self)

    def run(self):
        self.window.run()

    def damage_update_func(self):
        rhos, counts = self.simulation.rhos.flatten(), self.simulation.matrix.flatten()
        rhos, counts = rhos[counts > 0], counts[counts > 0]
        rhos_unique = sorted(np.unique(rhos))
        res = [counts[rhos == rho].sum() for rho in rhos_unique]
        return dict(x=rhos_unique, y=res)


    def draw_step(self, step_number: int, time_step_duration: float) -> None:
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
            self.window.update()

    def draw_step_prev(self, step_number: int, time_step_duration: float) -> None:
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
            # population_size = self.simulation.history.population_sizes
            # time = self.simulation.history.times
            # slopes, intercepts = [], []
            # xx1s, xx2s, yy1s, yy2s = [], [], [], []
            # peaks = None
            # for xx, yy, func in zip([xx1s, xx2s], [yy1s, yy2s], [argrelmin, argrelmax]):
            #     peaks = np.array(population_size)[list(func(np.array(population_size))[0])]
            #     times = np.array(time)[list(func(np.array(population_size))[0])]
            #     if len(peaks) < 3:
            #         continue
            #     else:
            #         n_points = 3
            #         peaks = peaks[-n_points:]
            #         times = times[-n_points:]
            #         coefficients = np.polyfit(times, peaks, 1)
            #         slope, intercept = coefficients
            #         slopes.append(slope)
            #         intercepts.append(intercept)
            #         xx.append(times[0])
            #         yy.append(xx[0] * slope + intercept)
            # if len(slopes) == 2:
            #     xx1s.append((intercepts[1]-intercepts[0])/(slopes[0]-slopes[1]))
            #     xx2s.append((intercepts[1] - intercepts[0]) / (slopes[0] - slopes[1]))
            #     print("CONVERGENCE ESTIMATE", xx1s[-1]*slopes[0] + intercepts[0])
            #     yy1s.append(xx1s[-1]*slopes[0] + intercepts[0])
            #     yy2s.append(xx2s[-1]*slopes[1] + intercepts[1])
            #     self.reg1.set_xdata(xx1s)
            #     self.reg1.set_ydata(yy1s)
            #     self.reg2.set_xdata(xx2s)
            #     self.reg2.set_ydata(yy2s)


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

    # def update_data(self):
    #     """
    #     put the current xdata and ydata on the plot
    #     :return:
    #     """
    #     self.layer.set_ydata(self.ydata)
    #     self.layer.set_xdata(self.xdata)

    def update_data(self):
        self.ax.clear()
        self.ax.plot(self.xdata, self.ydata, color=self.color, alpha=self.alpha, marker="*", linestyle="--")

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
        # self.ydata = self.drawer.simulation.rhos
        # self.ydata[self.ydata >= 1] = 0

    def update_data(self):
        """
        put the current xdata and ydata on the plot
        :return:
        """
        try:
            self.ax.clear()
            # self.ax.imshow(self.ydata, interpolation="none", aspect="auto")
            import matplotlib
            cmap = "Blues"
            mtx = self.ydata[:-1, :].copy()
            mtx /= mtx.sum()
            column_numbers = []
            row_numbers = []
            values = []
            death_values = []
            for i, row in enumerate(mtx):
                for j, value in enumerate(row):
                    column_numbers.append(j)
                    row_numbers.append(mtx.shape[0] - i)
                    if np.isnan(value):
                        value = 0
                    values.append(value)
            percentiles = np.array([50, 70, 90, 95, 99, 100])
            levels = np.percentile(np.array(values), percentiles)

            cmap = matplotlib.cm.get_cmap(cmap)
            colors_mtx = cmap(percentiles / 100) * 256
            colors = []
            for r in range(colors_mtx.shape[0]):
                r = colors_mtx[r, :]
                i, j, k, _ = r
                colors.append('#%02x%02x%02x' % (round(i), round(j), round(k)))
            #     ax.tricontourf(column_numbers, row_numbers, values,
            #                    levels=list(range(mtx.min(), mtx.max()+2)),
            #                    cmap=cmap, vmin=mtx.min(), vmax=mtx.max())
            # self.ax.tricontourf(column_numbers, row_numbers, values, levels=levels, colors=colors)
            self.ax.imshow(np.flip(np.log10(self.drawer.simulation.damage_death_rate),  axis=0), aspect=7)
            self.ax.imshow(np.flip(np.log10(self.drawer.simulation.matrix), axis=0), aspect=7)
        except Exception as e:
            print(e)

    def update_plot(self):
        pass


class Window():
    def __init__(self, drawer: Drawer, width=1920, height=1080):
        self.drawer = drawer
        self.initial_simulation = copy.deepcopy(self.drawer.simulation)
        self.window = tk.Tk()
        plot_frac_width = 0.9
        self.window.title('PCD simulation')
        self.window.geometry(f"{width}x{height}")
        self.f_left = tk.Frame(master=self.window, borderwidth=10, height=height, width=width * plot_frac_width)
        self.f_left.grid(row=0, column=0)

        f_right = tk.Frame(master=self.window, borderwidth=10, height=height, width=width * (1 - plot_frac_width))
        f_right.grid(row=0, column=1)
        self.canvas = FigureCanvasTkAgg(self.drawer.fig, master=self.f_left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.run_button = tk.Button(master=f_right,
                                     command=self.run_pressed,
                                     height=2,
                                     width=10,
                                     text="Run")
        self.run_button.grid(row=0, column=0, sticky="n")
        self.pause_button = tk.Button(master=f_right,
                                      command=self.pause_pressed,
                                      height=2,
                                      width=10,
                                      text="Pause/Resume")
        self.pause_button.grid(row=0, column=1, sticky="n")

        params = self.drawer.simulation.params.keys()
        self.param_entries = dict()
        for i, param in enumerate(params):
            label = tk.Label(master=f_right, text=f"{param}: ")
            label.grid(row=i + 1, column=0)
            e = tk.Entry(master=f_right)
            e.insert(tk.END, self.drawer.simulation.params[param])
            e.grid(row=i + 1, column=1)
            b = tk.Button(master=f_right, command=lambda p=param: self.update_simulation(p),
                          height=1, width=2, text="Apply")
            b.grid(row=i + 1, column=2)
            self.param_entries[param] = e
    def run(self):
        self.window.mainloop()

    def update(self):
        self.canvas.draw()

    def update_simulation(self, param):
        try:
            val = float(self.param_entries[param].get().strip())
        except ValueError:
            print("Invalid value")
        self.drawer.simulation.params[param] = val
        self.drawer.simulation.update_death_function()
        print(f"Updating {param} = {val}")

    def run_pressed(self):
        thread = threading.Thread(target=lambda: self.drawer.simulation.run(1000000000, infinite=True))
        thread.start()

    def pause_pressed(self):
        # If the STOP button is pressed then terminate the loop
        self.drawer.simulation.pause = not self.drawer.simulation.pause

