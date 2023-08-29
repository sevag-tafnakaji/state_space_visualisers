import tkinter as tk
from tkinter import ttk, filedialog
from typing import Tuple
import time
import queue
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from src.model.linear_state_space_model import StateSpace
from src.simulation.simulator import Simulator


CHOICES = ["A matrix", "B matrix", "C matrix", "process noise matrix - N",
           "process noise covariance - Q", "measurement noise covariance - R",
           "sampling interval", "initial state - x_0", "initial state error covariance - P_0",
           "LQR state weight matrix - Q", "LQR input weight matrix - R",
           "Reference value - r", "matrix transformation for state with reference values - M",
           "simulation timestep - delta t", "Simulation time - t"]


class ControlVisualisation:
    """
    Class to generate interface with data entry for model, saving and loading pre-defined model.
    """
    def __init__(self, root: tk.Tk):
        self.root: tk.Tk = root
        self.root.title("Data Input App")
        self.plot_fig, self.plot_ax = None, None
        self.data = {}  # To store the temporarily inputted data
        self.sim_output = queue.Queue(-1)
        self.delta_t = 0
        self.simulation_time = 3
        self.create_interface()
        self.create_empty_plot()

    def create_empty_plot(self):
        """
        Create empty matplotlib plot to use as placeholder in GUI.
        """
        # Create an empty figure and subplot for initial display
        self.plot_fig, self.plot_ax = plt.subplots()
        self.plot_ax.set_title("Simulation Result")
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=self.plot_frame)
        self.plot_canvas.get_tk_widget().pack()

    def create_interface(self):
        """
        Create GUI of visualiser.
        """
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Save Data", command=self.save_data_to_yaml)
        self.file_menu.add_command(label="Load Data", command=self.load_data_from_yaml)

        label = tk.Label(self.root, text="Select data type:")
        label.grid(row=0, column=0, pady=10)

        self.data_type_var = tk.StringVar()
        data_type_choices = CHOICES
        data_type_dropdown = ttk.Combobox(self.root, textvariable=self.data_type_var,
                                          values=data_type_choices)
        data_type_dropdown.grid(row=0, column=1, padx=10, pady=10)

        data_entry_label = tk.Label(self.root, text="Enter data:")
        data_entry_label.grid(row=1, column=0, pady=10)

        self.data_entry_text = tk.Text(self.root, height=5, width=40)  # Adjust the width as needed
        self.data_entry_text.grid(row=1, column=1, padx=10, pady=10)

        save_button = tk.Button(self.root, text="Save Data", command=self.save_data)
        save_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.start_button = tk.Button(self.root, text="Start Simulation", command=self.start_sim)
        self.start_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.grid(row=0, column=2, rowspan=4, padx=10, pady=10)

    def save_data(self):
        """
        Save data from textbox into dict attribute and reset text box and drop-down menu.
        """
        data_type = self.data_type_var.get()
        data_value = self.data_entry_text.get("1.0", "end-1c")  # Get data from Text widget

        if data_type and data_value:
            self.data[data_type] = data_value
            self.data_type_var.set("")  # Clear the data type selection
            self.data_entry_text.delete("1.0", "end")  # Clear the data entry field

    def save_data_to_yaml(self):
        """
        Save all model data into YAML file.
        """
        file_path = filedialog.asksaveasfilename(defaultextension=".yaml",
                                                 filetypes=[("YAML Files", "*.yaml")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(self.data, yaml_file)
            print(f"Data saved to {file_path}")

    def load_data_from_yaml(self):
        """
        Load pre-defined model from YAML file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as yaml_file:
                    self.data = yaml.safe_load(yaml_file)
                print(f"Data loaded from {file_path}")
            except FileNotFoundError:
                print("Data file not found.")

    def extract_data(self, raw_data: str) -> np.ndarray:
        """
        Convert string from textbox into usable data for model and simulation.

        Args:
            raw_data (str): Raw data from text entry box.

        Returns:
            np.ndarray: matrix containing data extracted from string.
        """

        no_brackets = raw_data[1:-1]
        extracted_data = []
        if ";" in no_brackets and "," in no_brackets:
            rows = no_brackets.split(";")
            for row in rows:
                new_row = []
                values = row.split(",")
                for value in values:
                    new_row.append(float(value.strip()))
                extracted_data.append(new_row)
            extracted_data = np.array(extracted_data)
            return extracted_data
        elif "," not in no_brackets and ";" in no_brackets:
            rows = no_brackets.split(";")
            for row in rows:
                extracted_data.append([float(row.strip())])
            extracted_data = np.array(extracted_data)
            return extracted_data
        elif ";" not in no_brackets and "," in no_brackets:
            cols = no_brackets.split(",")
            row = []
            for col in cols:
                row.append(float(col.strip()))
            extracted_data.append(row)
            extracted_data = np.array(extracted_data)
            return extracted_data
        return np.array([[float(raw_data)]])

    def define_model(self) -> Tuple[StateSpace, Simulator]:
        """
        Take all string data from entry box and convert to usable matrices,
        then define model and simulator.

        Returns:
            Tuple[StateSpace, Simulator]: model and simulator using provided data.
        """
        A = self.extract_data(self.data["A matrix"])
        B = self.extract_data(self.data["B matrix"])
        C = self.extract_data(self.data["C matrix"])
        N = self.extract_data(self.data["process noise matrix - N"])
        Q = self.extract_data(self.data["process noise covariance - Q"])
        R = self.extract_data(self.data["measurement noise covariance - R"])
        dt = float(self.data["sampling interval"])
        model = StateSpace(A, B, C, N, Q, R, dt)
        x_0 = self.extract_data(self.data["initial state - x_0"])
        P_0 = self.extract_data(self.data["initial state error covariance - P_0"])
        Q = self.extract_data(self.data["LQR state weight matrix - Q"])
        R = self.extract_data(self.data["LQR input weight matrix - R"])
        r = self.extract_data(self.data["Reference value - r"])
        M = self.extract_data(
            self.data["matrix transformation for state with reference values - M"])
        delta_t = float(self.data["simulation timestep - delta t"])
        if "Simulation time - t" in self.data:
            self.simulation_time = float(self.data["Simulation time - t"])

        simulator = Simulator(model, x_0, P_0, Q, R, r, M, delta_t)
        self.delta_t = delta_t
        return simulator

    def get_outputs(self, simulator: Simulator):
        """
        Simulate model, estimator, and controller capability of following reference value.

        Args:
            simulator (Simulator): Simulator class used to update simulation
        """
        i = 0

        delta_t = simulator.delta_t
        previous_estimate = simulator.kalman.previous_estimate
        previous_state = simulator.kalman.previous_estimate

        while i < self.simulation_time:
            start_time = time.time()
            state, output, estimate, _, _ = simulator.update(previous_state, previous_estimate)
            previous_estimate = estimate
            previous_state = state
            i += delta_t
            i = round(i, 2)
            if int(i) == i:
                print(i)
            self.sim_output.put(output)

            remaining_time = delta_t - (time.time() - start_time)

            if remaining_time > 0:
                time.sleep(remaining_time)

    def start_sim(self):
        """
        Checks if all data is input correctly and begin simulation. If data not provided, show which
        data is missing.
        """
        if len(self.data) == len(CHOICES) - 1 and "Simulation time - t" not in self.data:
            self.data["Simulation time - t"] = self.simulation_time
        if len(self.data) == len(CHOICES):
            simulator = self.define_model()
            self.plot_ax.clear()
            self.get_outputs(simulator)
            self.update_plot()  # Call a function to update the plot
        else:
            missing_data = [x for x in CHOICES if x not in self.data.keys()]
            print("Please enter data for all choices.")
            print(f"Missing: {missing_data}")

    def update_plot(self):
        """
        Use simulation result data to update plot.
        """
        self.plot_ax.set_title("Simulation Result")

        N = int(self.simulation_time / self.delta_t)
        data = np.zeros((2, N))
        i = 0
        while not self.sim_output.empty():
            data[:, i:i+1] = self.sim_output.get()
            i += 1
        data = np.array(data)
        t = np.linspace(0, self.simulation_time, N)
        m = data.shape[0]
        for i in range(m):
            self.plot_ax.plot(t, data[i, :])
            self.plot_ax.grid(True)
        self.plot_canvas.draw()
