# gui.py
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from PIL import Image
import customtkinter
import tksheet
from tkinter import filedialog
from model import Model
from model2 import Model1
from image_processing import loadimgred, loadimgblack
import torch
import numpy as np
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    # This function opens a dialog box to get a number input from the user
    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    # This function changes the appearance mode of the custom tkinter widgets
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    # This function changes the scaling factor of the custom tkinter widgets
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    # This function imports an image file and displays it on the canvas
    def import_image(self):
        # Open a file dialog to choose an image
        filepath = filedialog.askopenfilename(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')])
        # Load the image and display it on the canvas
        img = Image.open(filepath).convert('RGB')
        graphic_image = customtkinter.CTkImage(img, size=(400, 247))
        label = customtkinter.CTkLabel(self.img_frame, image=graphic_image, text='')
        label.place(x=0, y=0)
        self.update_plot(filepath)

    def update_plot(self, filepath):
        if self.radio_var.get() == 0:
            originaly, originalx, fitted = loadimgred(filepath)
        elif self.radio_var.get() == 1:
            originaly, originalx, fitted = loadimgblack(filepath)
        else:
            return

        plt.clf()  # clear the plot
        x = range(0, 1000)
        plt.plot(originalx, originaly, 'b', label='Original Data', lw=0.9)  # plot a blue straight line
        plt.plot(x, fitted, 'r', label='Fitted Data', lw=0.9)  # plot a red fitted line
        xtick_labels = ['0', '4000', '3500', '3000', '2500', '2000', '1500', '1000', '500', '0']
        plt.gca().set_xticklabels(xtick_labels)
        plt.xlabel('Wavenumber / cm⁻¹', fontsize=8)
        plt.ylabel('Transmittance', fontsize=8)
        self.plot_canvas_canvas.draw()  # redraw the plot canvas

        if self.radio_var1.get() == 0:
            self.run_ai(fitted)
        elif self.radio_var1.get() == 1:
            self.run_ai_all(fitted)

    # This function runs the first AI model on the fitted data and displays the results in a table
    def run_ai(self, fitted):
        func = ['alcohol', 'carboxylic acid', 'ester', 'amide', 'aldehyde', 'ketone', 'Alkane', 'amine', 'alkyl halide',
                'alkene', 'methyl', 'aromatic 6 MB ring']
        model = Model1()
        model.load_state_dict(torch.load('models/model3.pt'))
        model.eval()
        fitted = torch.from_numpy(np.array(fitted)).float().view(-1, 1, 1000)
        output = model(fitted)[0].tolist()
        pairs = [(func[i], output[i]) for i in range(len(output)) if output[i] > 0]
        self.table.set_sheet_data(data=pairs)

    # This function runs all the AI models on the fitted data and displays the results in a table
    def run_ai_all(self, fitted):
        func = ['alcohol', 'carboxylic acid', 'ester', 'amide', 'aldehyde', 'ketone', 'Alkane', 'amine', 'alkyl halide',
                'alkene', 'methyl', 'aromatic 6 MB ring']
        models = [Model() for _ in range(1, 13)]
        final = [float(model(torch.from_numpy(np.array(fitted)).float().view(-1, 1, 1000))[0][0]) for model in models]
        pairs = [[func[i], final[i]] for i in range(12) if final[i] > 0]
        self.table.set_sheet_data(data=pairs)

    def __init__(self):
        super().__init__()

        # configure window
        self.title("Functional group prediction.py")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Image Selection",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.import_image,
                                                        text='Import Image')
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                               values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))
        self.radio_var = tkinter.IntVar(value=0)
        self.sidebar_frame = customtkinter.CTkLabel(master=self.sidebar_frame, text="Spectra line Colour")
        self.sidebar_frame.grid(row=2, column=0, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.sidebar_frame, variable=self.radio_var, value=0,
                                                           text="Red")
        self.radio_button_1.grid(row=4, column=0, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.sidebar_frame, variable=self.radio_var, value=1,
                                                           text="Black")
        self.radio_button_2.grid(row=3, column=0, pady=10, padx=20, sticky="n")

        self.radio_var1 = tkinter.IntVar(value=0)
        self.sidebar_frame = customtkinter.CTkLabel(master=self.sidebar_frame, text="Model")
        self.sidebar_frame.grid(row=5, column=0, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.sidebar_frame, variable=self.radio_var1, value=0,
                                                           text="One for all")
        self.radio_button_1.grid(row=6, column=0, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.sidebar_frame, variable=self.radio_var1, value=1,
                                                           text="Many for all")
        self.radio_button_2.grid(row=7, column=0, pady=10, padx=20, sticky="n")

        self.img_frame = customtkinter.CTkFrame(master=self, width=250, border_width=0, height=260)
        self.img_frame.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.img_frame1 = customtkinter.CTkFrame(master=self, width=250, border_width=0)
        self.img_frame1.grid(row=1, column=1, padx=(20, 0), pady=(20, 20), sticky="nsew")

        fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=100)
        self.plot_canvas_canvas = FigureCanvasTkAgg(fig, master=self.img_frame1)
        self.plot_canvas_canvas.get_tk_widget().pack()
        ax.set_xlabel('Wavenumber / cm⁻¹', fontsize=8)
        ax.set_ylabel('Transmittance', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        plt.tight_layout()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.xaxis.set_tick_params(width=1)
        ax.yaxis.set_tick_params(width=1)

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, height=10, fg_color=('#D3D3D3', '#D3D3D3'))
        # self.textbox.configure(4)
        self.textbox.grid(row=1, column=4, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # create scrollable frame
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="Functional groups", height=260)
        self.scrollable_frame.grid(row=1, column=3, padx=(20, 0), pady=(20, 20))
        self.scrollable_frame.grid_columnconfigure(1, weight=0)
        self.scrollable_frame_switches = []
        func = ['alcohol', 'carboxylic acid', 'ester', 'amide', 'aldehyde', 'ketone', 'Alkane', 'amine', 'alkyl halide',
                'alkene', 'methyl', 'aromatic 6 MB ring']
        for i in range(len(func)):
            switch = customtkinter.CTkSwitch(master=self.scrollable_frame, text=func[i])
            switch.grid(row=i, column=0, padx=0, pady=(0, 0))
            self.scrollable_frame_switches.append(switch)

        self.table = tksheet.Sheet(self)
        self.table.grid(row=0, column=3, columnspan=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.table.headers(['Functional group', 'Probability'])

        for i in range(12):
            self.scrollable_frame_switches[i].select()
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.textbox.insert("0.0",
                            "INFORMATION:\n\n" + "This is a pre-pre-alpha version   0.01 which is most likely prone   to bugs and is not very accurate.\n\n")

    # This function opens a dialog box to get a number input from the user
    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    # This function changes the appearance mode of the custom tkinter widgets
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    # This function changes the scaling factor of the custom tkinter widgets
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    # This function imports an image file and displays it on the canvas
    def import_image(self):
        # Open a file dialog to choose an image
        filepath = filedialog.askopenfilename(filetypes=[('Image Files', '*.png;*.jpg;*.jpeg')])
        # Load the image and display it on the canvas
        img = Image.open(filepath).convert('RGB')
        graphic_image = customtkinter.CTkImage(img, size=(400, 247))
        label = customtkinter.CTkLabel(self.img_frame, image=graphic_image, text='')
        label.place(x=0, y=0)
        self.update_plot(filepath)

    def update_plot(self, filepath):
        if self.radio_var.get() == 0:
            originaly, originalx, fitted = loadimgred(filepath)
        elif self.radio_var.get() == 1:
            originaly, originalx, fitted = loadimgblack(filepath)
        else:
            return

        plt.clf()  # clear the plot
        x = range(0, 1000)
        plt.plot(originalx, originaly, 'b', label='Original Data', lw=0.9)  # plot a blue straight line
        plt.plot(x, fitted, 'r', label='Fitted Data', lw=0.9)  # plot a red fitted line
        xtick_labels = ['0', '4000', '3500', '3000', '2500', '2000', '1500', '1000', '500', '0']
        plt.gca().set_xticklabels(xtick_labels)
        plt.xlabel('Wavenumber / cm⁻¹', fontsize=8)
        plt.ylabel('Transmittance', fontsize=8)
        self.plot_canvas_canvas.draw()  # redraw the plot canvas

        if self.radio_var1.get() == 0:
            self.run_ai(fitted)
        elif self.radio_var1.get() == 1:
            self.run_ai_all(fitted)

    # This function runs the first AI model on the fitted data and displays the results in a table
    def run_ai(self, fitted):
        func = ['alcohol', 'carboxylic acid', 'ester', 'amide', 'aldehyde', 'ketone', 'Alkane', 'amine', 'alkyl halide',
            'alkene', 'methyl', 'aromatic 6 MB ring']
        model = Model1()
        model.load_state_dict(torch.load('models/model3.pt'))
        model.eval()
        fitted = torch.from_numpy(np.array(fitted)).float().view(-1, 1, 1000)
        output = model(fitted)[0].tolist()
        pairs = [(func[i], output[i]) for i in range(len(output)) if output[i] > 0]
        self.table.set_sheet_data(data=pairs)

    # This function runs all the AI models on the fitted data and displays the results in a table
    def run_ai_all(self, fitted):
        func = ['alcohol', 'carboxylic acid', 'ester', 'amide', 'aldehyde', 'ketone', 'Alkane', 'amine', 'alkyl halide',
            'alkene', 'methyl', 'aromatic 6 MB ring']
        models = [Model() for _ in range(1, 13)]
        final = [float(model(torch.from_numpy(np.array(fitted)).float().view(-1, 1, 1000))[0][0]) for model in models]
        pairs = [[func[i], final[i]] for i in range(12) if final[i] > 0]
        self.table.set_sheet_data(data=pairs)

