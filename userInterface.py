import sqlite3
import threading
import time
import tkinter as tk
from tkinter import INSERT
from tkinter import Menu
from tkinter import messagebox as msg
from tkinter import ttk
from tkinter import END
from tkinter import scrolledtext
import cv2
from PIL import ImageTk, Image
from tensorflow.keras.models import model_from_json
import operator


class ThreadRunner:
    """
    ThreadRunner runs the process to get the current frame from the video source
    and it uses the trained model to predict characters from hands on the video
    source. The process had to be separated from the UI thread to avoid freezes.

    There are three methods to get frame, roi frame and predicted text
    get_frame() -> gets the current frame
    get_roi_frame() -> gets the current roi frame
    get_prediction() -> gets the current predicted text
    """
    def __init__(self, video_source=0, width=None, height=None, fps=None):

        self.video_source = video_source
        self.width = width
        self.height = height
        self.fps = fps
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("ThreadRunner Unable to open video source", video_source)

        # Set the height according to the width
        if self.width and not self.height:
            ratio = self.width / self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * ratio)
        # Get video source width and height if width and height are not set
        if not self.width:
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # convert float to int
        if not self.height:
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # convert float to int
        if not self.fps:
            self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))  # convert float to int

        # set rectangle points relatively with the source width and height
        self.recX1 = int(self.width * 0.5)
        self.recY1 = int(self.width * 0.10)
        self.recX2 = int(self.width * 0.95)
        self.recY2 = int(self.width * 0.50)
        # default required values at start
        self.ret = False
        self.frame = None
        self.roi = None
        self.roi_ret = False
        self.roi_frame = None
        self.running = True
        self.model = None
        self.predicted_text = ""
        self.categories = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
                           5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                           10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
                           15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'K',
                           20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P',
                           25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V',
                           30: 'Y', 31: 'Z'}
        self.prediction_result = None
        self.current_prediction = None
        self.prediction_count = 0
        self.process_timer = 0
        # start thread
        self.thread = threading.Thread(target=self.process)
        self.thread.start()

    def process(self):
        # get the model from json and load the weights
        with open("model-bw.json", "r") as mj:
            self.model = model_from_json(mj.read())
        self.model.load_weights("model-bw.h5")
        # start video streaming and predicting process
        while self.running:
            ret, frame = self.vid.read()

            if ret:
                # process image
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, flipCode=1)
                cv2.rectangle(frame, (self.recX1, self.recY1), (self.recX2, self.recY2), (255, 255, 0), 1)
                self.roi = frame[self.recY1:self.recY2, self.recX1:self.recX2]
                self.roi = cv2.resize(self.roi, (64, 64))
                self.roi = cv2.cvtColor(self.roi, cv2.COLOR_BGR2GRAY)
                roi_ret, roi_frame = cv2.threshold(self.roi, 120, 255, cv2.THRESH_BINARY)
                self.prediction_result = self.model.predict(roi_frame.reshape(1, 64, 64, 1))
                prediction = {'0': self.prediction_result[0][0],
                              '1': self.prediction_result[0][1],
                              '2': self.prediction_result[0][2],
                              '3': self.prediction_result[0][3],
                              '4': self.prediction_result[0][4],
                              '5': self.prediction_result[0][5],
                              '6': self.prediction_result[0][6],
                              '7': self.prediction_result[0][7],
                              '8': self.prediction_result[0][8],
                              '9': self.prediction_result[0][9],
                              'A': self.prediction_result[0][10],
                              'B': self.prediction_result[0][11],
                              'C': self.prediction_result[0][12],
                              'D': self.prediction_result[0][13],
                              'E': self.prediction_result[0][14],
                              'F': self.prediction_result[0][15],
                              'G': self.prediction_result[0][16],
                              'H': self.prediction_result[0][17],
                              'I': self.prediction_result[0][18],
                              'K': self.prediction_result[0][19],
                              'L': self.prediction_result[0][20],
                              'M': self.prediction_result[0][21],
                              'N': self.prediction_result[0][22],
                              'O': self.prediction_result[0][23],
                              'P': self.prediction_result[0][24],
                              'R': self.prediction_result[0][25],
                              'S': self.prediction_result[0][26],
                              'T': self.prediction_result[0][27],
                              'U': self.prediction_result[0][28],
                              'V': self.prediction_result[0][29],
                              'Y': self.prediction_result[0][30],
                              'Z': self.prediction_result[0][31]
                              }
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret_thresh, thresh = cv2.threshold(imgray, 127, 255, 0)
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) < 150:
                    if self.prediction_count == 0:
                        self.current_prediction = prediction[0][0]
                    if self.current_prediction == prediction[0][0]:
                        self.prediction_count += 1
                    else:
                        self.prediction_count = 0
                    if self.prediction_count < 10:
                        self.predicted_text = "Hold still..."
                        self.process_timer = 0
                    else:
                        self.predicted_text = prediction[0][0]
                        self.process_timer += 1
                    if self.process_timer > 20:
                        self.predicted_text += str(prediction[0][0])
                        self.process_timer = 0
                    print(self.predicted_text)
                else:
                    self.predicted_text = "Can't detect!"
            else:
                print('ThreadRunner stream end:', self.video_source)
                # TODO: reopen stream
                self.running = False
                break

            # assign new frame
            self.ret = ret
            self.frame = frame
            self.roi_ret = roi_ret
            self.roi_frame = roi_frame

            # sleep for next frame
            time.sleep(1 / self.fps)

    # get current frame
    def get_frame(self):
        return self.ret, self.frame

    # get current roi frame
    def get_roi_frame(self):
        return self.roi_ret, self.roi_frame

    # get the predicted text
    def get_prediction(self):
        return self.predicted_text

    # Release the video source when the object is destroyed
    def __del__(self):
        # stop thread
        if self.running:
            self.running = False
            self.thread.join()

        # release stream
        if self.vid.isOpened():
            self.vid.release()


# Camera Frame object creates the camera frame and it updates
# the frame by getting the current frame from ThreadRunner object
class CameraFrame(tk.Frame):
    def __init__(self, container, video_source=None, width=None, height=None):
        super().__init__(container)
        self.container = container

        self.video_source = video_source
        if not self.video_source:
            self.video_source = 0
        self.vid = ThreadRunner(self.video_source, width, height)

        self.canvas = tk.Canvas(self, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        self.delay = int(1000 / self.vid.fps)

        print('CameraFrame source:', self.video_source)
        print('CameraFrame fps:', self.vid.fps, 'delay:', self.delay)

        self.image = None

        self.running = True
        self.update_frame()

    def update_frame(self):
        # widgets in tkinter already have method `update()` so I have to use different name -

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        if self.running:
            self.container.after(self.delay, self.update_frame)


# RoiFrame object creates the frame for roi frame
# updated by ThreadRunner object
class RoiFrame(tk.Frame):
    def __init__(self, container, video_capture, fps):
        super().__init__(container)
        self.container = container

        self.video_capture = video_capture
        self.fps = fps
        self.canvas = tk.Canvas(self, width=64, height=64)
        self.canvas.pack()
        self.roi_frame = None
        self.delay = int(1000 / self.fps)
        self.running = True
        self.isActive = True
        self.image = None

        self.update_roi_frame()

    def update_roi_frame(self):
        ret, self.roi_frame = self.video_capture.get_roi_frame()

        if ret:
            self.image = Image.fromarray(self.roi_frame)
            self.photo = ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        else:
            self.isActive = False

        if self.running:
            self.container.after(self.delay, self.update_roi_frame)


# PredictedText object creates a tkinter label object and updates
# the text attribute of label according to the prediction
class PredictedText(tk.Label):
    def __init__(self, container, video_capture, fps):
        super().__init__(container)
        self.container = container
        self.video_capture = video_capture
        self.fps = fps
        self.predicting = True
        self.label_text = ""

        self.label = tk.Label(self, text="")
        self.label.pack()
        self.delay = int(25000 / self.fps)
        self.update_label()

    def update_label(self):
        self.label.config(text=self.video_capture.get_prediction())
        if self.predicting:
            self.container.after(self.delay, self.update_label)


# PredictionText object creates a tkinter text object and
# concatenates the predicted text predicted by the model
class PredictionText(tk.Text):
    def __init__(self, container, video_capture, fps):
        super().__init__(container)
        self.container = container
        self.video_capture = video_capture
        self.fps = fps
        self.predicting = True
        self.text_entry = ""
        self.predicted_text = ""
        self.text = tk.Text(self, width=30, height=3, wrap=tk.WORD)
        self.text.pack()
        self.delay = int(25000 / self.fps)
        self.update_text()

    def update_text(self):
        self.predicted_text = self.video_capture.get_prediction()
        if self.predicted_text != "Hold still..." and self.predicted_text != "Can't detect!":
            self.text_entry = self.predicted_text
            self.text.insert(END, self.text_entry)
        if self.predicting:
            self.container.after(self.delay, self.update_text)


class LoginWindow:
    def __init__(self):
        self.running = False
        self.win = tk.Tk()
        self.win.title("Login")
        self.db = Database("database.db")
        self.widgets()

    def account_check(self):
        if self.username.get() == "" or self.password.get() == "":
            msg.showerror("Warning!", "Fields can not be empty!")
        else:
            userlist = self.db.queryFunction(
                f"SELECT * FROM Users WHERE username = '{self.username.get()}' and password = '{self.password.get()}'")
            if len(userlist) > 0:
                self.win.destroy()
                MainWindow(self.username)
            else:
                msg.showerror("Wrong user!", "User information did not match!")

    def exit_window(self):
        self.win.quit()
        self.win.destroy()
        exit()

    def register_user(self):
        RegisterUserWindow()

    def widgets(self):
        self.containerFrame = ttk.LabelFrame(self.win, text="Welcome to Turkish Sign Language Translator")
        self.containerFrame.grid(column=0, row=0, padx=10, pady=10, sticky=tk.NSEW)

        self.username_label = ttk.Label(self.containerFrame, text="Username:")
        self.username_label.grid(column=0, row=1, padx=10, pady=5, sticky=tk.NSEW)
        self.username = tk.StringVar()
        self.username_entry = ttk.Entry(self.containerFrame, textvariable=self.username)
        self.username_entry.grid(column=1, row=1, padx=10, pady=5, sticky=tk.NSEW)

        self.password_label = ttk.Label(self.containerFrame, text="Password:")
        self.password_label.grid(column=0, row=2, padx=10, pady=5, sticky=tk.NSEW)
        self.password = tk.StringVar()
        self.password_entry = ttk.Entry(self.containerFrame, show="*", textvariable=self.password)
        self.password_entry.grid(column=1, row=2, padx=10, pady=5, sticky=tk.NSEW)

        self.ok_button = ttk.Button(self.containerFrame, text="OK", command=self.account_check)
        self.ok_button.grid(column=0, row=3, padx=10, pady=5, sticky=tk.NSEW)
        self.cancel_button = ttk.Button(self.containerFrame, text="Cancel", command=self.exit_window)
        self.cancel_button.grid(column=1, row=3, padx=10, pady=5, sticky=tk.NSEW)

        self.username_label = ttk.Label(self.containerFrame, text="If you don't have ->")
        self.username_label.grid(column=0, row=4, padx=10, pady=5, sticky=tk.NSEW)

        self.register_button = ttk.Button(self.containerFrame, text="Register", command=self.register_user)
        self.register_button.grid(column=1, row=4, padx=10, pady=5, sticky=tk.NSEW)

        self.password_entry.bind("<Return>", lambda e: self.account_check())
        self.username_entry.bind("<Return>", lambda e: self.account_check())
        self.password_entry.bind("<Escape>", lambda e: self.exit_window())
        self.username_entry.bind("<Escape>", lambda e: self.exit_window())


class RegisterUserWindow:
    def __init__(self):
        self.win2 = tk.Tk()
        self.win2.title("Register")
        self.db = Database("database.db")
        self.widgets()

    def create_user(self):
        if self.username.get() == "" or self.password.get() == "":
            msg.showerror("Warning!", "Fields can not be empty!")
        else:
            userlistcheck = self.db.queryFunction(
                f"SELECT * FROM Users WHERE username = '{self.username.get()}' and password = '{self.password.get()}'")
            if (len(userlistcheck) > 0):
                msg.showerror("The user cannot be created.", "User exist")
            else:
                self.db.createUser(self.username.get(), self.password.get(), self.firstName.get(), self.lastName.get(),
                                   self.age.get(), self.gender.get())
                self.win2.destroy()

    def exit_window(self):
        self.win2.destroy()

    def widgets(self):
        self.containerFrame2 = ttk.LabelFrame(self.win2, text="Register a new User")
        self.containerFrame2.grid(column=0, row=0, padx=10, pady=10, sticky=tk.NSEW)

        self.username_label = ttk.Label(self.containerFrame2, text="Username:")
        self.username_label.grid(column=0, row=0, padx=10, pady=5, sticky=tk.NSEW)
        self.username = tk.StringVar(self.containerFrame2)
        self.username_entry = ttk.Entry(self.containerFrame2, textvariable=self.username)
        self.username_entry.grid(column=1, row=0, padx=10, pady=5, sticky=tk.NSEW)

        self.password_label = ttk.Label(self.containerFrame2, text="Password:")
        self.password_label.grid(column=0, row=1, padx=10, pady=5, sticky=tk.NSEW)
        self.password = tk.StringVar(self.containerFrame2)
        self.password_entry = ttk.Entry(self.containerFrame2, show="*", textvariable=self.password)
        self.password_entry.grid(column=1, row=1, padx=10, pady=5, sticky=tk.NSEW)

        self.firstName_label = ttk.Label(self.containerFrame2, text="First Name:")
        self.firstName_label.grid(column=0, row=2, padx=10, pady=5, sticky=tk.NSEW)
        self.firstName = tk.StringVar(self.containerFrame2)
        self.firstName_entry = ttk.Entry(self.containerFrame2, textvariable=self.firstName)
        self.firstName_entry.grid(column=1, row=2, padx=10, pady=5, sticky=tk.NSEW)

        self.lastName_label = ttk.Label(self.containerFrame2, text="Last Name:")
        self.lastName_label.grid(column=0, row=3, padx=10, pady=5, sticky=tk.NSEW)
        self.lastName = tk.StringVar(self.containerFrame2)
        self.lastName_entry = ttk.Entry(self.containerFrame2, textvariable=self.lastName)
        self.lastName_entry.grid(column=1, row=3, padx=10, pady=5, sticky=tk.NSEW)

        self.age_label = ttk.Label(self.containerFrame2, text="Age:")
        self.age_label.grid(column=0, row=4, padx=10, pady=5, sticky=tk.NSEW)
        self.age = tk.StringVar(self.containerFrame2)
        self.age_entry = ttk.Entry(self.containerFrame2, textvariable=self.age)
        self.age_entry.grid(column=1, row=4, padx=10, pady=5, sticky=tk.NSEW)

        self.gender_label = ttk.Label(self.containerFrame2, text="Gender:")
        self.gender_label.grid(column=0, row=5, padx=10, pady=5, sticky=tk.NSEW)
        self.gender = tk.StringVar(self.containerFrame2)
        self.gender_combobox = ttk.Combobox(self.containerFrame2, width=12, textvariable=self.gender)
        self.gender_combobox['values'] = ('Male', 'Female', 'Other')
        self.gender_combobox.grid(column=1, row=5, padx=10, pady=5, sticky=tk.NSEW)
        self.gender_combobox.current(0)

        self.ok_button = ttk.Button(self.containerFrame2, text="OK", command=self.create_user)
        self.ok_button.grid(column=0, row=6, padx=10, pady=5, sticky=tk.NSEW)
        self.cancel_button = ttk.Button(self.containerFrame2, text="Cancel", command=self.exit_window)
        self.cancel_button.grid(column=1, row=6, padx=10, pady=5, sticky=tk.NSEW)

        self.password_entry.bind("<Return>", lambda e: self.create_user())
        self.username_entry.bind("<Return>", lambda e: self.create_user())
        self.password_entry.bind("<Escape>", lambda e: self.exit_window())
        self.username_entry.bind("<Escape>", lambda e: self.exit_window())


class MainWindow:
    def __init__(self, username):
        self.win3 = tk.Tk()
        self.win3.title("Turkish Sign Language Translator")
        self.db = Database("database.db")
        self.cameraFrame = None
        self.signImage = None
        self.predictionLabel = None
        self.translationResult = None
        self.username = username
        self.widgets()

    def exit_window(self):
        self.cameraFrame.vid.running = False
        self.win3.quit()
        self.win3.destroy()
        quit()

    def about_window(self):
        msg.showinfo("About us", "This project created by: \nCanberk Enes SEN - 1609998, Muhammet Musa CAM - 1728774 and Furkan GULLE - 1728824 ")

    def on_closing(self):
        self.cameraFrame.vid.running = False
        self.win3.quit()
        self.win3.destroy()
        exit()

    def widgets(self):
        menu_bar = Menu(self.win3)
        self.win3.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.exit_window)
        file_menu.add_separator()

        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.about_window)

        self.win3.columnconfigure(0, weight=0)
        self.win3.columnconfigure(1, weight=1)
        self.win3.rowconfigure(0, weight=0)

        self.containerFrame3 = ttk.LabelFrame(self.win3, text="Info")
        self.containerFrame3.grid(column=0, row=0, padx=10, pady=10, sticky=tk.NSEW)
        self.welcome_user = ttk.Label(self.containerFrame3, text=f"Hi {self.username.get()}!")
        self.welcome_user.grid(column=0, row=0, padx=5, pady=5, sticky=tk.NSEW)
        self.takenote_label = ttk.Label(self.containerFrame3, text="Take a note ✎")
        self.takenote_label.grid(column=0, row=1, padx=5, pady=(10, 5), sticky=tk.NSEW)
        self.take_note = scrolledtext.ScrolledText(self.containerFrame3, width=15, height=20)
        self.take_note.grid(column=0, row=2, padx=5, pady=5, sticky=tk.NS)
        self.exit_button = ttk.Button(self.containerFrame3, text="Exit", command=self.exit_window)
        self.exit_button.grid(column=0, row=3, padx=5, pady=5, sticky=tk.NSEW)

        self.containerFrame4 = ttk.LabelFrame(self.win3, text="Translation")
        self.containerFrame4.grid(column=1, row=0, padx=10, pady=10, sticky=tk.NSEW)

        self.cameraImageLabel = ttk.Label(self.containerFrame4, text="Camera")
        self.cameraImageLabel.grid(column=1, row=0, padx=20, pady=5, sticky=tk.EW)
        self.cameraFrame = CameraFrame(self.containerFrame4, video_source=0, width=400)
        self.cameraFrame.grid(column=1, row=1, padx=20, pady=5, sticky=tk.EW)

        self.signImageLabel = ttk.Label(self.containerFrame4, text="Sign Image")
        self.signImageLabel.grid(column=0, row=0, padx=5, pady=5, sticky=tk.EW)
        self.signImage = RoiFrame(self.containerFrame4, self.cameraFrame.vid, self.cameraFrame.vid.fps)
        self.signImage.grid(column=0, row=1, padx=5, pady=5, sticky=tk.EW)

        self.predictionLabel = PredictedText(self.containerFrame4, self.cameraFrame.vid, self.cameraFrame.vid.fps)
        self.predictionLabel.grid(column=0, row=2, padx=5, pady=5, columnspan=3)

        self.translationResultLabel = ttk.Label(self.containerFrame4, text="Result")
        self.translationResultLabel.grid(column=0, row=2, padx=5, pady=5, sticky=tk.EW)
        self.translationResult = PredictionText(self.containerFrame4, self.cameraFrame.vid, self.cameraFrame.vid.fps)
        self.translationResult.grid(column=0, row=3, padx=5, pady=5, columnspan=3)

        self.win3.bind("<Escape>", lambda e: self.exit_window())


class Database:
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()
        self.cur.execute("""
                CREATE TABLE IF NOT EXISTS Users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL,
                    firstName TEXT,
                    lastName TEXT,
                    age INTEGER,
                    gender TEXT 
                );   
                """)
        self.conn.commit()

    def query_func2(self, query):
        data = self.cur.execute(query)
        rows = data.fetchone()
        return rows

    def queryFunction(self, query):
        data = self.cur.execute(query)
        rows = data.fetchall()
        return rows

    def createUser(self, username, password, firstName, lastName, age, gender):
        self.cur.execute("INSERT INTO Users VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                         (username, password, firstName, lastName, age, gender))
        self.conn.commit()

    def removeUser(self, id):
        self.cur.execute("DELETE FROM Users WHERE id=?", (id,))
        self.conn.commit()

    def updateUser(self, username, password, firstName, lastName, age, gender, id):
        self.cur.execute(
            "UPDATE Users SET username = ?, password = ?, firstName = ?, lastName = ?, age = ?, gender = ? WHERE id = ?",
            (username, password, firstName, lastName, age, gender, id))
        self.conn.commit()


if __name__ == "__main__":
    app = LoginWindow()
    app.win.mainloop()
