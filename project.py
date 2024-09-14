import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageTk
from pydicom import dcmread
from screeninfo import get_monitors
from tkinter import filedialog, Tk, Label, Frame, Button, LEFT, RIGHT
from copy import deepcopy
from multiprocessing import Process, Queue

@register_keras_serializable()
def IOU_METRIC(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())

@register_keras_serializable()
def DICE_COEFFICIENT(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + tf.keras.backend.epsilon()) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())

@register_keras_serializable()
def DICE_LOSS(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + tf.keras.backend.epsilon()) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())

@register_keras_serializable()
def BCE_DICE_LOSS(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = DICE_LOSS(y_true, y_pred)
    return bce + dice

### PATHS ###
TRAINING_IMAGES_PATH = 'Training'
TESTING_IMAGE_PATH = 'Test'

### NEURAL NETWORK PARAMETERS ###
OPTIMIZER = Adam()
LOSS = [BCE_DICE_LOSS]
METRICS = [IOU_METRIC, DICE_COEFFICIENT]
INPUT_SHAPE = [256, 256, 1]

### TRAINING DATA PARAMETERS ###
EPOCHS = 300
BATCH_SIZE = 8
SPLIT_PERCENT = 0.6
FILTER_MULTIPLIER = 32

class Fetcher:
    def __init__(self, path : str) -> None:
        self.__buffer : np.ndarray = None

        if type(path) != str:
            raise TypeError(f'Type of path should be str.')
        if not os.path.isdir(path):
            raise NotADirectoryError(f'Path should point to directory.')
        if len(os.listdir(path)) == 0:
            raise NotImplementedError(f'Directory is empty.')
        self.__path = path

    def load_dicom(self) -> None:
        tmp = []
        os.chdir(self.__path)
        files = sorted(os.listdir(), reverse=True)
        for file in files:
            try:
                full_image = dcmread(file).pixel_array
                image = full_image[147:403, 127:383] 
                image[image < 500] = 0 
                image[215:257, :] = 0
                tmp.append(image)
            except:
                print(f'File {file} could not be loaded. Skipping it.')
                pass
        self.__buffer = np.array(tmp, dtype=np.float32)
        os.chdir('..')

    def generate_masks(self) -> None:
        if os.path.exists('masks.npy'):
            self.__buffer = np.load('masks.npy')
            return
        else:
            pass

        def fill_holes(image : np.ndarray) -> np.ndarray:
            for itr in range(3):
                x, y = np.where(image == 0)
                for i,j in zip(x,y):
                    tmp = image[i-1:i+2, j-1:j+2]
                    if np.sum(tmp) > 4:
                        image[i][j] = 1
            return image
        
        masks = []
        for image in self.__buffer:
            max = np.max(image)
            image[image < max/2] = 0
            image[image > 0] = 1
            filled_mask = fill_holes(image)
            masks.append(filled_mask)
        self.__buffer = np.array(masks, dtype=np.uint8)
        np.save('masks.npy', self.__buffer)

    def get_buffer(self) -> np.ndarray:
        buffer = deepcopy(self.__buffer)
        return buffer

class Interface:
    def __init__(self, queue : Queue):
        self.__queue : Queue = queue
        self.__window : Tk = Tk()
        self.__picture_array : np.ndarray = None
        self.__picture : Label = None
        self.__buttons_frame : Frame = None
        self.__left_button : Button = None
        self.__right_button : Button = None
        self.__text : Label = None
        self.__pixel_area : float = None

        self.__window.title('Bone recognizer')
        monitor = get_monitors()[0]
        position_x = (monitor.width // 2) - (512 // 2)
        position_y = (monitor.height // 2) - (576 // 2)
        self.__window.geometry(f'{512}x{612}+{position_x}+{position_y}')

        to_display = Image.open('./main_image.jpg')
        default_picture = ImageTk.PhotoImage(to_display)
        self.__picture = Label(self.__window, image=default_picture)
        self.__picture.image = default_picture 
        self.__picture.pack()
        
        self.__text = Label(self.__window, text="", font=('Arial', 14))
        self.__text.place(x=0, y=512, width=512, height=40)
        self.__text.pack(pady=10)

        self.__buttons_frame = Frame(self.__window)
        self.__buttons_frame.pack(pady=12)
        self.__left_button = Button(self.__buttons_frame, text="Load picture", command=self._select_and_display)
        self.__left_button.pack(side=LEFT, padx=16)
        self.__right_button = Button(self.__buttons_frame, text="Analyse picture", command=self._analyse_and_display)
        self.__right_button.pack(side=RIGHT, padx=16)
        
    def _select_and_display(self) -> None:
        file_path = filedialog.askopenfilename()
        if file_path:
            self.__display(file_path)

    #                           : np.ndarray | str
    def __display(self, picture                   ) -> None:
        try:
            if isinstance(picture, str):
                tmp = dcmread(picture)
                full_image = tmp.pixel_array
                self.__pixel_area = tmp.PixelSpacing[0] * tmp.PixelSpacing[1]
                image = full_image[147:403, 127:383] 
                image[image < 500] = 0 
                image[215:257, :] = 0
                self.__picture_array = image
            elif isinstance(picture, np.ndarray):
                self.__picture_array = picture
            else:
                raise TypeError

            plt.figure(figsize=(5.12, 5.12))
            plt.imshow(self.__picture_array, cmap='gray')
            if os.path.exists('displayed.jpg'):
                os.remove('displayed.jpg')
            plt.imsave('displayed.jpg', self.__picture_array, cmap='gray')
            plt.close()
            
            image = Image.open('./displayed.jpg')
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            wrapped_picture = ImageTk.PhotoImage(image)
            self.__picture.config(image=wrapped_picture)
            self.__picture.image = wrapped_picture
        except TypeError:
            print(f'Type error!\n Try again loading .dcm picture or pass np.ndarray')
            self._select_and_display()

    def _analyse_and_display(self) -> None:
        if type(self.__picture_array) != np.ndarray:
            self._select_and_display()
        image = self.__picture_array
        self.__queue.put(image)
        result = self.__queue.get()
        prediction = np.squeeze(result)
        area = np.count_nonzero(prediction)
        self.__display(prediction * image)
        self.__text.config(text=f"Area of bone is {(area * self.__pixel_area):.2f} milimeters squared")
        
    def run(self) -> None:
        self.__window.mainloop()

class Model:
    def __init__(self, queue : Queue) -> None:
        self.__model = None
        self.__queue : Queue = queue
        self.__is_ready : bool = False
        self.__buffer : np.ndarray = None

    def __build_model(self) -> None:
        inputs = layers.Input(shape=INPUT_SHAPE)

        c1 = layers.Conv2D(FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        p1 = layers.Dropout(0.1)(p1)

        c2 = layers.Conv2D(2 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(2 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        p2 = layers.Dropout(0.1)(p2)

        c3 = layers.Conv2D(4 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(4 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        p3 = layers.Dropout(0.2)(p3)

        c4 = layers.Conv2D(6 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(6 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        p4 = layers.Dropout(0.2)(p4)

        c5 = layers.Conv2D(8 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(p4)
        c5 = layers.Conv2D(8 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c5)
        
        u6 = layers.Conv2DTranspose(6 * FILTER_MULTIPLIER, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(6 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(6 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c6)

        u7 = layers.Conv2DTranspose(4 * FILTER_MULTIPLIER, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(4 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(4 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c7)

        u8 = layers.Conv2DTranspose(2 * FILTER_MULTIPLIER, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(2 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(u8)
        c8 = layers.Conv2D(2 * FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c8)

        u9 = layers.Conv2DTranspose(FILTER_MULTIPLIER, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(u9)
        c9 = layers.Conv2D(FILTER_MULTIPLIER, (3, 3), activation='relu', padding='same')(c9)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.__model = models.Model(inputs=[inputs], outputs=[outputs])
        self.__model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    def __training(self) -> None:
        fetcher = Fetcher(TRAINING_IMAGES_PATH)
        fetcher.load_dicom()
        X = fetcher.get_buffer()

        fetcher = Fetcher(TRAINING_IMAGES_PATH)
        fetcher.load_dicom()
        fetcher.generate_masks()
        y = fetcher.get_buffer()

        split_index = int(len(X) * SPLIT_PERCENT)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        X_train = np.expand_dims(X_train, axis=-1)  
        y_train = np.expand_dims(y_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)    
        y_val = np.expand_dims(y_val, axis=-1)

        checkpoint = ModelCheckpoint('training.keras', save_best_only=True, monitor='val_loss', mode='min')

        self.__model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[checkpoint])
        self.__model.evaluate(X_val, y_val)
        self.__is_ready = True

    def __predict(self, picture : np.ndarray) -> None:
        to_predict = np.expand_dims(picture, axis=-1)
        self.__buffer = self.__model.predict(to_predict)

    def run(self) -> None:
        if os.path.exists('training.keras'):
            self.__model = load_model('training.keras')
            self.__is_ready = True
        else:
            self.__build_model()
            self.__training()
        while True:
            if self.__is_ready:
                tmp = self.__queue.get()
                reshaped_picture = tmp.reshape(1, 256, 256, 1)
                self.__predict(reshaped_picture)
                self.__queue.put(self.__buffer)

class Main:
    def __init__(self) -> None:
        self.__queue = Queue()
        self.__interface = Interface(self.__queue)
        self.__model = Model(self.__queue)
        self.__model_process = Process(target=self.__model.run)

    def run(self) -> None:
        self.__model_process.start()
        self.__interface.run()
        self.__model_process.join()

if __name__ == '__main__':
    main = Main()
    main.run()