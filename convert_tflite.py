from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
from PIL import Image
import urllib
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from pathlib import Path


img_rows, img_cols = 224, 224 #number of rows and columns to convert the images to
input_shape = (img_rows, img_cols, 3)#format to store the images (rows, columns,channels) called channels last

folder_train = './data/train'
folder_valid = './data/valid'

folder_ship_train = './data/train/ship'
folder_bike_train = './data/train/bike'
folder_ship_valid = './data/valid/ship'
folder_bike_valid = './data/valid/bike'


def load_cate_url(cate_id):
    page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + cate_id)
    
    soup = BeautifulSoup(page.content, 'html.parser')
    content = soup.getText()
    urls = content.split('\r\n')
    urls = [u.rstrip() for u in urls]
    print(len(urls))
    urls = [url for url in urls if len(url) > 0]
    print(len(urls))

    return urls

def make_data_folder():
    
    Path(folder_ship_train).mkdir(parents=True, exist_ok=True)
    Path(folder_bike_train).mkdir(parents=True, exist_ok=True)
    Path(folder_ship_valid).mkdir(parents=True, exist_ok=True)
    Path(folder_bike_valid).mkdir(parents=True, exist_ok=True)


def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image


def make_train_valid(urls, train_folder, valid_folder):
    # First images for train
    size_data = len(urls)
    num_train = int(size_data * 0.8)
    for progress in range(num_train): 
        if progress % 20 == 0:
            print(progress)

        if not urls[progress] == None:
            try:
                I = url_to_image(urls[progress])
                if (len(I.shape))==3: #check if the image has width, length and channels
                    save_path = train_folder + '/img' + str(progress) + '.jpg' #create a name of each image
                    cv2.imwrite(save_path,I)

            except:
                None
    
    # Next images for valid
    num_valid = size_data - num_train
    for progress in range(num_valid):
        if progress % 20 == 0:
            print(progress)

        if not urls[progress] == None:
            try:
                I = url_to_image(urls[num_train + progress]) #get images that are different from the ones used for training
                if (len(I.shape))==3: #check if the image has width, length and channels
                    save_path = valid_folder + '/img' + str(progress) + '.jpg' #create a name of each image
                    cv2.imwrite(save_path,I)

            except:
                None



def load_cate_image():
    bicyle_id = 'n02834778'
    ship_id = 'n04194289'

    make_data_folder()

    # bike_urls = load_cate_url(bicyle_id)
    ship_urls = load_cate_url(ship_id)

    # make_train_valid(bike_urls, folder_bike_train, folder_bike_valid)
    make_train_valid(ship_urls, folder_ship_train, folder_ship_valid)

    


def build_loader():

    train_datagen  = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
        
    train_generator = train_datagen.flow_from_directory(
            folder_train,
            target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
            batch_size=32,
            class_mode='categorical')

    valid_generator = test_datagen.flow_from_directory(
            folder_valid,
            target_size=(img_rows, img_cols),#The target_size is the size of your input images,every image will be resized to this size
            batch_size=32,
            class_mode='categorical')

    return train_generator, valid_generator
    
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def build_model():

    model2 = Sequential()
    model2.add(Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) 
    model2.add(Conv2D(8, (3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.25))
    #--------------------------
    model2.add(Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=input_shape)) 
    model2.add(Conv2D(8, (3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.25))
    #--------------------------
    model2.add(Flatten())
    model2.add(Dense(16, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(2, activation='softmax'))
    model2.summary()#prints the summary of the model that was created

    model2.compile(
        loss=keras.losses.categorical_crossentropy, 
        optimizer=keras.optimizers.Adam(), 
        metrics=['accuracy'])  

    model2.summary()

    return model2 


def train_model():
    train_gen, valid_gen = build_loader()

    model = build_model()

    model.fit_generator(train_gen, validation_data=valid_gen)  

    keras_file = 'ship_vs_bike_v1.h5'
    keras.models.save_model(model, keras_file, include_optimizer=False)
    print('Saved baseline model to:', keras_file)


def pretrain_model():
    '''
    Reference:: https://www.kaggle.com/pmigdal/transfer-learning-with-resnet-50-in-keras
    '''

    train_gen, valid_gen = build_loader()

    base_model = keras.applications.ResNet101(
        include_top=False, 
        weights='imagenet',
        input_shape = (224, 224, 3))
    
    # base_model.trainable = False 
    for layer in base_model.layers:
        layer.trainable = False 

    x = base_model.output 
    # use to convert convolution image features to list of feature 
    # averaging all image
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    predictions = keras.layers.Dense(2, activation='softmax')(x)    

    model = keras.Model(base_model.input, predictions)
    

    model.compile(
            loss=keras.losses.categorical_crossentropy, 
            optimizer=keras.optimizers.Adam(), 
            metrics=['accuracy'])  

    model.summary()

    print(model.input)

    model.fit(train_gen, validation_data=valid_gen)  

    keras_file = 'ship_vs_bike_v3_transfer_resnet101.h5'
    keras.models.save_model(model, keras_file, include_optimizer=False)
    print('Saved baseline model to:', keras_file)


def predict():
    model = keras.models.load_model('ship_vs_bike_v3_transfer_resnet101.h5')

    test_img_path = [
        'data/predict/bike/1.jpg',
        'data/predict/bike/2.jpeg',
        'data/predict/ship/1.jpg',
        'data/predict/ship/2.jpg'
    ]

    img_list = [Image.open(img_path) for img_path in test_img_path]

    _shape = (img_rows, img_cols)
    test_batch = np.stack([
        np.array(img.resize(_shape)) for img in img_list
    ])

    pred_probs = model.predict(test_batch)
    print(pred_probs)


def convert_tflite():
    model = keras.models.load_model('ship_vs_bike_v3_transfer_resnet101.h5')

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile('ship_vs_bike_v3_transfer_resnet101.tflite', 'wb') as f:
        f.write(tflite_model)


def convert_pretrained_tflite(path, save_path):
    # model = keras.models.load_model(path)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model_file(path) #from_keras_model(model)
    # converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TF Lite model.
    with tf.io.gfile.GFile(save_path, 'wb') as f:
        f.write(tflite_model)


    
def test_tflite():
    path = 'ship_vs_bike_v3_transfer_resnet101.tflite'
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)


    test_img_path = [
        'data/predict/bike/1.jpg',
        'data/predict/bike/2.jpeg',
        'data/predict/ship/1.jpg',
        'data/predict/ship/2.jpg'
    ]

    img_list = [Image.open(img_path) for img_path in test_img_path]

    _shape = (img_rows, img_cols)
    test_batch = [
        np.array(img.resize(_shape)).astype('float32') for img in img_list
    ]


    from tensorflow.keras.applications.resnet import preprocess_input


    for test in test_batch:
        start_time = time.time()
        x = np.expand_dims(test, axis=0)
        print(x.shape)
        interpreter.set_tensor(input_details[0]['index'], x)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)

        print("######--- %s seconds ---" % (time.time() - start_time))


    


if __name__ == '__main__':
    # load_cate_image()
    # train_model()
    # pretrain_model()
    # predict()
    # convert_tflite()
    # test_tflite()
    convert_pretrained_tflite('resnet18.h5', 'resnet18_v1.tflite')
