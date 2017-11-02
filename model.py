import cv2
import csv
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Concatenate, Input, ZeroPadding2D, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import os
import sklearn
import math
from sklearn.model_selection import train_test_split
import uuid


## Function for reading the csv files
def csv_reader(filename):
    lines = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

## Reading data from csv files and calculating left and right steering angles
def data_reader(training_files):
    images = []
    measurements = []
    
    for file in training_files:
        datapath = os.path.dirname(file) + '/IMG/'
        lines = csv_reader(file)
        for line in lines:
            source_path = line[0]
            file_name = source_path.split('/')[-1]
            img_center = datapath + file_name
            
            source_path = line[1]
            file_name = source_path.split('/')[-1]
            img_left = datapath + file_name
            
            source_path = line[2]
            file_name = source_path.split('/')[-1]
            img_right = datapath + file_name
            
            steering_center = float(line[3])
            
            # create adjusted steering measurements for the side camera images
            correction = 0.26 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            
            images.append(img_center)
            measurements.append(steering_center)
            images.append(img_left)
            measurements.append(steering_left)
            images.append(img_right)
            measurements.append(steering_right)
    return (images, measurements)

# Function to augmnet the data by flipping the image and measurements
def augment_data(image, measurement):
    new_image = cv2.flip(image, 1)
    new_measurement = measurement*-1.0
    return (new_image, new_measurement)

# batch data generator
def generator(image_paths, measurements, batch_size=64):
    num_samples = len(image_paths)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_images = image_paths[offset:offset + batch_size]
            batch_measurements = measurements[offset:offset + batch_size]
            images = []
            angles = []
            for batch_image, batch_measurement in zip(batch_images, batch_measurements):
                # Reading image file
                image = cv2.imread(batch_image)
                images.append(image)
                angles.append(batch_measurement)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# main function
def main():
    
    # training csv files
    training_files = ['./data/track2/Forward/driving_log.csv','./data/track2/Reverse/driving_log.csv','./data/track1/Forward/driving_log.csv','./data/track1/Reverse/driving_log.csv']
    images, measurements = data_reader(training_files)
    
    # temporary storage of augmented data
    aug_path = './aug/'
    aug_images = []
    aug_measurements = []
    for image_path, measurement  in zip(images, measurements):
        image = cv2.imread(image_path)
        aug_image, aug_measurement = augment_data(image, measurement)
        uniq_id = uuid.uuid4()
        aug_img_path = aug_path + str(uniq_id) + '.png'
        cv2.imwrite(aug_img_path, aug_image)
        aug_measurements.append(aug_measurement)
        aug_images.append(aug_img_path)
    
    images.extend(aug_images)
    measurements.extend(aug_measurements)

    images_shuff, measurements_shuff = sklearn.utils.shuffle(images, measurements)

    X_train, X_val, y_train, y_val = train_test_split(images_shuff, measurements_shuff, test_size=0.20)

    batch_size=128
    # Compile and train the model using the generator function
    train_generator = generator(X_train, y_train, batch_size=batch_size)
    validation_generator = generator(X_val, y_val, batch_size=batch_size)

    # Creating model
    img_input = Input(shape=(160,320,3))
    x = Lambda(lambda x: x / 255.0 - 0.5 , input_shape=(160,320,3))(img_input)
    x = Cropping2D(cropping=((70,25),(0,0)))(x)
    
    x1 = Conv2D(6,(7,7), strides=(2,2), activation='relu', padding='SAME')(x)
    x1 = Conv2D(9,(7,7), strides=(2,2), activation='relu', padding='SAME')(x1)
    x1 = Conv2D(12,(7,7), strides=(2,2), activation='relu', padding='SAME')(x1)

    x2 = Conv2D(6,(5,5), strides=(2,2), activation='relu', padding='SAME')(x)
    x2 = Conv2D(9,(5,5), strides=(2,2), activation='relu', padding='SAME')(x2)
    x2 = Conv2D(12,(5,5), strides=(2,2), activation='relu', padding='SAME')(x2)

    x3 = Conv2D(6,(3,3), strides=(2,2), activation='relu', padding='SAME')(x)
    x3 = Conv2D(9,(3,3), strides=(2,2), activation='relu', padding='SAME')(x3)
    x3 = Conv2D(12,(3,3), strides=(2,2), activation='relu', padding='SAME')(x3)

    x4 = Conv2D(6,(1,1), strides=(2,2), activation='relu', padding='SAME')(x)
    x4 = Conv2D(9,(1,1), strides=(2,2), activation='relu', padding='SAME')(x4)
    x4 = Conv2D(12,(1,1), strides=(2,2), activation='relu', padding='SAME')(x4)

    x = Concatenate()([x1, x2, x3, x4])

    x1 = Conv2D(12,(7,7), strides=(2,2), activation='relu', padding='SAME')(x)

    x2 = Conv2D(12,(5,5), strides=(2,2), activation='relu', padding='SAME')(x)

    x3 = Conv2D(12,(3,3), strides=(2,2), activation='relu', padding='SAME')(x)

    x4 = Conv2D(12,(1,1), strides=(2,2), activation='relu', padding='SAME')(x)

    x = Concatenate()([x1, x2, x3, x4])

    x = Conv2D(64,(3,3), activation='relu')(x)
    x = Conv2D(64,(3,3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    model = Model(img_input, x)
    model.compile(loss='mse', optimizer='adam')
    
    # training with 18 epochs
    model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(X_train)/batch_size), validation_data=validation_generator, validation_steps=math.ceil(len(X_val)/batch_size), epochs=18)
    # save model
    model.save('model.h5')


if __name__ == '__main__':
    main()
