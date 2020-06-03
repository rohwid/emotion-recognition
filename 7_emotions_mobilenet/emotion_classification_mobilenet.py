from __future__ import print_function

import os

from tensorflow.python.keras.applications import MobileNet
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.callbacks import TensorBoard


# MobileNet is designed to work with images of dim 224,224
img_rows, img_cols = 224, 224

MobileNet = MobileNet(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Here we freeze the last 4 layers
# Layers are set to trainable as True by default

for layer in MobileNet.layers:
    layer.trainable = True

# Let's print our layers
for (i, layer) in enumerate(MobileNet.layers):
    print(str(i), layer.__class__.__name__, layer.trainable)


def add_top_model_mobile_net(bottom_model, num_class):
    """creates the top or head of the model that will be
    placed on top of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_class, activation='softmax')(top_model)

    return top_model


num_classes = 7

FC_Head = add_top_model_mobile_net(MobileNet, num_classes)

model = Model(inputs=MobileNet.input, outputs=FC_Head)

print(model.summary())

dir_path = os.getcwd()

train_data_dir = dir_path + '/datasets/training'
validation_data_dir = dir_path + '/datasets/validation'

train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                          rotation_range=30,
                                          width_shift_range=0.3,
                                          height_shift_range=0.3,
                                          horizontal_flip=True,
                                          fill_mode='nearest')

validation_data_generator = ImageDataGenerator(rescale=1. / 255)

batch_size = 32

train_generator = train_data_generator.flow_from_directory(train_data_dir,
                                                           target_size=(img_rows, img_cols),
                                                           batch_size=batch_size,
                                                           class_mode='categorical')

validation_generator = validation_data_generator.flow_from_directory(validation_data_dir,
                                                                     target_size=(img_rows, img_cols),
                                                                     batch_size=batch_size,
                                                                     class_mode='categorical')

checkpoint = ModelCheckpoint('emotion_classification_mobilenet_7_emotions.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=10,
                           verbose=1,
                           restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=5,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.0001)

tensor_board = TensorBoard(log_dir='./graph')

callbacks = [early_stop, checkpoint, learning_rate_reduction, tensor_board]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 28789
nb_validation_samples = 3589

epochs = 25

history = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples // batch_size,
                              epochs=epochs,
                              callbacks=callbacks,
                              validation_data=validation_generator,
                              validation_steps=nb_validation_samples // batch_size)

model_json = model.to_json()

with open("emotion_classification_mobilenet_7_emotions.json", "w") as json_file:
    json_file.write(model_json)
