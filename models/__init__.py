# keras implementation of my DELE CA1 CNN models
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential

# CNN architectures are directly copied over from my best models in DELE CA1 part A
def create_model_128():
    model = Sequential()

    model.add(Conv2D(25, kernel_size=5, input_shape=(128, 128, 1), activation='relu', strides=2))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.35))
    model.add(Conv2D(50, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.4))
    model.add(Conv2D(100, kernel_size=4, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='softmax'))

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.load_weights('models/model_128_weights.h5')
    return model


def create_model_31():
    model = Sequential()

    model.add(Conv2D(40, kernel_size=3, input_shape=(31, 31, 1), activation='relu', strides=2, padding='same'))
    model.add(MaxPooling2D(2))
    model.add(Dropout(0.35))
    model.add(Conv2D(80, kernel_size=3, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(15, activation='softmax'))

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.load_weights('models/model_31_weights.h5')
    return model

# actual labelling so can convert
def get_labels():
    return ['Bean',
            'Bitter_Gourd',
            'Bottle_Gourd',
            'Brinjal',
            'Broccoli',
            'Cabbage',
            'Capsicum',
            'Carrot',
            'Cauliflower',
            'Cucumber',
            'Papaya',
            'Potato',
            'Pumpkin',
            'Radish',
            'Tomato']