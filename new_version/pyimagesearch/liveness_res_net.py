# import the necessary packages
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

class LivenessResNet:
    @staticmethod
    def build(width, height, depth, classes):
        # Create the ResNet50 model without the top layer
        net = ResNet50(include_top=False, weights='imagenet', input_shape=(width, height, depth))
        # net.load_weights('./resnet50_weights.h5', by_name=True)

        # Get the output of the ResNet50 model
        res = net.output

        # Flatten the output from the ResNet50 model
        res = Flatten()(res)

        # Add a Dropout layer for regularization
        res = Dropout(0.5)(res)

        # Add a Dense layer with 'softmax' activation for classification
        fc = Dense(classes, activation='softmax', name='fc2')(res)

        # Create the final model
        model = Model(inputs=net.input, outputs=fc)

        # Return the constructed network architecture
        return model
