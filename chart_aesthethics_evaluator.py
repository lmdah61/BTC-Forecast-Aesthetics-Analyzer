import os
import sys
import requests
import tensorflow as tf
from keras.utils import load_img, img_to_array


class AestheticModel:
    NIMA_MODEL_FILENAME = "nima_pretrained_model.h5"
    MODEL_DOWNLOAD_URL = "https://github.com/yunxiaoshi/Neural-IMage-Assessment/releases/download/v0.1.0/nima.h5"

    def __init__(self):
        self._cached_model = None  # Define _cached_model as an instance attribute here.

        # Try to download the NIMA pretrained model online.
        try:
            if not os.path.exists(self.NIMA_MODEL_FILENAME):
                # If the model doesn't exist, download it.
                print("Downloading the NIMA pretrained model...")
                response = requests.get(self.MODEL_DOWNLOAD_URL)

                # Check if the request was successful (status code 200).
                if response.status_code == 200:
                    # If successful, save the model to a local file.
                    with open(self.NIMA_MODEL_FILENAME, "wb") as f:
                        f.write(response.content)

                    print("Download complete.")
                else:
                    raise Exception("Failed to download the NIMA pretrained model.")
        except Exception as e:
            print(e)
            sys.exit(1)

        # Load the NIMA model.
        self.model = tf.keras.models.load_model(self.NIMA_MODEL_FILENAME)

    # Evaluates the aesthetics of an image using the NIMA model. Value between 0 and 1, with 1 being the most aesthetic
    def evaluate_aesthetics(self, image_path):
        if self._cached_model is None:  # Check if _cached_model is not initialized.
            self._cached_model = self.model  # Initialize _cached_model with the loaded model.

        model = self._cached_model

        # Load and preprocess the image.
        img = load_img(image_path, target_size=(224, 224))
        x = img_to_array(img)
        x = tf.expand_dims(x, axis=0)
        x = x / 255.0

        # Make predictions using the NIMA model.
        predictions = model.predict(x)
        aesthetic_score = predictions[0][0]

        return aesthetic_score
