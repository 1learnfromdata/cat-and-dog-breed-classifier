import PIL.Image as Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import json
model_folder_name = 'static/models/cat_vs_dog/'


class catDogClassifier():
    def __init__(self,folder_name, filename):
        self.folder_name = folder_name
        self.filename = filename
        self.model = None
        self.model_json = None
        self.mobilenet_model_json = None
        self.mobilenet_model = None

    # Stage 2: Load the pretrained model
    def run(self):
        json_file = model_folder_name + 'catanddog_model.json'
        with open(json_file, 'r') as f:
            self.model_json = f.read()

        self.model = tf.keras.models.model_from_json(self.model_json)
        # load weights into new model
        model_file = model_folder_name + 'catanddog_model.h5'
        self.model.load_weights(model_file)

    def mobilenet_run(self):
        mobilenet_json_file = model_folder_name + 'mobilenet.json'
        with open(mobilenet_json_file, 'r') as f:
            self.mobilenet_model_json = f.read()

        self.mobilenet_model = tf.keras.models.model_from_json(self.mobilenet_model_json)
        # load weights into new model
        mobilenet_model_file = model_folder_name + 'mobilenet_model.h5'
        self.mobilenet_model.load_weights(mobilenet_model_file)


    # call model to predict an image
    def api(self, full_path):
        IMAGE_SHAPE_2 = (128, 128)
        grace_hopper = Image.open(full_path).resize(IMAGE_SHAPE_2)
        grace_hopper = np.array(grace_hopper) / 255.0
        result = self.model.predict(grace_hopper[np.newaxis, ...])
        return result

    def prediction(self):
        indices = {0: 'Cat', 1: 'Dog', 2: 'Invasive carcinomar', 3: 'Normal'}
        full_name = self.folder_name + self.filename
        accuracy = self.api(full_name)
        result = int(accuracy[0][0].round())
        label = indices[result]
        return accuracy[0][0], label

    def mobilenet_prediction(self):

        def prepare_image(file):
            print('inside prepare',file)
            img = image.load_img(file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

        # =============================================================================
        # Image(filename='/home/batman/python_projects/flask_blog_version1/myblog/static/img/catAndDogImages/dog_im.jpeg')
        # =============================================================================

        def decode_predictions(preds, top=5, **kwargs):
            """Decodes the prediction of an ImageNet model.
            # Arguments
                preds: Numpy tensor encoding a batch of predictions.
                top: Integer, how many top-guesses to return.

            # Returns
                A list of lists of top class prediction tuples
                `(class_name, class_description, score)`.
                One list of tuples per sample in batch input.
            # Raises
                ValueError: In case of invalid shape of the `pred` array
                    (must be 2D).
            """
            global CLASS_INDEX

            if len(preds.shape) != 2 or preds.shape[1] != 1000:
                raise ValueError('`decode_predictions` expects '
                                 'a batch of predictions '
                                 '(i.e. a 2D array of shape (samples, 1000)). '
                                 'Found array with shape: ' + str(preds.shape))

            fpath = model_folder_name + "imagenet_class_index.json"
            with open(fpath) as f:
                CLASS_INDEX = json.load(f)
            results = []
            for pred in preds:
                top_indices = pred.argsort()[-top:][::-1]
                result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
                result.sort(key=lambda x: x[2], reverse=True)
                results.append(result)
            return results

        print(self.folder_name + self.filename)

        preprocessed_image = prepare_image(self.folder_name + self.filename)
        predictions = self.mobilenet_model.predict(preprocessed_image)
        results_var = decode_predictions(predictions)
        return results_var[0][0][1], results_var[0][0][2]



# this_var = catDogClassifier()
# this_var.run()
# this_var.prediction()


