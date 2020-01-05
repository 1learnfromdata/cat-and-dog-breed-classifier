from flask import Flask, render_template, flash, request, redirect, url_for
import os
from flask import send_from_directory
import time
from PIL import Image
from werkzeug.utils import secure_filename
from models.catAndDogClassifier import catDogClassifier

UPLOAD_FOLDER = 'static/img/catAndDogImages/'

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = UPLOAD_FOLDER
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG"]


@app.route('/')
def homepage():
    return render_template('cat_dog_index.html',image_file_name='')


# procesing uploaded file and predict it for cat and dog classifier
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    def allowed_image(file_name):
        if not "." in file_name:
            return False
        ext = file_name.rsplit(".", 1)[1]
        if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
            return True
        else:
            return False
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":
                return redirect(request.url)
            if allowed_image(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            else:
                return redirect(request.url)
            catdog_var = catDogClassifier(UPLOAD_FOLDER, filename)
            catdog_var.run()
            acc, label = catdog_var.prediction()
            img = Image.open(UPLOAD_FOLDER+filename)
            img = img.resize((256, 256), Image.ANTIALIAS)
            timestr = time.strftime("%m%d%Y%H%M%S")
            filename_to_show = UPLOAD_FOLDER + timestr + "." + image.filename.split('.')[1]
            file_to_send = timestr + "." + image.filename.split('.')[1]
            img.save(filename_to_show)

            if label == "Dog":
                accuracy = str(acc * 100)
            else:
                accuracy = str((1 - acc) * 100)

            if float(accuracy) > 98:
                catdog_var.mobilenet_run()
                label_mob, acc_mob = catdog_var.mobilenet_prediction()
                acc_mob = str(acc_mob * 100)
                return render_template('cat_dog_index.html', image_file_name=file_to_send,
                                       label=label, accuracy=accuracy[:5],breed=label_mob)
            else:
                catdog_var.mobilenet_run()
                label_mob, acc_mob = catdog_var.mobilenet_prediction()
                acc_mob = str(acc_mob * 100)
                if float(acc_mob) > 80:
                    return render_template('cat_dog_index.html', image_file_name=file_to_send,
                                       label=label_mob, accuracy=acc_mob[:5])
                else:
                    return render_template('cat_dog_index.html', image_file_name=file_to_send,
                                           label="Unknown", accuracy="0.0")

    return render_template("cat_dog_index.html", image_file_name='')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run()

