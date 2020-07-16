#importing packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.keras.backend import set_session
from PIL import Image
import numpy as np
import flask
import io
import tensorflow
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#initialize our flask applications
app = flask.Flask(__name__)
model = None

def decimal_str(x: float, decimals: int = 10) -> str:
    return format(x, f".{decimals}f").lstrip().rstrip('0')


def load_model():
	# load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
	global sess
	sess = tf.compat.v1.Session()
	global graph
	graph = tf.get_default_graph()
	global model
	set_session(sess)
	model = tensorflow.keras.models.load_model('model/vggtb.h5')
	model.summary()
	#model.load_weights("model/new_model.h5")

def prepare_image(image, target):
	#if the image 	mode is not RGB, Convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	#resize the input the image and processing it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis = 0)
	#image = imagenet_utils.preprocess_input(image)

	#return the processed Image
	return image

@app.route("/predict", methods = ['POST'])
def predict():
	#initialize the data dictionary that will be returned
	data = {"success": False}
	#ensure it is correctly uploaded
	if flask.request.method == "POST":
		#read the image using PIL format
		image = flask.request.files["image"].read()
		image = Image.open(io.BytesIO(image))

		#preparing image for prediction
		image = prepare_image(image, target = (100, 100))

		#classify the input image
		with graph.as_default():
			set_session(sess)
			preds = model.predict(image)
		predsar = preds.tolist()
		print(decimal_str(predsar[0][0], 10))
		#results = imagenet_utils.decode_predictions(preds)
		#data["prediction"] = preds

		#loop over the results and add them to then initialize
		#for (imagenetID, label, prob) in results[0]:
			#r = {"label": label, "probability": float(prob)}
			#data["prediction"].append(r)


		#indicate request
		#data["success"] = True

	#return the data dictionary
	return str(preds)


if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
	load_model()
	app.run()
