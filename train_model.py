# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle
# python train_model.py --embeddings output/PyPower_embed.pickle --recognizer output/PyPower_recognizer.pickle --le output/PyPower_label.pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,
                help="tells about path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,
                help="defines about the path to the output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
                help="defines about the path to the output label encoder")
args = vars(ap.parse_args())

print("[INFO] here we can see loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

print("[INFO] here we can see encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
