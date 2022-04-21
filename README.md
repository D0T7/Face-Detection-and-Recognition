# Face-Detection-and-Recognition
Python program to detect faces and recognize them.

Steps to run this program:<br/><br/>
This program required [openface_nn4.small2.v1.t7](https://github.com/pyannote/pyannote-data/blob/master/openface.nn4.small2.v1.t7) to run.
Download repo as zip, download openface_nn4.small2.v1.t7 and copy it to the code directory.
Create a "dataset" folder with subfolders named as the person's name contaning that particular person's images.
Open Terminal in the code directory and run <br/><br/>
```python extract_embeddings.py --dataset dataset --embeddings output/faceEmbed.pickle --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7```<br/><br/>
This will extract embedding and process images for training.<br/>
Next run<br/><br/>
```python train_model.py --embeddings output/faceEmbed.pickle --recognizer output/recognizer.pickle --le output/label.pickle```<br/><br/>
This will train the model to recognize the person.<br/>
Finally to recognise the person run<br/><br/>
```python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/label.pickle```<br/><br/>
This will open the webcam and recognize the person in realtime<br/>
                OR<br/>
Create a folder "images" in the code directory and add images of persons to try out face recognition with the following code<br/><br/>
```python recognize.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/label.pickle --image images/```
