from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



pre=load_model('ResNet.h5')

labels=pd.read_csv('E:\Projects\CropClassificaton\labels.csv')


app=Flask(__name__)


@app.route('/')
def home():
    
    return render_template('home.html')

@app.route('/predict',methods=["post"])

def predict():
    image=request.files['img']
    imge=plt.imread(image)
    imge= cv2.resize(imge,(224,224))
    imge=imge/255
    imgee=np.expand_dims(imge,axis=0)
    imgee=np.array(imgee)
    pr=pre.predict(imgee)
    pr=np.argmax(pr)
    pr=labels['0'].loc[pr]
    pr=str(pr)
    return render_template('predict.html',imag=pr)
   


if __name__=='__main__':
    app.run(debug=True)

