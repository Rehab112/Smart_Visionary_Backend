from ultralytics import YOLO
import os

model = YOLO("flaskr/models/best.pt")

def predict(image_path, filename):

    prediction = model(image_path)
    # print(prediction)
    names = model.names
    output = "you got "
    sum = 0
    for r in prediction:
        for c in r.boxes.cls:
            if(names[int(c)] == "5EGP"):
                output += "5 pounds\n"
                sum+= 5
            elif(names[int(c)] == "20EGP"):
                output += "10 pounds\n"
                sum+= 10
            elif(names[int(c)] == "5EGP"):
                output += "20 pounds\n"
                sum+= 20
            elif(names[int(c)] == "50EGP"):
                output += "50 pounds\n"
                sum+= 50
            elif(names[int(c)] == "100EGP"):
                output += "100 pounds\n"
                sum+= 100
            elif(names[int(c)] == "200EGP"):
                output += "200 pounds\n"
                sum+= 200
            
    output += "total money you have is "+ str(sum) + " egyptian pounds"
    return output

