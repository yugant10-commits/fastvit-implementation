import torch
import models
from timm.models import create_model
from models.modules.mobileone import reparameterize_model
from PIL import Image
import torchvision.transforms as transforms
import os

model = create_model("fastvit_t8")


def preprocess_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((225, 225)),  # Resize to the size expected by the model
        transforms.ToTensor(),  # Convert the image to a tensor

    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    return input_batch

# Please add the checkpoint of your trained model below and if you have a GPU available you can set the map_location to gpu. 
checkpoint = torch.load("output/20240910-225941-fastvit_t8-256/checkpoint-99.pth.tar", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model_inf = reparameterize_model(model)

# You can download some image and give the path below to test.
processed_img = preprocess_image("commercial_items/validation/can/can0.jpeg")

print(processed_img.shape)

# This function changes the output of the model to a more suitable form to get our results. 
def predict(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)
        probabilities = output[0]
        # print(f"Output:{output}")
        # print(f"Output:{output.shape}")

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(f"probaba:{probabilities}")
    return probabilities

probabilities = predict(model, processed_img)
top5_prob, top5_catid = torch.topk(probabilities, 3)

"""
This part below is required as without this our output would just be Category 1 and so on. 
The path give below to the os.listdir() is the path to the training image folder. 
"""

table_list = sorted(os.listdir('commercial_items/train'))
for i in range(top5_prob.size(0)):
    print(f"Category: {top5_catid[i].item()}, Probability: {top5_prob[i].item()}, Item name:{table_list[top5_catid[i].item()]}")



