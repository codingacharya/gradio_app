import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr
import requests

# Load pretrained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(LABELS_URL)
classes = response.text.strip().split('\n')

# Image transformation with manual mean/std
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

def classify_image(img: Image.Image):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_idx = torch.topk(probs, 5)
    return {classes[i]: float(top5_prob[idx]) for idx, i in enumerate(top5_idx)}

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="ResNet18 Image Classifier",
    description="Upload an image to classify it using a pretrained ResNet18 model."
)

iface.launch()
