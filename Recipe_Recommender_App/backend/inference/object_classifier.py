import torch
from torchvision import models, transforms
from PIL import Image
from backend.inference.utils import get_device

device = get_device()

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("backend/model/all_items_classifier.pt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_labels = ['Eggplant', 'Potato', 'Orange', 'Cherry']

def classify_object(image: Image.Image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        pred = torch.argmax(outputs, dim=1).item()
    return class_labels[pred]
