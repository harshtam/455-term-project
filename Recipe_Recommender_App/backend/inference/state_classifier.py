import torch
from torchvision import models, transforms
from backend.inference.utils import get_device

device = get_device()

state_models = {
    "Eggplant": "backend/model/eggplant_state_classifier.pt",
    "Potato": "backend/model/potato_state_classifier.pt",
    "Orange": "backend/model/orange_state_classifier.pt",
    "Cherry": "backend/model/cherry_state_classifier.pt"
}

state_labels = {
    "Eggplant": ["Whole", "Halved", "Sliced and Cooked"],
    "Potato": ["Whole", "Peeled", "Fries"],
    "Orange": ["Whole", "Peeled", "Segmented"],
    "Cherry": ["With Stem", "Pitted", "In a Bowl"]
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_loaded_models = {}

def load_state_model(object_class):
    if object_class not in _loaded_models:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model.load_state_dict(torch.load(state_models[object_class], map_location=device))
        model.to(device)
        model.eval()
        _loaded_models[object_class] = model
    return _loaded_models[object_class]

def classify_state(image, object_class):
    model = load_state_model(object_class)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        pred = torch.argmax(outputs, dim=1).item()
    return state_labels[object_class][pred]
