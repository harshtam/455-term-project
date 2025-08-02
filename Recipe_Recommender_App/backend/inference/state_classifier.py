import torch
from torchvision import models, transforms
from PIL import Image
from backend.inference.utils import get_device

device = get_device()

state_models = {
    "Eggplant": "backend/model/eggplant_state_classifier.pt",
    "Potato": "backend/model/potato_state_classifier.pt",
    "Orange": "backend/model/orange_state_classifier.pt",
    "Cherry": "backend/model/cherry_state_classifier.pt"
}

state_labels = {
    "Eggplant": ['Halved', 'Sliced and Cooked', 'Whole'],
    "Potato": ['Fries', 'Peeled', 'Whole'],
    "Orange": ['Peeled', 'Segmented', 'Whole'],
    "Cherry": ['In a Bowl', 'Pitted', 'Stem Attached']
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_loaded_models = {}

def load_state_model(object_class: str) -> torch.nn.Module:
    if object_class not in _loaded_models:
        if object_class not in state_models:
            raise ValueError(f"No model found for object class '{object_class}'")
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        checkpoint = torch.load(state_models[object_class], map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        _loaded_models[object_class] = model
    return _loaded_models[object_class]

def classify_state(image: Image.Image, object_class: str) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL.Image.Image")
    model = load_state_model(object_class)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        pred = torch.argmax(outputs, dim=1).item()
    labels = state_labels.get(object_class)
    if not labels:
        raise ValueError(f"No state labels defined for object class '{object_class}'")
    return labels[pred]
