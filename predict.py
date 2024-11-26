import torch
from torchvision import transforms
from PIL import Image
from model import MyModel

def predict_image(image_path):
    # Load model
    model = MyModel()
    model.load_state_dict(torch.load('best_model.pth'))  # using the best model we saved
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output)
        pred_class = "car" if prediction.item() > 0.5 else "flower"
        confidence = prediction.item() if pred_class == "car" else 1 - prediction.item()
        
    return pred_class, confidence

# Test both images
test_images = [
    "kaggel-dataset/car_test_image.jpg",
    "kaggel-dataset/flower_test_image.jpg"
]

for image_path in test_images:
    pred_class, confidence = predict_image(image_path)
    print(f"\nImage: {image_path}")
    print(f"Prediction: {pred_class}")
    print(f"Confidence: {confidence:.2%}")