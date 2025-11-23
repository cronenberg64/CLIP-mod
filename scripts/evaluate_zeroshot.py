import torch
import open_clip
from torchvision.datasets import CIFAR100
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

def evaluate_zeroshot(model_name, dataset_root="data/cifar100", batch_size=32, device='cuda'):
    print(f"Evaluating Zero-shot Accuracy for {model_name} on CIFAR-100...")
    
    # 1. Load Model
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=False, # We are testing the architecture/random weights for now, or load specific checkpoint if available
            force_custom_text=True if 'convnext' in model_name else False
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return

    # 2. Load Dataset
    try:
        dataset = CIFAR100(root=dataset_root, download=True, train=False, transform=preprocess)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
        class_names = dataset.classes
    except Exception as e:
        print(f"Failed to load CIFAR-100: {e}")
        return

    # 3. Create Text Classifier
    print("Building text classifier...")
    text_inputs = torch.cat([open_clip.tokenize(f"a photo of a {c}") for c in class_names]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 4. Evaluation Loop
    correct = 0
    total = 0
    
    print("Running inference...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T)
            probs = similarity.softmax(dim=-1)
            
            pred = probs.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
    accuracy = 100 * correct / total
    print(f"\nModel: {model_name}")
    print(f"Zero-shot Accuracy on CIFAR-100: {accuracy:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    # Evaluate Baseline
    # evaluate_zeroshot("ViT-B-32")
    
    # Evaluate Our Model
    evaluate_zeroshot("convnext_v2_tiny")
