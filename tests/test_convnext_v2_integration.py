import torch
import open_clip
from open_clip import create_model_and_transforms

def test_convnext_v2_integration():
    print("Testing ConvNeXt-V2 integration...")
    
    # 1. Create Model
    model_name = 'convnext_v2_tiny'
    try:
        model, _, preprocess = create_model_and_transforms(
            model_name, 
            pretrained=False,
            force_custom_text=True # Ensure we use the config we just made if needed, though create_model should find it
        )
        print(f"Successfully created model: {model_name}")
    except Exception as e:
        print(f"Failed to create model: {e}")
        return

    # 2. Dummy Data
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224)
    text = open_clip.tokenize(["a photo of a cat", "a photo of a dog"])

    # 3. Forward Pass
    try:
        image_features, text_features, logit_scale = model(image, text)
        print(f"Forward pass successful.")
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return

    # 4. Loss Calculation (Contrastive)
    try:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        labels = torch.arange(batch_size, device=image.device)
        
        loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2
        print(f"Loss calculated: {loss.item()}")
    except Exception as e:
        print(f"Loss calculation failed: {e}")
        return

    # 5. Backward Pass
    try:
        loss.backward()
        print("Backward pass successful.")
    except Exception as e:
        print(f"Backward pass failed: {e}")
        return

    print("\nIntegration Test PASSED!")

if __name__ == "__main__":
    test_convnext_v2_integration()
