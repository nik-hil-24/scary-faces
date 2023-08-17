import os
from torchvision.utils import save_image


# Saving Generated Images
def inference(model, e):
    # Folder for Images to be Stored in
    if 'vae' not in os.listdir():
        try:
            os.mkdir('vae')
        except:
            pass

    # Generate Image
    images = model.generate()
    # Save Images
    for i in range(2):
        out = images[i].view(3, 64, 64)
        save_image(out, f"vae/generated_at_epoch_{e+1}_iter_{i+1}.png")
