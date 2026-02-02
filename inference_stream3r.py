import os
import torch
from stream3r.models.stream3r import STream3R
from stream3r.models.components.utils.load_fn import load_and_preprocess_images
from pdb import set_trace as st

device = "cuda" if torch.cuda.is_available() else "cpu"

model = STream3R.from_pretrained("yslan/STream3R").to(device)
model.eval()

example_dir = "examples/static_room"
image_names = [os.path.join(example_dir, file) for file in sorted(os.listdir(example_dir))]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
        # Use one mode "causal", "window", or "full" in a single forward pass
        predictions = model(images, mode="causal")
        for k, v in predictions.items():
            print(f"{k}: {v.shape}")
        pass
