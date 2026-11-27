"""RAM++ image tagger shared by the RAM-based pipelines (tag filtering as in the existing comparison scripts)."""

from __future__ import annotations

import cv2
import torch
from PIL import Image
from torchvision import transforms

RAM_PLUS_CKPT = "/home/ravindu/SceneGraphNav/models/checkpoints/ram_plus_swin_large_14m.pth"

REMOVE_TAGS = [
    "room", "kitchen", "office", "house", "home", "building", "corner",
    "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium",
    "apartment", "image", "city", "blue", "skylight", "hallway",
    "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor",
    "floor", "ladder", "sink", "counter top", "door", "screen door",
    "hardwood", "shower curtain", "curtain", "slide", "peak", "closet",
    "man", "woman", "child", "boy", "girl", "person", "human", "drawer",
    "food", "vegetable", "fruit",
]
ADD_TAGS = ["handle", "cabinet", "plate"]


class RamTagger:
    def __init__(self, device="cuda"):
        from ram.models import ram_plus
        self.device = device
        self.model = ram_plus(pretrained=RAM_PLUS_CKPT, image_size=384, vit="swin_l").eval().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def tags(self, frame_bgr) -> list[str]:
        from ram import inference_ram
        small = cv2.resize(frame_bgr, (384, 384), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        tag_str = inference_ram(self.transform(pil).unsqueeze(0).to(self.device), self.model)[0]
        classes = [t.strip() for t in tag_str.replace(" | ", ".").split(".") if t.strip()]
        classes = [c for c in classes if not any(r in c.lower() for r in REMOVE_TAGS)]
        for a in ADD_TAGS:
            if a not in classes:
                classes.append(a)
        return classes
