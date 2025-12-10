import torch, timm, numpy as np, random, cv2
from PIL import Image
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

device = torch.device('cuda')

MODELS = {
    ('ViT', 'ORIGINAL'): 'ORIGINAL_DATA/Vision Transformers/ViT/ViT_TumorClassifier.pth',
    ('MedViT', 'ORIGINAL'): 'ORIGINAL_DATA/Biomedical Transformers/MedViT/MedViT_DermClassifier.pth',
    ('MedViT', 'DCGAN'): 'DCGAN_data/biomedical Transformers/MedViT/MedViT_DCGAN.pth',
    ('MedViT', 'DCGAN_UPSCALED'): 'DCGAN_data_upscaled/Biomedical Transformers/MedViT/MedViT_DCGAN_Upscaled.pth',
    ('MedViT', 'DDPM'): 'DDPM_data/Biomedical Transformers/MedViT/MedViT_DDPM.pth',
}

DIRS = {
    'ORIGINAL': 'ORIGINAL_DATA/ORIGINAL_IMAGES',
    'DCGAN': 'DCGAN_data/images',
    'DCGAN_UPSCALED': 'DCGAN_data_upscaled/upscaled_images',
    'DDPM': 'DDPM_data/images_ddpm',
    'DDPM_UPSCALED': 'DDPM_upscaled_data/images_ddpm_upscaled',
}

def sample(d):
    imgs = list(Path(d).glob('*.jpg')) + list(Path(d).glob('*.png'))
    random.shuffle(imgs)
    return [(str(p), 0 if i<5 else 1, p.name) for i,p in enumerate(imgs[:10])]

samples = {k: sample(v) for k,v in DIRS.items() if Path(v).exists()}
print(f"Samples: {sum(len(v) for v in samples.values())}")

tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def reshape_vit(x):
    return x[:, 1:, :].reshape(x.shape[0], 14, 14, x.shape[2]).permute(0, 3, 1, 2)

for i, ((name, ds), path) in enumerate(MODELS.items(), 1):
    print(f"\n[{i}/{len(MODELS)}] {name}_{ds}...")
    if not Path(path).exists():
        print(f"  SKIP: checkpoint not found")
        continue
    
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        
        # Strip prefixes
        new_sd = {}
        for k, v in sd.items():
            key = k.replace('medvit.', '').replace('vit.', '').replace('model.', '')
            new_sd[key] = v
        
        # Detect num_classes
        nc = 2
        for k in new_sd.keys():
            if 'head.weight' in k:
                nc = new_sd[k].shape[0]
                break
        
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=nc)
        model.load_state_dict(new_sd, strict=False)
        
        # Wrap if single class
        if nc == 1:
            class Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.model = m
                def forward(self, x):
                    logit = self.model(x)
                    prob = torch.sigmoid(logit)
                    return torch.cat([1-prob, prob], dim=1)
            model = Wrapper(model)
        
        model = model.to(device).eval()
        
        # Get correct layer
        base = model.model if nc == 1 else model
        cam = GradCAM(model=model, target_layers=[base.blocks[-1].norm1], reshape_transform=reshape_vit)
        
        outdir = Path(f'gradcam_outputs/{name}_{ds}')
        outdir.mkdir(exist_ok=True)
        
        # Clear existing files
        for f in outdir.glob('*.jpg'):
            f.unlink()
        
        cnt = 0
        for src, imgs in samples.items():
            for pth, lbl, fname in imgs:
                try:
                    img = Image.open(pth).convert('RGB')
                    tens = tfm(img).unsqueeze(0).to(device)
                    gcam = cam(input_tensor=tens, targets=[ClassifierOutputTarget(lbl)])[0]
                    
                    img_small = img.resize((224,224), Image.Resampling.LANCZOS).convert('RGB')
                    arr = np.array(img_small, dtype=np.float32) / 255.0
                    
                    heatmap = np.uint8(255 * gcam)
                    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                    img_uint = np.uint8(255 * arr)
                    vis = np.uint8(0.5 * img_uint + 0.5 * heatmap_colored)
                    
                    Image.fromarray(vis).save(outdir / f"{src}_{fname}".replace('.png','.jpg'), 'JPEG', quality=95)
                    cnt += 1
                except:
                    pass
        
        print(f"  {cnt} heatmaps OK")
        del model, cam
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nDONE!")
