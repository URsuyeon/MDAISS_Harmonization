import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from sr3d_simple_class import Simple3DClassifier, MedicalVolumeDataset  

def extract_features_and_gradients(model, target_layer):
    features = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    fw_handle = target_layer.register_forward_hook(forward_hook)
    bw_handle = target_layer.register_full_backward_hook(backward_hook)
    return fw_handle, bw_handle, lambda: (features, gradients)

def compute_gradcam_3d(model, volume, target_class, target_layer, device):
    model.eval()
    volume = volume.to(device).unsqueeze(0)

    fw_handle, bw_handle, get_data = extract_features_and_gradients(model, target_layer)
    outputs = model(volume)
    score = outputs[0, target_class]
    model.zero_grad()
    score.backward(retain_graph=True)
    features, gradients = get_data()
    fw_handle.remove(); bw_handle.remove()

    weights = torch.mean(gradients.view(gradients.size(0), -1), dim=1)
    d_prime, h_prime, w_prime = features.shape[2], features.shape[3], features.shape[4]
    cam = torch.zeros(d_prime, h_prime, w_prime, device=device)
    for i, w in enumerate(weights):
        cam += w * features[0, i]

    cam = F.relu(cam)
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()
    print(f"[Grad-CAM Debug] min={cam.min():.4f}, max={cam.max():.4f}, mean={cam.mean():.4f}")

    cam = cam.unsqueeze(0).unsqueeze(0)
    cam = F.interpolate(
        cam,
        size=volume.shape[-3:],
        mode='trilinear',
        align_corners=False
    )
    return cam.detach()[0, 0].cpu().numpy()

def visualize_slices(volume, cam, hospital_name, true_label, pred_label, pred_conf, output_dir=None, show=False):
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'gradcam')
    os.makedirs(output_dir, exist_ok=True)

    vol = volume.squeeze().cpu().numpy()
    d, h, w = cam.shape
    slices = {'axial': d//2, 'coronal': h//2, 'sagittal': w//2}

    for plane, idx in slices.items():
        if plane == 'axial': orig, heat = vol[idx, :, :], cam[idx, :, :]
        if plane == 'coronal': orig, heat = vol[:, idx, :], cam[:, idx, :]
        if plane == 'sagittal': orig, heat = vol[:, :, idx], cam[:, :, idx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # 원본
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title(f'Original - {plane.capitalize()} {idx}', pad=15)
        axes[0].axis('off')
        # Grad-CAM
        axes[1].imshow(orig, cmap='gray')
        axes[1].imshow(heat, cmap='jet', alpha=0.5)
        title = (f'{hospital_name} True:{true_label} | Pred:{pred_label} '
                 f'({pred_conf*100:.1f}%) - {plane.capitalize()}')
        axes[1].set_title(title, pad=15)
        axes[1].axis('off')

        plt.tight_layout()
        fig.subplots_adjust(top=0.88)

        fname = f'gradcam_{hospital_name}_true{true_label}_pred{pred_label}_{plane}.png'
        save_path = os.path.join(output_dir, fname)
        fig.savefig(save_path)
        print(f'[Grad-CAM] Saved {plane} slice: {save_path}')

        if show:
            plt.show()
        plt.close(fig)

def main():
     # ——— 변환 데이터(Transformed)용 ———
    csv_path = '/public/sylee/MDAISS_2/output/transformed/sr3d/test_sr_transformed.csv'
    model_path = '/public/sylee/MDAISS_2/models/classifier/simple_transformed/best_model_simple.pth'
    
    # ——— 원본 데이터(Original)용 ———
    # csv_path = '/public/sylee/MDAISS_2/output/test_dataset_2.csv'
    # model_path = '/public/sylee/MDAISS_2/models/classifier/simple_original/best_model_simple.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(model_path, map_location=device)
    num_classes = state['fc.weight'].size(0)
    model = Simple3DClassifier(input_channels=1, base_filters=32, num_classes=num_classes).to(device)
    model.load_state_dict(state)

    dataset = MedicalVolumeDataset(csv_file=csv_path, target_shape=(256,256,256))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    label_to_hospital = {v: k for k, v in dataset.hospital_to_label.items()}
    visualize_on_screen = False

    seen = {}
    for idx, (vol, lbl) in enumerate(loader):
        lbl = lbl.item()
        if lbl not in seen:
            seen[lbl] = idx
        if len(seen) == num_classes:
            break
    samples = [(idx, lbl) for lbl, idx in seen.items()]

    target_layer = model.layers[-1][0]
    for idx, true_label in samples:
        volume, lbl = dataset[idx]
        # 예측 수행
        model.eval()
        with torch.no_grad():
            inp = volume.to(device).unsqueeze(0)
            out = model(inp)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred_label = int(probs.argmax())
        pred_conf = float(probs[pred_label])

        cam = compute_gradcam_3d(model, volume, target_class=pred_label,
                                 target_layer=target_layer, device=device)
        hospital_name = label_to_hospital[true_label]
        visualize_slices(volume, cam, hospital_name,
                         true_label, pred_label, pred_conf,
                         show=visualize_on_screen)

if __name__ == '__main__':
    main()
