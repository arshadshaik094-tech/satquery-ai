import torch
ckpt = torch.load('weights/change_levir.pt', map_location='cpu', weights_only=False)
print('Keys:', list(ckpt.keys()))
print('Epoch:', ckpt.get('epoch'))
print('Val IoU:', ckpt.get('val_iou'))
print('Arch:', ckpt.get('arch'))
sd_keys = list(ckpt['state_dict'].keys())[:8]
print('First 8 state_dict keys:', sd_keys)
