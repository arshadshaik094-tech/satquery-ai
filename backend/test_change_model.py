import os, glob, requests, json

imgs = sorted(glob.glob("uploads/*.jpg"))[:2]
if len(imgs) < 2:
    print("Need at least 2 images in uploads/")
    exit()

print(f"Testing with: {imgs[0]} vs {imgs[1]}")
resp = requests.post(
    "http://localhost:8000/api/change-detection",
    json={"image_t1_path": imgs[0], "image_t2_path": imgs[1]},
    timeout=120,
)
data = resp.json()
print(f"Engine:   {data.get('engine')}")
print(f"Change%:  {data.get('change_percent')}")
print(f"Trend:    {data.get('trend')}")
print(f"Note:     {data.get('note')}")
