import os
levir = r'c:\Users\Shaheed Arshad\Desktop\satquery-ai\backend\train\data\LEVIR-CD'
rsvqa = r'c:\Users\Shaheed Arshad\Desktop\satquery-ai\backend\train\data\RSVQA-LR'
checks = {
    'LEVIR train/A': os.path.isdir(os.path.join(levir,'train','A')),
    'LEVIR val/A':   os.path.isdir(os.path.join(levir,'val','A')),
    'RSVQA questions': os.path.isfile(os.path.join(rsvqa,'RSVQA_LR_dataset_questions.json')),
    'RSVQA answers':   os.path.isfile(os.path.join(rsvqa,'RSVQA_LR_dataset_answers.json')),
    'RSVQA images':    os.path.isdir(os.path.join(rsvqa,'Images_LR')),
}
for k,v in checks.items():
    print(("OK" if v else "MISSING") + ": " + k)
