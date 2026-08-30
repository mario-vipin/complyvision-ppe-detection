# check_labels.py
import os

label_dir = "PPEs-8/train/labels"
class_names = ["glove","goggles","helmet","mask","no-suit","no_glove",
               "no_goggles","no_helmet","no_mask","no_shoes","shoes","suit"]

counts = {i: 0 for i in range(12)}

for fname in os.listdir(label_dir):
    with open(os.path.join(label_dir, fname)) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                counts[class_id] += 1

for i, name in enumerate(class_names):
    print(f"{name}: {counts[i]} instances")