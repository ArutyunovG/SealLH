import torch

# This collate function is designed to handle batches of data 
# where each item is a dictionary containing an "image" tensor 
# and associated "bboxes" and "labels".
#  It stacks the images into a single tensor and keeps the targets
#  as a list of dictionaries, which is suitable for object detection tasks 
# where the number of bounding boxes can vary between images.
def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    targets = [
        {k: item[k] for k in item if k != "image"}
        for item in batch
    ]
    return images, targets
