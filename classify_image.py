# USAGE
# python classify_image.pt --image images/boat.png

from pytorch_pretrained.config import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2


def preprocessImage(image):
    # Swap the color channel from BGR to RGB
    # CV2 read image in BGR mode but pytorch pretrain model read it in RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    image = image.astype("float32") / 255.0

    # normalize image
    image -= config.MEAN
    image /= config.STD
    # pytorch needs channel first  Order
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)

    return image


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input Image")
ap.add_argument("-m", "--model", type=str, default="vgg16",
                choices=["vgg16", "inception", "resnet", "densenet", "vgg19"],
                help="name of the pretrained network to use")

args = vars(ap.parse_args())
print(args)
MODELS = {
    "vgg16": models.vgg16(pretrained=True),
    "vgg19": models.vgg19(pretrained=True),
    "inception": models.inception_v3(pretrained=True),
    "densenet": models.densenet121(pretrained=True),
    "resnet": models.resnet50(pretrained=True)
}

print("[INFO] loading {} ...".format(args["model"]))
model = MODELS[args["model"]].to(config.DEVICE)
model.eval()

print("[INFO] LOADING IMAGE")

testImage = cv2.imread(args["image"])
origImage = testImage.copy()

testImage = preprocessImage(testImage)

# convert numpy image to pytorch tensor

testImage = torch.from_numpy(testImage)
testImage = testImage.to(config.DEVICE)

print("[INFO] LOADING IMAGENET LABELS ...")

imagenetLabels = dict(enumerate(open(config.IN_LABELS)))

print("[INFO] classify the image and extract the prediction")

logits = model(testImage)
# convert logits into probability

probability = torch.nn.Softmax(dim=-1)(logits)
sortedProbability = torch.argsort(probability, dim=-1, descending=True)
print(sortedProbability[0, :5])
# Loop over sorted probability and print top 5 probability and class labels

for (i, idx) in enumerate(sortedProbability[0, :5]):
    print("{} . {}: {:.2f}%  ".format(
        i, imagenetLabels[idx.item()].strip(),
        probability[0, idx.item()] * 100
    ))

# draw labels on image

(label, prob) = imagenetLabels[probability.argmax().item()], probability.max().item()
cv2.putText(origImage, "Label: {} , {:.2f}% ".format(label.strip(), prob * 100),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imshow("origImageWithLabel",origImage)
cv2.waitKey(0)
cv2.imwrite("images/classfiedImage.png", origImage)
