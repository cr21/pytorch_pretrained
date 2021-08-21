import  torch

IMAGE_SIZE = 224

# IMAGE NET MEAN AND STANDARD DEVIATION
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Specify the path to input labels
IN_LABELS = 'ilsvrc2012_wordnet_lemmas.txt'
