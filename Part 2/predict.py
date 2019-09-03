import json

from get_extra_input_predict import get_extra_input
from load_checkpoint import load_checkpoint
from predict_fun import predict

arg = get_extra_input()

path_checkpoint = 'default_saving_folder/' + arg.checkpoint
path_image = arg.path_to_image

model = load_checkpoint(path_checkpoint)
model.to("cuda" if arg.gpu else "cpu")

with open(arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

image_name = cat_to_name[path_image.split("/")[2]]

p, c = predict(path_image, model, arg.top_k, arg.gpu)
probs, classes = [], []
for x in c:
    classes.append(cat_to_name[str(x)]+"({})".format(x))
for x in p:
    probs.append(float(x)*100)
results = dict(zip(classes, probs))

print("True name of the input: ", image_name)
print("Results: ")
for x, y in results.items():
    print("{} : {:.2f}%".format(x,y))
