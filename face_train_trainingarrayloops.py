import glob
from PIL import Image

class result_class():
    image = Image.Image()
    index = " "
    def __init__(self,person,id):
        self.image = person
        self.index = id


def add_element(result = [] ):
    result.append(result_class () )
    return result

def match_pairs(max, img = [] ):
    result = []
    max_length = len(img)
    for i in range(max_length):
        if i < max:
            print(i)
            result.append([])
            result[i].append([])
            result[i][0].append(result_class(img[i],i))
            j = i +1
            result[i].append([])
            result[i][1].append(result_class(img[j],j))
            i = j +1
        else:
            break
    return result




image_list = []
for filename in glob.glob('dataset/aug/test/BradPittTest/*.jpg'):
    im = Image.open(filename)
    image_list.append(im)




result = match_pairs(50, image_list)


print(result)

print(result[:1])

image = Image.Image()
image = result[0][0][0].image

image.show()