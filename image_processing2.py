import PIL
import matplotlib
im = PIL.Image.open("res.JPG")

black=0
red=0
for pixel in im.getdata():
    if pixel is (0,0,0):
        black += 1
    else:
        red += 1
print(red)