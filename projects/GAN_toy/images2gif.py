import imageio
import os
import sys

gan_name = 'gan'
duration = 0.1
images = []
dir_name = sys.path[0]+'/'+gan_name+'/images'
firenames = {int(fn[:-4]):fn for fn in os.listdir(dir_name) if fn.endswith('png')}

for i in sorted(firenames):
    images.append(imageio.imread(sys.path[0]+'/'+gan_name+'/images/'+firenames[i]))
imageio.mimsave(sys.path[0]+'/gifs/'+gan_name+'.gif', images, duration=duration)