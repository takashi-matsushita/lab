"""
Image detection example with YOLO
by Takashi MATSUSHITA
"""

import socket
socket.setdefaulttimeout(5) # give-up early for image retrieval

import urllib
import cv2
import numpy as np
import os
import urllib
import threading
import glob

NTHREAD = 4     # number of threads for image downloads
IMG_SIZE = 400  # output square image size
OUTPUT_DIR = 'img'

## for thread safe counter
lock = threading.Lock()
picid = 0


from PIL import Image
import yolo


### get `unavailable` image to ignore
url = 'https://s.yimg.com/pw/images/en-us/photo_unavailable.png'
response = urllib.request.urlopen(url)
data = np.frombuffer(response.read(), np.uint8)
unavailable = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def resize(img, size):
  """
  resize the input image to square with the specified size
  in order to be proportional, black padding is used for the output image
  """
  shape = img.shape[:2]
  ratio = float(size)/max(shape)
  scaled = tuple([int(x*ratio) for x in shape])
  im = cv2.resize(img, (scaled[1], scaled[0]))
  dw = size - scaled[1]
  dh = size - scaled[0]
  top, bottom = dh//2, dh-(dh//2)
  left, right = dw//2, dw-(dw//2)
  colour = [0, 0, 0]
  return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour)


def worker(tid, urls):
  """
  worker thread for image downloading
  """
  global picid
  #print("dbg> thread {:2d}: starting".format(tid))

  for url in urls:
    try:
      if picid >= min_pics: break

      #print("dbg> thread {:2d}: getting {}".format(tid, url))
      response = urllib.request.urlopen(url)
      data = np.frombuffer(response.read(), np.uint8)
      img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

      # skip if image is 'unavailable'
      if unavailable.shape == img.shape:
        diff = cv2.subtract(unavailable, img)
        if not np.any(diff):
          print('inf> thread {:2d}: unavailable ...{}'.format(tid, url))
          continue

      resized_image = resize(img, IMG_SIZE)
      path = "{}/{:04d}.jpg".format(OUTPUT_DIR, picid)
      cv2.imwrite(path, resized_image)
      print("inf> thread {:2d}: saving as picid = {}".format(tid, picid))
      with lock: # to avoid race condition
        picid += 1

    except Exception as e:
      print("war> thread {:2d} skipping...{}".format(tid, str(e)))

  #print("dbg> thread {:2d}: ending".format(tid))


def store_raw_images(wnid, min_pics):
  """
  get list of image links from ImageNet with the given WordNet ID
  distribute links to worker threads
  """
  url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'.format(wnid)
  request = urllib.request.Request(url)
  response = urllib.request.urlopen(request)
  urls = response.read().decode('utf-8').split()

  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

  lists = []
  for ii in range(NTHREAD):
    lists.append(urls[ii:-1:NTHREAD])

  size = len(lists)
  nn = size//NTHREAD
  for ii in range(size%NTHREAD):
    lists[ii].append(urls[NTHREAD*nn+ii])

  threads = []
  for ii in range(len(lists)):
    t = threading.Thread(target=worker, args=(ii, lists[ii]))
    threads.append(t)

  for t in threads:
    t.start()

  for t in threads:
    t.join()



from nltk.corpus import wordnet as wn
thing = 'cat'   # image to get, can be 'dog', 'cow', etc.

synsets = wn.synsets(thing)
min_pics = 50   # number of minimum pictures to download
for item in synsets:
  if thing not in item.name(): continue
  wnid = '{}{:08d}'.format(item.pos(), item.offset())
  #print('dbg> name = {} wnid = {}'.format(item.name(), wnid))
  store_raw_images(wnid, min_pics)
  if picid >= min_pics: break

#print('dbg> last picid = {}'.format(picid))

print('inf> initialising yolo...')
o = yolo.YOLO()
print('inf> initialising yolo...done')

files = glob.glob('{}/*'.format(OUTPUT_DIR))
ii = 0
for item in files:
  #print('dbg> scanning ... {}'.format(item))
  image = Image.open(item)
  r_image = o.detect_image(image)
  r_image.show()
  r_image.save('yolo-{}-{:03d}.png'.format(thing, ii))
  ii += 1
  
o.close_session()

# eof
