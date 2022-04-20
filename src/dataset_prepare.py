import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os

# convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# making folders
outer_names = ['test','train']
inner_names = ['Angrily Disgusted', 'Fearfully Angry', 'Fearfully Surprised', 'Happily Disgusted', 'Happily Surprised', 'Sadly Angry', 'Sadly Disgusted']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data',outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data',outer_name,inner_name), exist_ok=True)

# to keep count of each category
Angrily_Disgusted = 0
Fearfully_Angry = 0
Fearfully_Surprised = 0
Happily_Disgusted = 0
Happily_Surprised = 0
Sadly_Angry = 0
Sadly_Disgusted = 0
Angrily_Disgusted_test = 0
Fearfully_Angry_test = 0
Fearfully_Surprised_test = 0
Happily_Disgusted_test = 0
Happily_Surprised_test = 0
Sadly_Angry_test = 0
Sadly_Disgusted_test = 0

df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48,48),dtype=np.uint8)
print("Saving images...")

# read the csv file line by line
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # the image size is 48x48
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

    # train
    if i < 28709:
        if df['emotion'][i] == 0:
            img.save('train/Angrily Disgusted/im'+str(Angrily_Disgusted)+'.png')
            Angrily_Disgusted += 1
        elif df['emotion'][i] == 1:
            img.save('train/Fearfully Angry/im'+str(Fearfully_Angry)+'.png')
            Fearfully_Angry += 1
        elif df['emotion'][i] == 2:
            img.save('train/Fearfully Surprised/im'+str(Fearfully_Surprised)+'.png')
            Fearfully_Surprised += 1
        elif df['emotion'][i] == 3:
            img.save('train/Happily Disgusted/im'+str(Happily_Disgusted)+'.png')
            Happily_Disgusted += 1
        elif df['emotion'][i] == 4:
            img.save('train/Happily Surprised/im'+str(Happily_Surprised)+'.png')
            Happily_Surprised += 1
        elif df['emotion'][i] == 5:
            img.save('train/Sadly Angry/im'+str(Sadly_Angry)+'.png')
            Sadly_Angry += 1
        elif df['emotion'][i] == 6:
            img.save('train/Sadly Disgusted/im'+str(Sadly_Disgusted)+'.png')
            Sadly_Disgusted += 1

    # test
    else:
        if df['emotion'][i] == 0:
            img.save('test/Angrily Disgusted/im'+str(Angrily_Disgusted_test)+'.png')
            Angrily_Disgusted_test += 1
        elif df['emotion'][i] == 1:
            img.save('test/Fearfully Angry/im'+str(Fearfully_Angry_test)+'.png')
            Fearfully_Angry_test += 1
        elif df['emotion'][i] == 2:
            img.save('test/Fearfully Surprised/im'+str(Fearfully_Surprised_test)+'.png')
            Fearfully_Surprised_test += 1
        elif df['emotion'][i] == 3:
            img.save('test/Happily Disgusted/im'+str(Happily_Disgusted_test)+'.png')
            Happily_Disgusted_test += 1
        elif df['emotion'][i] == 4:
            img.save('test/Happily Surprised/im'+str(Happily_Surprised_test)+'.png')
            Happily_Surprised_test += 1
        elif df['emotion'][i] == 5:
            img.save('test/Sadly Angry/im'+str(Sadly_Angry_test)+'.png')
            Sadly_Angry_test += 1
        elif df['emotion'][i] == 6:
            img.save('test/Sadly Disgusted/im'+str(Sadly_Disgusted_test)+'.png')
            Sadly_Disgusted_test += 1

print("Done!")
