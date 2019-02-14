import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'dev'
Cls=['/0', '/1', '/2', '/3', '/4', '/5', '/6', '/7', '/8', '/9']

for i in range(10):
 os.makedirs(root_dir +'/train' + Cls[i])
 os.makedirs(root_dir +'/val' + Cls[i])
 os.makedirs(root_dir +'/test' + Cls[i])

for i in range(10):
 # Creating partitions of the data after shuffeling
 currentCls = Cls[i]
 src = "dev"+currentCls # Folder to copy images from

 allFileNames = os.listdir(src)
 np.random.shuffle(allFileNames)
 train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.6), int(len(allFileNames)*0.80)])


 train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
 val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
 test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

 print(i)
 print('Total images of: ', len(allFileNames))
 print('Training: ', len(train_FileNames))
 print('Validation: ', len(val_FileNames))
 print('Testing: ', len(test_FileNames))

 # Copy-pasting images
 for name in train_FileNames:
     shutil.copy(name, "dev/train"+currentCls)

 for name in val_FileNames:
     shutil.copy(name, "dev/val"+currentCls)

 for name in test_FileNames:
     shutil.copy(name, "dev/test"+currentCls)