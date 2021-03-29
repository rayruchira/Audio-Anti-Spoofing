import pandas as pd
import os
import shutil

df=pd.read_csv('F:\Samsung Prism\dataset\DS_10283_3336\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt',' ',names=['id','name','-','technique','type'])
df.to_csv (r'F:\Samsung Prism\dataset\DS_10283_3336\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.csv', index=None)
bonafide=df.loc[df['type']=='bonafide']
spoof=df.loc[df['type']=='spoof']
print("eval")
#print(list(set(spoof['technique'])))
print(len(bonafide))
print(len(spoof))

'''
path = r'F:\Samsung Prism\dataset\DS_10283_3336\LA\LA\ASVspoof2019_LA_dev\flac'
s_to  = r'F:\Samsung Prism\dataset\spoof'
b_to = r'F:\Samsung Prism\dataset\bonafide'



for i in bonafide['name']:
    ##
    filepath = path + '\\' + i + '.flac'
    shutil.move(filepath, b_to)
    print(filepath)

for i in spoof['name']:
    filepath = path + '\\' + i + '.flac'
    shutil.move(filepath, s_to)
    ##
'''
