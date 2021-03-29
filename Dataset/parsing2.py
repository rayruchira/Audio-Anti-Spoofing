import numpy as np
import pandas as pd

df=pd.read_csv('ASVspoof2019.LA.cm.train.trn.txt',delim_whitespace=True,header=None)

persons=np.unique(df[0])

for i in persons:
     npath=path+'\\'+i
     os.mkdir(npath)
     for x in np.unique(df[3]):
             os.mkdir(npath+'\\'+x)


import soundfile as sf

for i in range(len(df)):
     f=audio_path+'\\'+df.loc[i][1]+'.flac'
     data,sr=sf.read(f,dtype='float32')
     d=dpath+'\\'+df.loc[i][0]+'\\'+df.loc[i][3]+'\\'+df.loc[i][1]+'.flac'
     sf.write(d,data,sr)

