Python 3.6.8 (tags/v3.6.8:3c6b436a57, Dec 24 2018, 00:16:47) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
== RESTART: E:/samsung prism/LA/ASVspoof2019_LA_cm_protocols/extraction.py ==

>>> jd=json.loads(f)
Traceback (most recent call last):
  File "<pyshell#0>", line 1, in <module>
    jd=json.loads(f)
  File "D:\python installation\lib\json\__init__.py", line 348, in loads
    'not {!r}'.format(s.__class__.__name__))
TypeError: the JSON object must be str, bytes or bytearray, not 'TextIOWrapper'
>>> jd=json.loads('ASVspoof2019.LA.cm.dev.trl.txt')
Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    jd=json.loads('ASVspoof2019.LA.cm.dev.trl.txt')
  File "D:\python installation\lib\json\__init__.py", line 354, in loads
    return _default_decoder.decode(s)
  File "D:\python installation\lib\json\decoder.py", line 339, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "D:\python installation\lib\json\decoder.py", line 357, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
>>> import pandas as pd
>>> type(f)
<class '_io.TextIOWrapper'>
>>> f[0]
Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    f[0]
TypeError: '_io.TextIOWrapper' object is not subscriptable
>>> f
<_io.TextIOWrapper name='ASVspoof2019.LA.cm.dev.trl.txt' mode='r' encoding='cp1252'>
>>> print(f)
<_io.TextIOWrapper name='ASVspoof2019.LA.cm.dev.trl.txt' mode='r' encoding='cp1252'>
>>> f.read()
''
>>> print(f.read())

>>> 
== RESTART: E:/samsung prism/LA/ASVspoof2019_LA_cm_protocols/extraction.py ==

>>> df=pd.read.csv('ASVspoof2019.LA.cm.dev.trl.txt')
Traceback (most recent call last):
  File "<pyshell#9>", line 1, in <module>
    df=pd.read.csv('ASVspoof2019.LA.cm.dev.trl.txt')
AttributeError: module 'pandas' has no attribute 'read'
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt')
>>> df
       LA_0069 LA_D_1047731 - - bonafide
0      LA_0069 LA_D_1105538 - - bonafide
1      LA_0069 LA_D_1125976 - - bonafide
2      LA_0069 LA_D_1293230 - - bonafide
3      LA_0069 LA_D_1340209 - - bonafide
4      LA_0069 LA_D_1376638 - - bonafide
...                                  ...
24838   LA_0078 LA_D_9924204 - A06 spoof
24839   LA_0078 LA_D_9931163 - A06 spoof
24840   LA_0078 LA_D_9935163 - A06 spoof
24841   LA_0078 LA_D_9944718 - A06 spoof
24842   LA_0078 LA_D_9967770 - A06 spoof

[24843 rows x 1 columns]
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt')
>>> df
                       S.No\tfname\ttype
0      LA_0069 LA_D_1047731 - - bonafide
1      LA_0069 LA_D_1105538 - - bonafide
2      LA_0069 LA_D_1125976 - - bonafide
3      LA_0069 LA_D_1293230 - - bonafide
4      LA_0069 LA_D_1340209 - - bonafide
...                                  ...
24839   LA_0078 LA_D_9924204 - A06 spoof
24840   LA_0078 LA_D_9931163 - A06 spoof
24841   LA_0078 LA_D_9935163 - A06 spoof
24842   LA_0078 LA_D_9944718 - A06 spoof
24843   LA_0078 LA_D_9967770 - A06 spoof

[24844 rows x 1 columns]
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt')
>>> df
                  sno\tfname\t-\t-\ttype
0      LA_0069 LA_D_1047731 - - bonafide
1      LA_0069 LA_D_1105538 - - bonafide
2      LA_0069 LA_D_1125976 - - bonafide
3      LA_0069 LA_D_1293230 - - bonafide
4      LA_0069 LA_D_1340209 - - bonafide
...                                  ...
24839   LA_0078 LA_D_9924204 - A06 spoof
24840   LA_0078 LA_D_9931163 - A06 spoof
24841   LA_0078 LA_D_9935163 - A06 spoof
24842   LA_0078 LA_D_9944718 - A06 spoof
24843   LA_0078 LA_D_9967770 - A06 spoof

[24844 rows x 1 columns]
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt')
>>> df
                      sno fname - - type
0      LA_0069 LA_D_1047731 - - bonafide
1      LA_0069 LA_D_1105538 - - bonafide
2      LA_0069 LA_D_1125976 - - bonafide
3      LA_0069 LA_D_1293230 - - bonafide
4      LA_0069 LA_D_1340209 - - bonafide
...                                  ...
24839   LA_0078 LA_D_9924204 - A06 spoof
24840   LA_0078 LA_D_9931163 - A06 spoof
24841   LA_0078 LA_D_9935163 - A06 spoof
24842   LA_0078 LA_D_9944718 - A06 spoof
24843   LA_0078 LA_D_9967770 - A06 spoof

[24844 rows x 1 columns]
>>> read_file.to_csv (r'ASVspoof2019.LA.cm.dev.trl.csv', index=None)

Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    read_file.to_csv (r'ASVspoof2019.LA.cm.dev.trl.csv', index=None)
NameError: name 'read_file' is not defined
>>> df.to_csv (r'ASVspoof2019.LA.cm.dev.trl.csv', index=None)
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt')
>>> df
                     sno fname - -- type
0      LA_0069 LA_D_1047731 - - bonafide
1      LA_0069 LA_D_1105538 - - bonafide
2      LA_0069 LA_D_1125976 - - bonafide
3      LA_0069 LA_D_1293230 - - bonafide
4      LA_0069 LA_D_1340209 - - bonafide
...                                  ...
24839   LA_0078 LA_D_9924204 - A06 spoof
24840   LA_0078 LA_D_9931163 - A06 spoof
24841   LA_0078 LA_D_9935163 - A06 spoof
24842   LA_0078 LA_D_9944718 - A06 spoof
24843   LA_0078 LA_D_9967770 - A06 spoof

[24844 rows x 1 columns]
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ',usecols=['id','name','-''method','type'])
Traceback (most recent call last):
  File "<pyshell#22>", line 1, in <module>
    df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ',usecols=['id','name','-''method','type'])
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 1959, in __init__
    _validate_usecols_names(usecols, self.orig_names)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 1255, in _validate_usecols_names
    "columns expected but not found: {missing}".format(missing=missing)
ValueError: Usecols do not match columns, columns expected but not found: ['name', '-method', 'id', 'type']
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ')
>>> df
       LA_0069  LA_D_1047731  -  -.1  bonafide
0      LA_0069  LA_D_1105538  -    -  bonafide
1      LA_0069  LA_D_1125976  -    -  bonafide
2      LA_0069  LA_D_1293230  -    -  bonafide
3      LA_0069  LA_D_1340209  -    -  bonafide
4      LA_0069  LA_D_1376638  -    -  bonafide
...        ...           ... ..  ...       ...
24838  LA_0078  LA_D_9924204  -  A06     spoof
24839  LA_0078  LA_D_9931163  -  A06     spoof
24840  LA_0078  LA_D_9935163  -  A06     spoof
24841  LA_0078  LA_D_9944718  -  A06     spoof
24842  LA_0078  LA_D_9967770  -  A06     spoof

[24843 rows x 5 columns]
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ',header=0)
>>> df
       LA_0069  LA_D_1047731  -  -.1  bonafide
0      LA_0069  LA_D_1105538  -    -  bonafide
1      LA_0069  LA_D_1125976  -    -  bonafide
2      LA_0069  LA_D_1293230  -    -  bonafide
3      LA_0069  LA_D_1340209  -    -  bonafide
4      LA_0069  LA_D_1376638  -    -  bonafide
...        ...           ... ..  ...       ...
24838  LA_0078  LA_D_9924204  -  A06     spoof
24839  LA_0078  LA_D_9931163  -  A06     spoof
24840  LA_0078  LA_D_9935163  -  A06     spoof
24841  LA_0078  LA_D_9944718  -  A06     spoof
24842  LA_0078  LA_D_9967770  -  A06     spoof

[24843 rows x 5 columns]
>>> 
KeyboardInterrupt
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ',header=None)
>>> df
             0             1  2    3         4
0      LA_0069  LA_D_1047731  -    -  bonafide
1      LA_0069  LA_D_1105538  -    -  bonafide
2      LA_0069  LA_D_1125976  -    -  bonafide
3      LA_0069  LA_D_1293230  -    -  bonafide
4      LA_0069  LA_D_1340209  -    -  bonafide
...        ...           ... ..  ...       ...
24839  LA_0078  LA_D_9924204  -  A06     spoof
24840  LA_0078  LA_D_9931163  -  A06     spoof
24841  LA_0078  LA_D_9935163  -  A06     spoof
24842  LA_0078  LA_D_9944718  -  A06     spoof
24843  LA_0078  LA_D_9967770  -  A06     spoof

[24844 rows x 5 columns]
>>> df.to_csv (r'ASVspoof2019.LA.cm.dev.trl.csv', index=None)
Traceback (most recent call last):
  File "<pyshell#29>", line 1, in <module>
    df.to_csv (r'ASVspoof2019.LA.cm.dev.trl.csv', index=None)
  File "D:\python installation\lib\site-packages\pandas\core\generic.py", line 3228, in to_csv
    formatter.save()
  File "D:\python installation\lib\site-packages\pandas\io\formats\csvs.py", line 183, in save
    compression=self.compression,
  File "D:\python installation\lib\site-packages\pandas\io\common.py", line 399, in _get_handle
    f = open(path_or_buf, mode, encoding=encoding, newline="")
PermissionError: [Errno 13] Permission denied: 'ASVspoof2019.LA.cm.dev.trl.csv'
>>> df.to_csv (r'ASVspoof2019.LA.cm.dev.trl.csv', index=None)
>>> bonafide=df.loc[df[4]=='bonafide']
>>> bonafide
            0             1  2  3         4
0     LA_0069  LA_D_1047731  -  -  bonafide
1     LA_0069  LA_D_1105538  -  -  bonafide
2     LA_0069  LA_D_1125976  -  -  bonafide
3     LA_0069  LA_D_1293230  -  -  bonafide
4     LA_0069  LA_D_1340209  -  -  bonafide
...       ...           ... .. ..       ...
2543  LA_0108  LA_D_9561057  -  -  bonafide
2544  LA_0108  LA_D_9623720  -  -  bonafide
2545  LA_0108  LA_D_9841321  -  -  bonafide
2546  LA_0108  LA_D_9841868  -  -  bonafide
2547  LA_0108  LA_D_9899632  -  -  bonafide

[2548 rows x 5 columns]
>>> spoof=df.loc[df[4]=='spoof')
SyntaxError: invalid syntax
>>> spoof=df.loc(df[4]=='spoof')
Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    spoof=df.loc(df[4]=='spoof')
  File "D:\python installation\lib\site-packages\pandas\core\indexing.py", line 112, in __call__
    axis = self.obj._get_axis_number(axis)
  File "D:\python installation\lib\site-packages\pandas\core\generic.py", line 402, in _get_axis_number
    axis = cls._AXIS_ALIASES.get(axis, axis)
  File "D:\python installation\lib\site-packages\pandas\core\generic.py", line 1886, in __hash__
    " hashed".format(self.__class__.__name__)
TypeError: 'Series' objects are mutable, thus they cannot be hashed
s
>>> df
             0             1  2    3         4
0      LA_0069  LA_D_1047731  -    -  bonafide
1      LA_0069  LA_D_1105538  -    -  bonafide
2      LA_0069  LA_D_1125976  -    -  bonafide
3      LA_0069  LA_D_1293230  -    -  bonafide
4      LA_0069  LA_D_1340209  -    -  bonafide
...        ...           ... ..  ...       ...
24839  LA_0078  LA_D_9924204  -  A06     spoof
24840  LA_0078  LA_D_9931163  -  A06     spoof
24841  LA_0078  LA_D_9935163  -  A06     spoof
24842  LA_0078  LA_D_9944718  -  A06     spoof
24843  LA_0078  LA_D_9967770  -  A06     spoof

[24844 rows x 5 columns]
>>> spoof=df.loc(df[4]=='spoof')
Traceback (most recent call last):
  File "<pyshell#36>", line 1, in <module>
    spoof=df.loc(df[4]=='spoof')
  File "D:\python installation\lib\site-packages\pandas\core\indexing.py", line 112, in __call__
    axis = self.obj._get_axis_number(axis)
  File "D:\python installation\lib\site-packages\pandas\core\generic.py", line 402, in _get_axis_number
    axis = cls._AXIS_ALIASES.get(axis, axis)
  File "D:\python installation\lib\site-packages\pandas\core\generic.py", line 1886, in __hash__
    " hashed".format(self.__class__.__name__)
TypeError: 'Series' objects are mutable, thus they cannot be hashed
>>> df[0][4]
'LA_0069'
>>> df[0]
0        LA_0069
1        LA_0069
2        LA_0069
3        LA_0069
4        LA_0069
          ...   
24839    LA_0078
24840    LA_0078
24841    LA_0078
24842    LA_0078
24843    LA_0078
Name: 0, Length: 24844, dtype: object
>>> df
             0             1  2    3         4
0      LA_0069  LA_D_1047731  -    -  bonafide
1      LA_0069  LA_D_1105538  -    -  bonafide
2      LA_0069  LA_D_1125976  -    -  bonafide
3      LA_0069  LA_D_1293230  -    -  bonafide
4      LA_0069  LA_D_1340209  -    -  bonafide
...        ...           ... ..  ...       ...
24839  LA_0078  LA_D_9924204  -  A06     spoof
24840  LA_0078  LA_D_9931163  -  A06     spoof
24841  LA_0078  LA_D_9935163  -  A06     spoof
24842  LA_0078  LA_D_9944718  -  A06     spoof
24843  LA_0078  LA_D_9967770  -  A06     spoof

[24844 rows x 5 columns]
>>> df
             0             1  2    3         4
0      LA_0069  LA_D_1047731  -    -  bonafide
1      LA_0069  LA_D_1105538  -    -  bonafide
2      LA_0069  LA_D_1125976  -    -  bonafide
3      LA_0069  LA_D_1293230  -    -  bonafide
4      LA_0069  LA_D_1340209  -    -  bonafide
...        ...           ... ..  ...       ...
24839  LA_0078  LA_D_9924204  -  A06     spoof
24840  LA_0078  LA_D_9931163  -  A06     spoof
24841  LA_0078  LA_D_9935163  -  A06     spoof
24842  LA_0078  LA_D_9944718  -  A06     spoof
24843  LA_0078  LA_D_9967770  -  A06     spoof

[24844 rows x 5 columns]
>>> df[0][0]
'LA_0069'
>>> df[0]
0        LA_0069
1        LA_0069
2        LA_0069
3        LA_0069
4        LA_0069
          ...   
24839    LA_0078
24840    LA_0078
24841    LA_0078
24842    LA_0078
24843    LA_0078
Name: 0, Length: 24844, dtype: object
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ',header=['id','name','-','technique','type')
		   
SyntaxError: invalid syntax
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ',header=['id','name','-','technique','type'])
		   
Traceback (most recent call last):
  File "<pyshell#44>", line 1, in <module>
    df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ',header=['id','name','-','technique','type'])
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 1900, in __init__
    ParserBase.__init__(self, kwds)
  File "D:\python installation\lib\site-packages\pandas\io\parsers.py", line 1403, in __init__
    raise ValueError("header must be integer or list of integers")
ValueError: header must be integer or list of integers
>>> df=pd.read_csv('ASVspoof2019.LA.cm.dev.trl.txt',' ',names=['id','name','-','technique','type'])
		   
>>> df
		   
            id          name  - technique      type
0      LA_0069  LA_D_1047731  -         -  bonafide
1      LA_0069  LA_D_1105538  -         -  bonafide
2      LA_0069  LA_D_1125976  -         -  bonafide
3      LA_0069  LA_D_1293230  -         -  bonafide
4      LA_0069  LA_D_1340209  -         -  bonafide
...        ...           ... ..       ...       ...
24839  LA_0078  LA_D_9924204  -       A06     spoof
24840  LA_0078  LA_D_9931163  -       A06     spoof
24841  LA_0078  LA_D_9935163  -       A06     spoof
24842  LA_0078  LA_D_9944718  -       A06     spoof
24843  LA_0078  LA_D_9967770  -       A06     spoof

[24844 rows x 5 columns]
>>> df.to_csv (r'ASVspoof2019.LA.cm.dev.trl.csv', index=None)
		   
>>> bonafide=df.loc[df['type']=='bonafide']
		   
>>> bonafide
		   
           id          name  - technique      type
0     LA_0069  LA_D_1047731  -         -  bonafide
1     LA_0069  LA_D_1105538  -         -  bonafide
2     LA_0069  LA_D_1125976  -         -  bonafide
3     LA_0069  LA_D_1293230  -         -  bonafide
4     LA_0069  LA_D_1340209  -         -  bonafide
...       ...           ... ..       ...       ...
2543  LA_0108  LA_D_9561057  -         -  bonafide
2544  LA_0108  LA_D_9623720  -         -  bonafide
2545  LA_0108  LA_D_9841321  -         -  bonafide
2546  LA_0108  LA_D_9841868  -         -  bonafide
2547  LA_0108  LA_D_9899632  -         -  bonafide

[2548 rows x 5 columns]
>>> spoof=df.loc(df['type']=='spoof')
		   
Traceback (most recent call last):
  File "<pyshell#50>", line 1, in <module>
    spoof=df.loc(df['type']=='spoof')
  File "D:\python installation\lib\site-packages\pandas\core\indexing.py", line 112, in __call__
    axis = self.obj._get_axis_number(axis)
  File "D:\python installation\lib\site-packages\pandas\core\generic.py", line 402, in _get_axis_number
    axis = cls._AXIS_ALIASES.get(axis, axis)
  File "D:\python installation\lib\site-packages\pandas\core\generic.py", line 1886, in __hash__
    " hashed".format(self.__class__.__name__)
TypeError: 'Series' objects are mutable, thus they cannot be hashed
>>> df['type']
		   
0        bonafide
1        bonafide
2        bonafide
3        bonafide
4        bonafide
           ...   
24839       spoof
24840       spoof
24841       spoof
24842       spoof
24843       spoof
Name: type, Length: 24844, dtype: object
>>> df[0]
		   
Traceback (most recent call last):
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 2897, in get_loc
    return self._engine.get_loc(key)
  File "pandas\_libs\index.pyx", line 107, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 131, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1607, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1614, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#52>", line 1, in <module>
    df[0]
  File "D:\python installation\lib\site-packages\pandas\core\frame.py", line 2980, in __getitem__
    indexer = self.columns.get_loc(key)
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 2899, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas\_libs\index.pyx", line 107, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 131, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1607, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1614, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 0
>>> df.rows
		   
Traceback (most recent call last):
  File "<pyshell#53>", line 1, in <module>
    df.rows
  File "D:\python installation\lib\site-packages\pandas\core\generic.py", line 5179, in __getattr__
    return object.__getattribute__(self, name)
AttributeError: 'DataFrame' object has no attribute 'rows'
>>> df
		   
            id          name  - technique      type
0      LA_0069  LA_D_1047731  -         -  bonafide
1      LA_0069  LA_D_1105538  -         -  bonafide
2      LA_0069  LA_D_1125976  -         -  bonafide
3      LA_0069  LA_D_1293230  -         -  bonafide
4      LA_0069  LA_D_1340209  -         -  bonafide
...        ...           ... ..       ...       ...
24839  LA_0078  LA_D_9924204  -       A06     spoof
24840  LA_0078  LA_D_9931163  -       A06     spoof
24841  LA_0078  LA_D_9935163  -       A06     spoof
24842  LA_0078  LA_D_9944718  -       A06     spoof
24843  LA_0078  LA_D_9967770  -       A06     spoof

[24844 rows x 5 columns]
>>> df[:][0]
		   
Traceback (most recent call last):
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 2897, in get_loc
    return self._engine.get_loc(key)
  File "pandas\_libs\index.pyx", line 107, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 131, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1607, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1614, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#55>", line 1, in <module>
    df[:][0]
  File "D:\python installation\lib\site-packages\pandas\core\frame.py", line 2980, in __getitem__
    indexer = self.columns.get_loc(key)
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 2899, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas\_libs\index.pyx", line 107, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 131, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1607, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1614, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 0
>>> df[0][0]
		   
Traceback (most recent call last):
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 2897, in get_loc
    return self._engine.get_loc(key)
  File "pandas\_libs\index.pyx", line 107, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 131, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1607, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1614, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#56>", line 1, in <module>
    df[0][0]
  File "D:\python installation\lib\site-packages\pandas\core\frame.py", line 2980, in __getitem__
    indexer = self.columns.get_loc(key)
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 2899, in get_loc
    return self._engine.get_loc(self._maybe_cast_indexer(key))
  File "pandas\_libs\index.pyx", line 107, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\index.pyx", line 131, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1607, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1614, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 0
>>> df[:]
		   
            id          name  - technique      type
0      LA_0069  LA_D_1047731  -         -  bonafide
1      LA_0069  LA_D_1105538  -         -  bonafide
2      LA_0069  LA_D_1125976  -         -  bonafide
3      LA_0069  LA_D_1293230  -         -  bonafide
4      LA_0069  LA_D_1340209  -         -  bonafide
...        ...           ... ..       ...       ...
24839  LA_0078  LA_D_9924204  -       A06     spoof
24840  LA_0078  LA_D_9931163  -       A06     spoof
24841  LA_0078  LA_D_9935163  -       A06     spoof
24842  LA_0078  LA_D_9944718  -       A06     spoof
24843  LA_0078  LA_D_9967770  -       A06     spoof

[24844 rows x 5 columns]
>>> df.loc[0]
		   
id                LA_0069
name         LA_D_1047731
-                       -
technique               -
type             bonafide
Name: 0, dtype: object
>>> df.loc[0][0]
		   
'LA_0069'
>>> df.loc[0][4]
		   
'bonafide'
>>> df.loc[0]['spoof']
		   
Traceback (most recent call last):
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 4736, in get_value
    return libindex.get_value_box(s, key)
  File "pandas\_libs\index.pyx", line 51, in pandas._libs.index.get_value_box
  File "pandas\_libs\index.pyx", line 47, in pandas._libs.index.get_value_at
  File "pandas\_libs\util.pxd", line 98, in pandas._libs.util.get_value_at
  File "pandas\_libs\util.pxd", line 83, in pandas._libs.util.validate_indexer
TypeError: 'str' object cannot be interpreted as an integer

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#61>", line 1, in <module>
    df.loc[0]['spoof']
  File "D:\python installation\lib\site-packages\pandas\core\series.py", line 1068, in __getitem__
    result = self.index.get_value(self, key)
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 4744, in get_value
    raise e1
  File "D:\python installation\lib\site-packages\pandas\core\indexes\base.py", line 4730, in get_value
    return self._engine.get_value(s, k, tz=getattr(series.dtype, "tz", None))
  File "pandas\_libs\index.pyx", line 80, in pandas._libs.index.IndexEngine.get_value
  File "pandas\_libs\index.pyx", line 88, in pandas._libs.index.IndexEngine.get_value
  File "pandas\_libs\index.pyx", line 131, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\_libs\hashtable_class_helper.pxi", line 1607, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\_libs\hashtable_class_helper.pxi", line 1614, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'spoof'
>>>  bonafide=df.loc[df['type']=='bonafide']
		   
SyntaxError: unexpected indent
>>> bonafide=df.loc[df['type']=='bonafide']
		   
>>> bonafide
		   
           id          name  - technique      type
0     LA_0069  LA_D_1047731  -         -  bonafide
1     LA_0069  LA_D_1105538  -         -  bonafide
2     LA_0069  LA_D_1125976  -         -  bonafide
3     LA_0069  LA_D_1293230  -         -  bonafide
4     LA_0069  LA_D_1340209  -         -  bonafide
...       ...           ... ..       ...       ...
2543  LA_0108  LA_D_9561057  -         -  bonafide
2544  LA_0108  LA_D_9623720  -         -  bonafide
2545  LA_0108  LA_D_9841321  -         -  bonafide
2546  LA_0108  LA_D_9841868  -         -  bonafide
2547  LA_0108  LA_D_9899632  -         -  bonafide

[2548 rows x 5 columns]
>>>  spoof=df.loc[df['type']=='spoof']
		   
SyntaxError: unexpected indent
>>> spoof=df.loc[df['type']=='spoof']
		   
>>> spoof
		   
            id          name  - technique   type
2548   LA_0069  LA_D_1008730  -       A01  spoof
2549   LA_0069  LA_D_1034049  -       A01  spoof
2550   LA_0069  LA_D_1048723  -       A01  spoof
2551   LA_0069  LA_D_1067573  -       A01  spoof
2552   LA_0069  LA_D_1091909  -       A01  spoof
...        ...           ... ..       ...    ...
24839  LA_0078  LA_D_9924204  -       A06  spoof
24840  LA_0078  LA_D_9931163  -       A06  spoof
24841  LA_0078  LA_D_9935163  -       A06  spoof
24842  LA_0078  LA_D_9944718  -       A06  spoof
24843  LA_0078  LA_D_9967770  -       A06  spoof

[22296 rows x 5 columns]
>>> spoof[0:200]
		   
           id          name  - technique   type
2548  LA_0069  LA_D_1008730  -       A01  spoof
2549  LA_0069  LA_D_1034049  -       A01  spoof
2550  LA_0069  LA_D_1048723  -       A01  spoof
2551  LA_0069  LA_D_1067573  -       A01  spoof
2552  LA_0069  LA_D_1091909  -       A01  spoof
...       ...           ... ..       ...    ...
2743  LA_0069  LA_D_5423215  -       A01  spoof
2744  LA_0069  LA_D_5461745  -       A01  spoof
2745  LA_0069  LA_D_5505761  -       A01  spoof
2746  LA_0069  LA_D_5508053  -       A01  spoof
2747  LA_0069  LA_D_5509091  -       A01  spoof

[200 rows x 5 columns]
>>> sp=spoof[0:200]
		   
>>> bo=bonafide[0:200]
		   
>>> sp['name']
		   
2548    LA_D_1008730
2549    LA_D_1034049
2550    LA_D_1048723
2551    LA_D_1067573
2552    LA_D_1091909
            ...     
2743    LA_D_5423215
2744    LA_D_5461745
2745    LA_D_5505761
2746    LA_D_5508053
2747    LA_D_5509091
Name: name, Length: 200, dtype: object
>>> sp
		   
           id          name  - technique   type
2548  LA_0069  LA_D_1008730  -       A01  spoof
2549  LA_0069  LA_D_1034049  -       A01  spoof
2550  LA_0069  LA_D_1048723  -       A01  spoof
2551  LA_0069  LA_D_1067573  -       A01  spoof
2552  LA_0069  LA_D_1091909  -       A01  spoof
...       ...           ... ..       ...    ...
2743  LA_0069  LA_D_5423215  -       A01  spoof
2744  LA_0069  LA_D_5461745  -       A01  spoof
2745  LA_0069  LA_D_5505761  -       A01  spoof
2746  LA_0069  LA_D_5508053  -       A01  spoof
2747  LA_0069  LA_D_5509091  -       A01  spoof

[200 rows x 5 columns]
>>> sp
		   
           id          name  - technique   type
2548  LA_0069  LA_D_1008730  -       A01  spoof
2549  LA_0069  LA_D_1034049  -       A01  spoof
2550  LA_0069  LA_D_1048723  -       A01  spoof
2551  LA_0069  LA_D_1067573  -       A01  spoof
2552  LA_0069  LA_D_1091909  -       A01  spoof
...       ...           ... ..       ...    ...
2743  LA_0069  LA_D_5423215  -       A01  spoof
2744  LA_0069  LA_D_5461745  -       A01  spoof
2745  LA_0069  LA_D_5505761  -       A01  spoof
2746  LA_0069  LA_D_5508053  -       A01  spoof
2747  LA_0069  LA_D_5509091  -       A01  spoof

[200 rows x 5 columns]
>>> sp['name']
		   
2548    LA_D_1008730
2549    LA_D_1034049
2550    LA_D_1048723
2551    LA_D_1067573
2552    LA_D_1091909
            ...     
2743    LA_D_5423215
2744    LA_D_5461745
2745    LA_D_5505761
2746    LA_D_5508053
2747    LA_D_5509091
Name: name, Length: 200, dtype: object
>>> 
