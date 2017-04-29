import csv
import os
import pandas as pd
import numpy as np
import datetime as dt

import re
ticker_regex = re.compile('.+_(?P<ticker>.+)\.csv')

get_ticker = lambda x : ticker_regex.match(x).groupdict()['ticker']
ret = lambda x,y: np.log(y/x) #Log return 
zscore = lambda x:(x - x.mean()) / x.std() # zscore

directory = os.path.join("./","data/daily/")

for root,dirs,files in os.walk(directory):
	
	Final = pd.DataFrame()
	
	for file in files:
		filepath = directory + file

		D = pd.read_csv(filepath, header=None, names=['UNK','o','h','l','c','v'])

		# D = df[-2933:]

		D.index = pd.to_datetime(D.index, format='%Y%m%d')

		Res = pd.DataFrame()
		ticker = get_ticker(filepath)

		Res['c_2_o'] = zscore(ret(D.o,D.c))
		Res['h_2_o'] = zscore(ret(D.o,D.h))
		Res['l_2_o'] = zscore(ret(D.o,D.l))
		Res['c_2_h'] = zscore(ret(D.h,D.c))
		Res['h_2_l'] = zscore(ret(D.h,D.l))
		Res['c1_c0'] = ret(D.c,D.c.shift(-1)).fillna(0) #Tommorows return 
		Res['vol'] = zscore(D.v)
		Res['ticker'] = ticker

		Final = Final.append(Res)


	pivot_columns = Final.columns[:-1]


	P = Final.pivot_table(index=Final.index, columns='ticker', values=pivot_columns)
	


	mi = P.columns.tolist()
	new_ind = pd.Index(e[1] +'_' + e[0] for e in mi)
	P.columns = new_ind
	P = P.sort(axis=1)


	# 1998-01-02
	start = P.index.searchsorted(dt.datetime(2002, 1, 1))
	# 2013-08-09
	end = P.index.searchsorted(dt.datetime(2012, 12, 31)) 

	P = P.ix[start:end]

	clean_and_flat = P.dropna(1)

	clean_and_flat.to_csv('./data/full.csv')

	# target_cols = list(filter(lambda x: 'c1_c0' in x, clean_and_flat.columns.values))
	# input_cols  = list(filter(lambda x: 'c1_c0' not in x, clean_and_flat.columns.values))

	# InputDF = clean_and_flat[input_cols]
	# TargetDF = clean_and_flat[target_cols]

	print 'Symbols: ', len(clean_and_flat.columns) / 7
