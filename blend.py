import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

b1 = pd.read_csv('./blending/lgsub_02242LB.csv').rename(columns={'deal_probability':'dp1'})
b2 = pd.read_csv('./blending/xgb_tfidf0.21864_02241LB.csv').rename(columns={'deal_probability':'dp2'})
b3 = pd.read_csv('./blending/keras_keras_60k_d300_10fold_avg_02233LB.csv').rename(columns={'deal_probability':'dp3'})

b1 = pd.merge(b1, b2, how='left', on='item_id')
b1 = pd.merge(b1, b3, how='left', on='item_id')

b1['deal_probability'] = (b1['dp1'] * 1/3) + (b2['dp2'] * 1/3) + (b3['dp3'] * 1/3)
b1[['item_id','deal_probability']].to_csv('./blending/blend_xgb_lbg_cnn_update.csv', index=False)