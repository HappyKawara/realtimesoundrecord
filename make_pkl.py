import numpy as np
import pickle


mfccs_t = [np.full(12,0)]
ls = [10]
a = [mfccs_t,ls]
with open('np_data','wb') as f:
    pickle.dump(a,f)
with open ('fit.pkl','wb') as f:
    pickle.dump(([],[]),f)
