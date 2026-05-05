import wfdb
import numpy as np
import pandas as pd


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def array_to_dict(data):
    matrix = []
    for i in range(12):
        matrix.append(data[:,i]*1000)
    return np.array(matrix)

def normalization(Z):
    mini=Z.min()
    maxi=Z.max()
    return(-1+((Z-mini)*(2))/(maxi-mini))


def load_ptb_xl_data(path="physionet/ptb_xl/raw/physionet.org/files/ptb-xl/1.0.3/"):
    df = pd.read_csv(path+"ptbxl_database.csv")
    matrix_ecg = []
    matrix_lead = []
    matrix_ecg_id = []
    for i in range(1,len(df)+1):
        try:
            tracks_num = array_to_dict(wfdb.rdsamp(path+df["filename_hr"][df["ecg_id"] == i].reset_index(drop=True)[0])[0])
            for l in range(12):
                tracks_num_norm = normalization(tracks_num[l,:])
                if np.isnan(tracks_num_norm).any():
                    continue
                else:
                    matrix_ecg.append(tracks_num_norm)
                    matrix_lead.append(l)
                    matrix_ecg_id.append(i)
        except KeyError:
            pass
        

    matrix_ecg = np.array(matrix_ecg)
    matrix_ecg = np.expand_dims(matrix_ecg, axis=1)

    matrix_lead = np.array(matrix_lead)
    matrix_ecg_id = np.array(matrix_ecg_id)   


    np.random.seed(42)
    indices = np.arange(len(matrix_ecg))
    np.random.shuffle(indices)
    matrix_ecg = matrix_ecg[indices]
    matrix_lead = np.array(matrix_lead)[indices]
    matrix_ecg_id = np.array(matrix_ecg_id)[indices] 


    train_data = matrix_ecg[:int(0.8*len(matrix_ecg))]
    test_data = matrix_ecg[int(0.8*len(matrix_ecg)):int(0.9*len(matrix_ecg))]
    val_data = matrix_ecg[int(0.9*len(matrix_ecg)):]

    train_id = matrix_ecg_id[:int(0.8*len(matrix_ecg_id))]
    test_id = matrix_ecg_id[int(0.8*len(matrix_ecg_id)):int(0.9*len(matrix_ecg_id))]
    val_id = matrix_ecg_id[int(0.9*len(matrix_ecg_id)):]

    train_leads = matrix_lead[:int(0.8*len(matrix_lead))]
    test_leads = matrix_lead[int(0.8*len(matrix_lead)):int(0.9*len(matrix_lead))]
    val_leads = matrix_lead[int(0.9*len(matrix_lead)):]

    return(train_data, test_data, val_data, train_id, test_id, val_id, train_leads, test_leads, val_leads)