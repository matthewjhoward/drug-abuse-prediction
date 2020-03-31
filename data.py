import pandas as pd
import numpy as np
import pickle
import numpy
import os

from census import Census
from us import states

DATA_DIR = os.path.join('data')


def clean_teds_csv(filename='data/cleaned/teds-cleaned.csv'):
    pass

def dummy_clean_teds(df, filename='data/cleaned/teds-cleaned-dummy.csv'):
    all_cols = list(df.columns)
    dummy_cols = list(all_cols)
    non_dummy = ['CASEID', 'STFIPS', 'CBSA10', 'YEAR']
    for col in non_dummy:
        dummy_cols.remove(col)

    data_dummy = pd.get_dummies(df, columns=dummy_cols)
    data_dummy.to_csv(filename, index=False)

def load_clean_data():
    data = pd.read_csv('data/cleaned/teds-cleaned.csv', dtype=np.int64)
    return data

def load_clean_dummy():
    data = pd.read_csv('data/cleaned/teds-cleaned-dummy.csv')
    return data


def save_county_features(df, filename):
    drop_cols = ['CASEID', 'STFIPS', 'YEAR', 'CBSA10']
    years = sorted(df.YEAR.unique())
    cbsa_year_features = {}
    for year in years:
        year_df = df[df.YEAR == year]
        cbsas = year_df.CBSA10.unique()
        for cbsa in cbsas:
            means = year_df[year_df.CBSA10==cbsa].mean().drop(drop_cols)
            cbsa_year_features[(cbsa, year)]=np.array(means)
    
    with open(filename, 'wb') as output:
        pickle.dump(cbsa_year_features, output)
    
    return cbsa_year_features

def load_means(filename):
    cbsa_means = {}
    with open(filename, 'rb') as fin:
        cbsa_means = pickle.load(fin)
    return cbsa_means

    

def save_admission_counts(data):
    years = sorted(data.YEAR.unique())
    cbsa_year_counts = {}
    for year in years:
        counts = data[data.YEAR == year].CBSA10.value_counts()
        for cbsa, count in counts.iteritems():
            cbsa_year_counts[(cbsa,year)]=count
    
    with open('data/cleaned/admission-counts.pkl', 'wb') as fout:
        pickle.dump(cbsa_year_counts, fout)
    return cbsa_year_counts

def load_admission_counts():
    admissions = {}
    with open('data/cleaned/admission-counts.pkl', 'rb') as fin:
        admissions = pickle.load(fin)
    return admissions

def save_cbsa2fips(data):
    cbsa_fips = pd.read_excel(os.path.join(DATA_DIR, 'raw', "CBSA-FIPS.xls"), header=2,nrows=1882,convert_float=False, dtype='object')

    cbsa2fips = {}
    all_cbsas = data.CBSA10.unique()
    for idx,row in cbsa_fips.iterrows():
        cb = int(row['CBSA Code'])
        if cb in all_cbsas:
            fips = fix_fips(row['FIPS State Code'], row['FIPS County Code'])
            if not (cb in cbsa2fips):
                cbsa2fips[cb] = [fips]
            else:
                cbsa2fips[cb].append(fips)

    with open('data/cleaned/cbsa2fips.pkl', 'wb') as fout:
        pickle.dump(cbsa2fips, fout)
    
    return cbsa2fips

def load_cbsa2fips():
    cbsa2fips = {}
    with open('data/cleaned/cbsa2fips.pkl', 'rb') as fin:
        cbsa2fips = pickle.load(fin)
    return cbsa2fips

#must have saved cbsa2fips first
def save_populations():
    CENSUS_API_KEY = "846367a9aca0d09de354714a8eb90669e7ec5837"
    c = Census(CENSUS_API_KEY, year=2010)
    pop_results = c.sf1.state_county('P001001', Census.ALL,Census.ALL)

    fips_pops = {}
    for p in pop_results:
        fips = p['state']+p['county']
        pop = int(p['P001001'])
        fips_pops[fips]=pop
        
    cbsa_pops = {}
    cbsa2fips = load_cbsa2fips()
    for cb in cbsa2fips:
        fips = cbsa2fips[cb]
        s = 0
        for f in fips:
            s += fips_pops[f]
        cbsa_pops[cb]=s 

    with open('data/cleaned/cbsa-populations-2010.pkl', 'wb') as fout:
        pickle.dump(cbsa_pops, fout)
    
    return cbsa_pops

def load_populations():
    pops = {}
    with open('data/cleaned/cbsa-populations-2010.pkl', 'rb') as fin:
        pops = pickle.load(fin)
    return pops

def fix_fips(state, county):
    state = str(state)
    county = str(county)
    if len(state)<2:
        state = "0"*(2-len(state))+state
    if len(county)<3:
        county = "0"*(3-len(county))+county
    
    return state + county
    




