import pandas as pd
import numpy as np
import os
import data as dt

import pickle

import tensorflow as tf
from tensorflow import keras
import models


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.join('data')



def features_nonSeq(start_year, n_years, regress=False, usePrevLabels=False, useAggregates=False):
    test_year = start_year + n_years
    years = np.arange(start_year, test_year)
    
    #load aggregates, totals, and normalizers
    cbsa_means = dt.load_means(os.path.join(DATA_DIR, 'cleaned', 'cbsa-means-dummy.pkl'))
    counts = dt.load_admission_counts()
    pops = dt.load_populations()


    X_train, Y_train, X_test, Y_test = [],[],[],[]

    for key in cbsa_means.keys():
        curr_year = key[1]
        cbsa = key[0]
        features = []
        if key[1] in years or key[1]==test_year:
            if useAggregates:
                features.extend(cbsa_means[key])
            

            if usePrevLabels:
                prev_labels = [-1]*(test_year-2000)
                for i in range(0, len(prev_labels)-1):
                    yr = 2001+i
                    if yr < curr_year:
                        prev_label = int( counts[(cbsa, yr)] > counts[(cbsa, yr - 1)])
                        prev_labels[i] = prev_label
                        yr += 1
                features.extend(prev_labels)
        
        if regress:
            label = counts[key]/pops[cbsa]
        else:
            label = int( counts[key] > counts[(key[0], key[1]-1)] )
        
        if curr_year == test_year:
            X_test.append(features)
            Y_test.append(label)
        else:
            X_train.append(features)
            Y_train.append(label)
        
    X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

    return X_train, Y_train, X_test, Y_test


def features_seq(start_year, n_years, regress=False, usePrevLabels=False, useAggregates=False, prependPatients=False):
    test_year = start_year + n_years
    years = np.arange(start_year, test_year)
    
    #load aggregates, totals, and normalizers
    cbsa_means = dt.load_means(os.path.join(DATA_DIR, 'cleaned', 'cbsa-means-dummy.pkl'))
    counts = dt.load_admission_counts()
    pops = dt.load_populations()
    cbsas = list(set([key[0] for key in cbsa_means.keys()]))
    records = dt.load_clean_dummy()
    records = records[records.YEAR.isin(years)]


    X_train, Y_train, X_test, Y_test = [],[],[],[]

    for cbsa in cbsas:
        seq = []
        for yr in years:
            key = (cbsa, yr)
            features = []
            if useAggregates:
                features.extend(cbsa_means[key])
            if usePrevLabels:
                past_label = int ( counts[(cbsa,yr-1)] > counts[(cbsa, yr-2)])
                features.append(past_label)
            
            seq.append(features)
        
        label = int( counts[(cbsa, test_year-1)] > counts[(cbsa, test_year - 2)])
        if not prependPatients:
            X_train.append(seq)
            Y_train.append(label)
        else:
            cbsaRecords = records[records.CBSA10 == cbsa]
            for idx, row in cbsaRecords.iterrows():
                patient_vec = list(np.array(row))
                newSeq = [patient_vec + seq[i] for i in range(0, len(seq))]
                
                X_train.append(newSeq)
                Y_train.append(label)
                    

        
        seq = []
        for yr in years:
            yr = yr+1
            key = (cbsa,yr)
            features = []
            if useAggregates:
                features.extend(cbsa_means[key])
            if usePrevLabels:
                past_label = int ( counts[(cbsa,yr-1)] > counts[(cbsa, yr-2)])
                features.append(past_label)
            seq.append(features)
        
        label = int( counts[(cbsa, test_year)] > counts[(cbsa, test_year - 1)])
        
        if not prependPatients:
            X_test.append(seq)
            Y_test.append(label)
        else:
            newSeq = [seq[i] + seq[i] for i in range(0, len(seq))] #dummy to simulate average patient and maintain proper shape
            X_test.append(newSeq)
            Y_test.append(label)

            
    
    X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
    return X_train, Y_train, X_test, Y_test


def train_nonSeq(model, data, regress=False, epochs=50):
    X_train, Y_train, X_test, Y_test = data

    if model=='svm':
        clf = SVC(kernel='rbf', C=10000, degree=2, tol=1e-4)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print(metrics.accuracy_score(Y_test, Y_pred))
    elif model == 'logreg':
        clf = LogisticRegression(tol=1e-5, C=1e-3)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        print(metrics.accuracy_score(Y_test, Y_pred))
    elif model == 'main':
        m = models.baseNN(X_train.shape[-1], 1, regress)
        if regress:
            m.compile('adam', loss='mean_absolute_percentage_error')
        else:
            m.compile('adam', loss='binary_crossentropy', metrics=['acc'])
        
        m.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)
        print(m.evaluate(X_test, Y_test))
    else:
        print("Invalid model name!")

def train_seq(model, data, regress=False, epochs=50):
    X_train, Y_train, X_test, Y_test = data
    if model=='main':
        m = models.seqNN(X_train.shape[-1],X_train.shape[-2], 1, regress)
        if regress:
            m.compile('adam', loss='mean_absolute_percentage_error')
        else:
            m.compile('adam', loss='binary_crossentropy', metrics=['acc'])
        
        m.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs)
        print(m.evaluate(X_test, Y_test))
    else:
        print("Invalid model name!")



def exp_nonSeq(start_year, n_years, useAggregates, usePrevLabels):
    data = features_nonSeq(start_year, n_years, useAggregates=useAggregates, usePrevLabels=usePrevLabels)

    models = ['svm', 'logreg', 'main']
    for m in models:
        print(m)
        train_nonSeq(m, data)


def exp_seq(start_year, n_years, useAggregates, usePrevLabels, prependPatients):
    data = features_seq(start_year, n_years, useAggregates=useAggregates, usePrevLabels=usePrevLabels, prependPatients=prependPatients)
    train_nonSeq('main', data)


def main(nonSeq=True, seq=True):
    #Exp 1 - non-sequential
    if nonSeq:
        print("=== Non-Sequential ===")
        #n=5
        n = 5
        start_years = np.arange(2001, 2016-n+1)
        print("N={}".format(n))
        for start_year in start_years:
            print("Aggregates only:")
            exp_nonSeq(start_year, n, True, False)
            print("Past-labels only:")
            exp_nonSeq(start_year, n, False, True)
            print("Aggregates + Past-Labels:")
            exp_nonSeq(start_year, n, True, True)
        n = 10
        start_years = np.arange(2001, 2016-n+1)
        print("N={}".format(n))
        for start_year in start_years:
            print("Aggregates only:")
            exp_nonSeq(start_year, n, True, False)
            print("Past-labels only:")
            exp_nonSeq(start_year, n, False, True)
            print("Aggregates + Past-Labels:")
            exp_nonSeq(start_year, n, True, True)
        n = 15
        start_years = np.arange(2001, 2016-n+1)
        print("N={}".format(n))
        for start_year in start_years:
            print("Aggregates only:")
            exp_nonSeq(start_year, n, True, False)
            print("Past-labels only:")
            exp_nonSeq(start_year, n, False, True)
            print("Aggregates + Past-Labels:")
            exp_nonSeq(start_year, n, True, True)

    if seq:
        print("=== Sequential ===")
        #Exp 2 - Sequential
        n = 5
        start_years = np.arange(2001, 2016-n+1)
        print("N={}".format(n))
        for start_year in start_years:
            print("Aggregates only (no patient):")
            exp_seq(start_year, n, True, False, False)
            print("Aggregates only (patient):")
            exp_seq(start_year, n, True, False, True)


            print("Past-labels only (no patient):")
            exp_seq(start_year, n, False, True, False)
            print("Past-labels only (patient):")
            exp_seq(start_year, n, False, True, True)

            print("Aggregates + Past-Labels (no patient):")
            exp_seq(start_year, n, True, True, False)
            print("Aggregates + Past-Labels (patient):")
            exp_seq(start_year, n, True, True, True)

        n = 10
        start_years = np.arange(2001, 2016-n+1)
        print("N={}".format(n))
        for start_year in start_years:
            print("Aggregates only (no patient):")
            exp_seq(start_year, n, True, False, False)
            print("Aggregates only (patient):")
            exp_seq(start_year, n, True, False, True)


            print("Past-labels only (no patient):")
            exp_seq(start_year, n, False, True, False)
            print("Past-labels only (patient):")
            exp_seq(start_year, n, False, True, True)

            print("Aggregates + Past-Labels (no patient):")
            exp_seq(start_year, n, True, True, False)
            print("Aggregates + Past-Labels (patient):")
            exp_seq(start_year, n, True, True, True)

        n = 15
        start_years = np.arange(2001, 2016-n+1)
        print("N={}".format(n))
        for start_year in start_years:
            print("Aggregates only (no patient):")
            exp_seq(start_year, n, True, False, False)
            print("Aggregates only (patient):")
            exp_seq(start_year, n, True, False, True)


            print("Past-labels only (no patient):")
            exp_seq(start_year, n, False, True, False)
            print("Past-labels only (patient):")
            exp_seq(start_year, n, False, True, True)

            print("Aggregates + Past-Labels (no patient):")
            exp_seq(start_year, n, True, True, False)
            print("Aggregates + Past-Labels (patient):")
            exp_seq(start_year, n, True, True, True)

if __name__ == '__main__':
    main(nonSeq=True, seq=False)
