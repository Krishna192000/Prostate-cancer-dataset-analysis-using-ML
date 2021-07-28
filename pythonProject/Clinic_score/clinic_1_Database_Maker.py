import pandas as pd
import pickle5 as pickle
import openpyxl
import numpy as np
import os

def Database_Maker():
    print("------------Starting run of clinic_1_Database_Maker------------\n")
    print("This file will combine & filter data from original datasets: \nprad_tcga_clinical_data.xlsx & prad_tcga_genes.xlsx\n")
    print("Dataset creation")

    clinical_data = pd.read_excel('ProstateCancerDataset/prad_tcga_clinical_data.xlsx')
    print("Read: " , clinical_data.shape)
    # 5 rows deducted because these patients are not present on the other dataset
    #clinical_data = clinical_data.drop(['TCGA-HC-7741', 'TCGA-HC-8212', 'TCGA-KC-A4BO', 'TCGA-V1-A8MJ', 'TCGA-YL-A8SF'])
    clinical_data = clinical_data.drop([214, 228, 298, 372, 460])
    print('Dropped 5 rows that are missing in genes data. \nNew Length: ' , clinical_data.shape)
    #clinical_data = clinical_data.head(494)
    #print("clinal new length" + clinical_data.columns.len)

    #Take clinic so that we can add it in the new dataset {five classes: 6, 7, 8, 9, 10}
    clinTscore = clinical_data.loc[: , "CLIN_T_STAGE"]

    # Read data from gene dataset
    data_path = "ProstateCancerDataset/prad_tcga_genes.xlsx"

    # For faster access of data I have used pickle
    gene_data = pd.read_excel(data_path, engine="openpyxl")
    pickle.dump(gene_data, open('Clinic_score/gene_exp.pickle', 'wb'))

    new_dataset = []
    with open('Clinic_score/gene_exp.pickle','rb') as f:
        df2 = pickle.load(f)
        columns = list(df2[df2.columns[0]])

    for i in range(1, len(df2.columns)):
        #each column from prad_tcga_genes.xlsx data, 60483 columns in total for 60483 genes
        #Each row in the prad_tcga_genes.xlsx is now is stacked as columns in new_dataset
        new_dataset.append(df2.iloc[:,i])

    index = list(clinical_data[clinical_data.columns[0]])

    new_dataset = np.array(new_dataset)

    new_df = pd.DataFrame(data=new_dataset, index=index, columns=columns)

    #Add clinic as another column in new_dataset file
    new_df['clinTscore'] = np.array(clinTscore)

    new_df.to_csv('Clinic_score/clinic_Data_1_Predict_Tscore.csv', sep=',')

    #Cleanup
    os.remove("Clinic_score/gene_exp.pickle")

    print("Done creating dataset. Please find the file clinic_Data_1_Predict_Tscore.csv")
    print("\n------------Ending run of clinic_1_Database_Maker------------")
