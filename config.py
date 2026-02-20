#Dir Path
Data_dir = "C:/Users/HP/Desktop/Fraud Detection/"
Model = "C:/Users/HP/Desktop/Model_output/"

#Training Data
Train_folds = Data_dir + "train_folds.csv"
Train_transaction = Data_dir + "train_transaction.csv"
Train_identity = Data_dir + "train_identity.csv"

#Test Data
Test_transaction = Data_dir + "test_transaction.csv"
Test_identity = Data_dir + "test_identity.csv"
Test_folds = Data_dir + "test_folds.csv"

#Categorical Features
Categorical_Features = [
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3",
    "M4", "M5", "M6", "M7", "M8", "M9", "DeviceType", "DeviceInfo", "id_12",
    "id_13", "id_14", "id_15", "id_16", "id_17",
    "id_18", "id_19", "id_20", "id_21", "id_22", "id_23",
    "id_24", "id_25", "id_26", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_32", "id_33", "id_34", "id_35",
    "id_36", "id_37", "id_38"
]
