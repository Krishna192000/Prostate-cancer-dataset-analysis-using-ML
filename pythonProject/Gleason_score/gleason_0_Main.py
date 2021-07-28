from gleason_1_Database_Maker import Database_Maker
from gleason_2_Low_Variance import Low_Variance
from gleason_3_FS_RF import FS_RF
from gleason_4_Final_Score import Final_Score

if __name__ == '__main__':
    print("\n************ Gleason Score Prediction ************\n")
    Database_Maker()
    Low_Variance()
    FS_RF()
    Final_Score()