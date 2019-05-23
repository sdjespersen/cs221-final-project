import sqlite3
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn as sk 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sqlite3 import Error
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None
 
def main():
    database = "../input/FPA_FOD_20170508.sqlite"
 
    # create a database connection
    conn = create_connection(database)
    with conn:
        df = pd.read_sql_query("SELECT STAT_CAUSE_CODE,FIRE_YEAR,LATITUDE,LONGITUDE,DISCOVERY_DATE,FIRE_SIZE FROM 'Fires' WHERE STATE='CA' ", conn)
        #df = df[df.CONT_DATE.notna()]
        df3 = pd.read_sql_query("""SELECT STAT_CAUSE_CODE, FIRE_YEAR, LATITUDE, LONGITUDE, DISCOVERY_DATE,FIRE_SIZE, STATE
        FROM 'Fires'""", conn)
        print(len(df3))
        df3 = pd.concat([df3,pd.get_dummies(df3['STATE'], prefix='STATE')],axis=1).drop(['STATE'],axis=1)
        #remove causes 9 and 13 from dataset
        df3 = df3[(df3['STAT_CAUSE_CODE'] != 9) & (df3['STAT_CAUSE_CODE'] != 13)]
        #shuffle dataset
        df3 = shuffle(df3)
        
        y = df3.iloc[:,0]
        print(type(y))
        x = df3.iloc[:,1:]
        #print(df.head())
        #print(df.STAT_CAUSE_CODE.unique())
        df2 = pd.read_sql_query("SELECT DISTINCT STATE FROM 'Fires'", conn)
        #print(df2)
        size = len(y)
        split = int(size*0.7)
        y_train = y.iloc[:split]
        X_train = x.iloc[:split]
        y_test = y.iloc[split:]
        X_test = x.iloc[split:]
        """
        LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)  
        LR.predict(X_test)  
        print(round(LR.score(X_test,y_test), 4))
        """
        """
        RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)  
        RF.predict(X_test)  
        print(round(RF.score(X_test, y_test), 4))
        #0.3845 with no shuffle, and all categories, n_estimators = 1000, max_depth = 10, random_state = 0
        """
        mlp = MLPClassifier()
        mlp.fit(X_train, y_train)
        from sklearn.metrics import balanced_accuracy_score
        y_pred=mlp.predict(X_test)
        print(balanced_accuracy_score(y_test,y_pred))
        #0.3087
if __name__ == '__main__':
    main()