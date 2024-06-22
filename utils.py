from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import uuid


class FeatureCreationAPI(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selected_columns = [
            "id", "name", "sex", "dob", "race",
            "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
            "c_case_number", "c_charge_degree", "c_charge_desc", "c_offense_date",
            "c_arrest_date", "c_jail_in"
        ]

    def fit(self, X, y=None):
        # This method doesn't need to do anything for this transformer
        return self

    def transform(self, X):
        # Make a copy of the dataframe to avoid modifying the original input
        df = X.copy()

        # Convert dates to datetime
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'], errors='coerce')

        # Age Calculation as an integer
        df['age_at_arrest'] = (df['c_jail_in'] - df['dob']).dt.days // 365

        # Age group
        age_bins = [0, 24, 34, 44, 54, 100]
        age_labels = ['<25', '25-34', '35-44', '45-54', '55+']
        df['age_group'] = pd.cut(df['age_at_arrest'], bins=age_bins,
                                 labels=age_labels, right=False)

        # Replace c_charge_degree
        df['c_charge_degree'] = df['c_charge_degree'].str.replace(
            'F', 'Felony')
        df['c_charge_degree'] = df['c_charge_degree'].str.replace(
            'M', 'Misdemeanor')
        df['c_charge_degree'] = df['c_charge_degree'].fillna('Missing')

        #  weight_race for each class:
        #race              weights_race
        #African-American  0.326498        3121
        #Caucasian         0.489198        2083
        #Hispanic          1.866300         546
        #Other             3.135385         325
        #Asian             39.192308         26
        #Native American   78.384615         13
        
        weights = {
            'African-American' : 0.33,
            'Caucasian': 0.49,
            'Hispanic': 1.87,
            'Other': 3.14,
            'Asian': 39.19,
            'Native American': 78.38
        }
        
        # based on the column race i want to create another column with the weights_race using the weights dictionary
        df['weights_race'] = df['race'].map(weights)

        # Drop unnecessary columns
        df = df.drop(['dob', 'c_case_number', 'c_charge_desc', 'c_offense_date',
                     'c_arrest_date', 'name', 'c_jail_in', 'age_at_arrest'], axis=1)

        return df

def generate_unique_id():
    # Function to generate a unique numeric ID (you can customize as needed)
    # Example: using a counter or any other method to ensure uniqueness
    # Here we can use a UUID and hash it to an integer for simplicity
    return int(uuid.uuid4().int & (1<<25)-1)