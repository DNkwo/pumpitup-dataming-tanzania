import pandas as pd

def preprocess_data(data):
    #reduce cardinality function (top n most categories, label rest as 'other')
    def reduce_cardinality(df, column, n=15):
        #grabs top n most categories
        top_n_categories = df[column].value_counts().nlargest(n).index
        #replace the rest with 'other'
        df[column] = df[column].where(df[column].isin(top_n_categories), other='Other')
        return df

    #reducing cardinalities in selected features
    for feature in ['funder', 'installer', 'scheme_name', 'ward', "lga"]:
        data = reduce_cardinality(data, feature)

    #dropping more categories that do not seem so useful (e.g too many missing features, not relevant)
    #-----------------------------------------------------------------------
    #wpt_name - doubt arbritary names will affect predictive power
    #subvillage - doubt arbritary names will affect predictive power
    #recorded_by - mostly identical rows, probably wont affect predictive power
    #num_private - all identical rows
    #waterpoint_type_group - basically the same as 'waterpoint_type'
    #quantity_group - basically the same as 'quantity'
    #payment_type - basically the same as 'payment'
    #extraction_type_group - basically same as 'extraction_type'
    #water_quality - redundant as 'quality_group' is just a generalised form, mostly the same, so can be removed
    #------------------------------------------------------------------------
    data.drop(['wpt_name', 'subvillage', "recorded_by",
            "num_private", "waterpoint_type_group", "quantity_group", "payment_type",
            "extraction_type_group", "water_quality"], axis=1, inplace=True)


    # converting 'date_recorded' into more useful formats
    data['date_recorded'] = pd.to_datetime(data['date_recorded'])
    #probably sufficient to only include the year recorded, we reduce features this way
    data['year_recorded'] = data['date_recorded'].dt.year
    # drop original 'date_recorded' column
    data.drop('date_recorded', axis=1, inplace=True)
    
    
    #-----------OMITTED CODE (manually completed instead)--------------#
    #correlation-based feature elimination on numerical features
    # corr_matrix = data[numerical_features].corr().abs()

    # Select upper triangle of correlation matrix
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find feature columns with correlation of greater than 0.95
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    # Drop features 
    # data = data.drop(to_drop, axis=1)
    
    
    return data