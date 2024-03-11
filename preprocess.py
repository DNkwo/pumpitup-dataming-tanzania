import pandas as pd

def preprocess_data(data):
    #reduce cardinality function (top n most categories, label rest as 'other')
    def reduce_cardinality(df, column, n=100):
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
    
    return data