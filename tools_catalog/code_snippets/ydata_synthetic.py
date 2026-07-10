from ydata_profiling import ProfileReport
from temlops.src.artifact_types import Data, Configuration, Report

"""
The profiling offers comprehensive insights into various types of data, including tabular, time-series and text data.

Tabular data: when dealing with tabular data, such as spreadsheets or databases, the profiling provides valuable statistics on data distribution, central tendencies, and categorical variable frequencies. It identifies multivariate relations such as correlations and interactions in a visual manner. It also identifies missing data.
Time-series data: when dealing with data with temporal dimensions, the profiling extends its capabilities to capture trends, seasonality, cyclic patterns and missing data gaps. It can reveal information about data volatility, periodicity, and anomalies, facilitating a deeper understanding of time-dependent trends.
Text: when it comes to text data, such as strings or documents, the profiling offers insightful statistics on the distribution of word frequencies, common phrases, and unique words.
"""

def generate_synthetic_data(input_data: Data, report: Report):
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file(report.resulting_filepath)
    return Report(report.resulting_filepath)

def generate_synthetic_data(input_data: Data, config: Configuration):
    from pmlb import fetch_data

    from data_synthetic.synthesizers.regular import RegularSynthesizer
    from data_syntheticsynthesizers import ModelParameters, TrainParameters
    # Load data
    data = fetch_data('adult')
    num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass','education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'native-country', 'target']
    synth = RegularSynthesizer(modelname='fast')
    synth.fit(data=data, num_cols=num_cols, cat_cols=cat_cols)
    
    synth_data = synth.sample(1000)
    print(synth_data)