from ydata_profiling import ProfileReport
from library.src.artifact_types import Data, Configuration, Report

"""
The profiling offers comprehensive insights into various types of data, including tabular, time-series and text data.

Tabular data: when dealing with tabular data, such as spreadsheets or databases, the profiling provides valuable statistics on data distribution, central tendencies, and categorical variable frequencies. It identifies multivariate relations such as correlations and interactions in a visual manner. It also identifies missing data.
Time-series data: when dealing with data with temporal dimensions, the profiling extends its capabilities to capture trends, seasonality, cyclic patterns and missing data gaps. It can reveal information about data volatility, periodicity, and anomalies, facilitating a deeper understanding of time-dependent trends.
Text: when it comes to text data, such as strings or documents, the profiling offers insightful statistics on the distribution of word frequencies, common phrases, and unique words.
"""

def data_profiling_eda(input_data: Data, report: Report):
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file(report.resulting_filepath)
    return Report(report.resulting_filepath)
