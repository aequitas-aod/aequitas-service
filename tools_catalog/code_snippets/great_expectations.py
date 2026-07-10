import great_expectations as gx
import great_expectations.expectations as gxe

from library.src.artifact_types import Data, Artifact, Model, Configuration, Report

def data_validation_values_between(data: Data, config: Configuration) -> Report:
    # Create a data context
    context = gx.get_context(mode="file")
    datasource = context.data_sources.add_pandas_filesystem(name="recruitmentdataset0", base_directory="../data")
    data_asset = datasource.add_csv_asset(name="recruitmentdataset-2022")
    # Create a batch request
    batch_definition_name = "recruitmentdataset-2022_1"
    batch_definition_path = "recruitmentdataset-2022.csv"

    batch_definition = data_asset.add_batch_definition_path(
        name=batch_definition_name, path=batch_definition_path
    )
    # Get the dataframe as a Batch
    batch = batch_definition.get_batch()
    # Create an Expectation to test
    expectation = gxe.ExpectColumnValuesToBeBetween(
        column="ind-university_grade", max_value=80, min_value=60
    )
    # Test the Expectation
    validation_results = batch.validate(expectation)
    return validation_results

def data_validation_table_row_count(data: Data, config: Configuration) -> Report:
    # Create a data context
    context = gx.get_context(mode="file")
    datasource = context.data_sources.add_pandas_filesystem(name="recruitmentdataset0", base_directory="../data")
    data_asset = datasource.add_csv_asset(name="recruitmentdataset-2022")
    # Create a batch request
    batch_definition_name = "recruitmentdataset-2022_1"
    batch_definition_path = "recruitmentdataset-2022.csv"

    batch_definition = data_asset.add_batch_definition_path(
        name=batch_definition_name, path=batch_definition_path
    )
    # Get the dataframe as a Batch
    batch = batch_definition.get_batch()
    # Create an Expectation to test
    expectation = gxe.ExpectTableRowCountToEqual(
        column="ind-university_grade", max_value=80, min_value=60
    )
    # Test the Expectation
    validation_results = batch.validate(expectation)
    return validation_results