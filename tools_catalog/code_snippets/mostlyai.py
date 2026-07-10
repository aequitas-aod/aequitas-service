import pandas as pd
from mostlyai.sdk import MostlyAI
from library.src.artifact_types import Data, Configuration

def generate_synthetic_data(data: Data):
    original_df = data.load_dataset()
    mostly = MostlyAI(local=True)

    # train a single-table generator, with default configs
    g = mostly.train(
        name="Demo - Titanic",
        data=original_df,
    )

    # display the quality assurance report
    g.reports(display=True)

    # generate a representative synthetic dataset, with default configs
    sd = mostly.generate(g)
    df = sd.data()

    # or simply probe for some samples
    df = mostly.probe(g, size=100)
    df_samples = mostly.probe(g, seed=pd.DataFrame({
        'age': [28, 44],
        'sex': ['Male', 'Female'],
        'native_country': ['Cuba', 'Mexico'],
    }))
    
    #Create a new Synthetic Dataset via a batch job to conditionally generate 1'000'000 statistically representative synthetic samples.
    sd = mostly.generate(g, size=1_000_000)
    df_synthetic = sd.data()
    df_synthetic
    
    return df_samples
    
    
if __name__ == "__main__":
    data = Data(filepath="https://github.com/mostly-ai/public-demo-data/raw/dev/titanic/titanic.csv")
    generate_synthetic_data(data)