from variationist import Inspector, InspectorArgs, Visualizer, VisualizerArgs
from library.src.artifact_types import Data, Configuration


def run_inspector(data: Data, config: Configuration):
    # Define the inspector arguments
    ins_args = InspectorArgs(text_names=["text"], var_names=["label"], 
        metrics=["npw_pmi"], n_tokens=1, language="en", stopwords=True, lowercase=True)

    # Run the inspector and get the results
    res = Inspector(dataset=data.get_dataset(), args=ins_args).inspect()

    # Define the visualizer arguments
    vis_args = VisualizerArgs(output_folder=config.output_folder, output_formats=["html"])

    # Create interactive charts for all metrics
    charts = Visualizer(input_json=res, args=vis_args).create()
    
    
if __name__ == "__main__":
    data = Data(filepath="data/people_data.json")
    config = Config(config={"output_folder": "output"})
    run_inspector(data, config)