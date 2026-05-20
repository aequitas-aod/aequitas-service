import yaml
from src.modelling import Modelling


def test_train():
    with open("metadata/aipc.yaml", "r") as yaml_file:
        aipc_configs = yaml.safe_load(yaml_file)
        print(aipc_configs)
    aipc_configs[""]
    # data = new Data()
    # modelling = Modelling()
    # assert train()=="model trained"


def test_bias_mitigation_in_process_train():
    pass
