kind create cluster
kubectl config get-contexts
kubectl config use-context kind-kind

# local deployment of kserve
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.15/hack/quick_install.sh" | bash

kubectl create namespace kserve-test
#Create an InferenceService to deploy the model.
TODO
# apply the InferenceSefvice to the cluster
kubectl apply -n kserve-test -f kserve_container.yaml
# check Inferenceservice status
kubectl get inferenceservices hiring-sklearn-mlflow -n kserve-test
# detailed logs
kubectl get inferenceservice -n kserve-test hiring-sklearn-mlflow -oyaml


INGRESS_GATEWAY_SERVICE=$(kubectl get svc -n istio-system --selector="app=istio-ingressgateway" -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward -n istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80
SERVICE_HOSTNAME=$(kubectl get inferenceservice hiring-sklearn-mlflow -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)



# figure out the address the istio is listening on
kubectl get svc istio-ingressgateway -n istio-system

# Some useful commands for debugging 
kubectl get namespaces
kubectl get pods -n kserve-test
kubectl -n kserve-test logs hiring-sklearn-mlflow-predictor-00001-deployment-674dcbdb7kl5n9
kubectl delete -f kserve_container.yaml -n kserve-test

