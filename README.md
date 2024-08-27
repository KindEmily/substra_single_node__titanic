# Source 
https://docs.substra.org/en/stable/examples/substra_core/titanic_example/run_titanic.html

# Run 
```
cd <root dir>
conda env create -f substra-environment.yml 
conda activate <name>
python run_pipeline.py
```

# Output: 

```
(substra_single_node) C:\Users\probl\Work\Substra_env\substra_single_node>python run_pipeline.py
Importing dependencies...
Importing dependencies finished without errors.
Instantiating the Substra Client
Substra setup complete. Connected to network as: MyOrg1MSP
Dataset key c5185d7d-1704-4ea6-b96c-34652ca80fa3
10 data samples were registered
2 data samples were registered
Metric key eedcef68-5efd-4916-b241-297bdddc45ac
Train function key 16bdca13-49b6-46a4-a182-0ea3a38eeb20
Predict function key ed401dd5-c5e5-4f30-9ce8-1e6e7ccda4dd
Train task key efe41fb7-0c21-4917-967b-0c82243c74e3
Test task key 5df4018a-0678-41ee-9c41-346aa2bc0420
Test tasks status: ComputeTaskStatus.done
Metric:  Testing with Accuracy metric
Performance on the metric:  0.8212290502793296

(substra_single_node) C:\Users\probl\Work\Substra_env\substra_single_node>
```