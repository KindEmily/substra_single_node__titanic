# substra_single_node/run_pipeline.py

try:
# [1]
    # Import all the dependencies
    print("Importing dependencies...")

    import os
    import zipfile
    from pathlib import Path

    import substra
    from substra.sdk.schemas import (
        AssetKind,
        DataSampleSpec,
        DatasetSpec,
        FunctionSpec,
        FunctionInputSpec,
        FunctionOutputSpec,
        Permissions,
        TaskSpec,
        ComputeTaskOutputSpec,
        InputRef,
    )
    print("Importing dependencies finished without errors.")


# [2]
# Instantiating the Substra Client
    print("Instantiating the Substra Client")
    client = substra.Client(client_name="org-1")
    print(f"Substra setup complete. Connected to network as: {client.organization_info().organization_id}")


# [3]
    permissions = Permissions(public=True, authorized_ids=[])


# [4]
    root_dir = Path.cwd()

    assets_directory = root_dir / "assets"
    assert assets_directory.is_dir(), """Did not find the asset directory, a directory called 'assets' is
    expected in the same location as this py file"""

# [5]
    dataset = DatasetSpec(
        name="Titanic dataset - Org 1",
        data_opener=assets_directory / "dataset" / "titanic_opener.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions,
        logs_permission=permissions,
    )
    
    dataset_key = client.add_dataset(dataset)
    print(f"Dataset key {dataset_key}")

# [6]
    train_data_sample_folder = assets_directory / "train_data_samples"
    train_data_sample_keys = client.add_data_samples(
        DataSampleSpec(
            paths=list(train_data_sample_folder.glob("*")),
            data_manager_keys=[dataset_key],
        )
    )

    print(f"{len(train_data_sample_keys)} data samples were registered")

# [7]
    test_data_sample_folder = assets_directory / "test_data_samples"
    test_data_sample_keys = client.add_data_samples(
        DataSampleSpec(
            paths=list(test_data_sample_folder.glob("*")),
            data_manager_keys=[dataset_key],
        )
    )

    print(f"{len(test_data_sample_keys)} data samples were registered")



# Metrics 
# [8]
    inputs_metrics = [
        FunctionInputSpec(identifier="datasamples", kind=AssetKind.data_sample, optional=False, multiple=True),
        FunctionInputSpec(identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False),
        FunctionInputSpec(identifier="predictions", kind=AssetKind.model, optional=False, multiple=False),
    ]

    outputs_metrics = [FunctionOutputSpec(identifier="performance", kind=AssetKind.performance, multiple=False)]


    METRICS_DOCKERFILE_FILES = [
        assets_directory / "metric" / "titanic_metrics.py",
        assets_directory / "metric" / "Dockerfile",
    ]

    metric_archive_path = assets_directory / "metric" / "metrics.zip"

    with zipfile.ZipFile(metric_archive_path, "w") as z:
        for filepath in METRICS_DOCKERFILE_FILES:
            z.write(filepath, arcname=os.path.basename(filepath))

    metric_function = FunctionSpec(
        inputs=inputs_metrics,
        outputs=outputs_metrics,
        name="Testing with Accuracy metric",
        description=assets_directory / "metric" / "description.md",
        file=metric_archive_path,
        permissions=permissions,
    )

    metric_key = client.add_function(metric_function)

    print(f"Metric key {metric_key}")


# Adding Function
## Train function 
    ALGO_TRAIN_DOCKERFILE_FILES = [
        assets_directory / "function_random_forest/titanic_function_rf.py",
        assets_directory / "function_random_forest/train/Dockerfile",
    ]

    train_archive_path = assets_directory / "function_random_forest" / "function_random_forest.zip"
    with zipfile.ZipFile(train_archive_path, "w") as z:
        for filepath in ALGO_TRAIN_DOCKERFILE_FILES:
            z.write(filepath, arcname=os.path.basename(filepath))

    train_function_inputs = [
        FunctionInputSpec(identifier="datasamples", kind=AssetKind.data_sample, optional=False, multiple=True),
        FunctionInputSpec(identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False),
    ]

    train_function_outputs = [FunctionOutputSpec(identifier="model", kind=AssetKind.model, multiple=False)]

    train_function = FunctionSpec(
        name="Training with Random Forest",
        inputs=train_function_inputs,
        outputs=train_function_outputs,
        description=assets_directory / "function_random_forest" / "description.md",
        file=train_archive_path,
        permissions=permissions,
    )


    train_function_key = client.add_function(train_function)

    print(f"Train function key {train_function_key}")

## Predict function 
    ALGO_PREDICT_DOCKERFILE_FILES = [
        assets_directory / "function_random_forest/titanic_function_rf.py",
        assets_directory / "function_random_forest/predict/Dockerfile",
    ]

    predict_archive_path = assets_directory / "function_random_forest" / "function_random_forest.zip"
    with zipfile.ZipFile(predict_archive_path, "w") as z:
        for filepath in ALGO_PREDICT_DOCKERFILE_FILES:
            z.write(filepath, arcname=os.path.basename(filepath))

    predict_function_inputs = [
        FunctionInputSpec(identifier="datasamples", kind=AssetKind.data_sample, optional=False, multiple=True),
        FunctionInputSpec(identifier="opener", kind=AssetKind.data_manager, optional=False, multiple=False),
        FunctionInputSpec(identifier="models", kind=AssetKind.model, optional=False, multiple=False),
    ]

    predict_function_outputs = [FunctionOutputSpec(identifier="predictions", kind=AssetKind.model, multiple=False)]

    predict_function_spec = FunctionSpec(
        name="Predicting with Random Forest",
        inputs=predict_function_inputs,
        outputs=predict_function_outputs,
        description=assets_directory / "function_random_forest" / "description.md",
        file=predict_archive_path,
        permissions=permissions,
    )

    predict_function_key = client.add_function(predict_function_spec)

    print(f"Predict function key {predict_function_key}")


# Registering tasks 
# [11]
    data_manager_input = [InputRef(identifier="opener", asset_key=dataset_key)]
    train_data_sample_inputs = [InputRef(identifier="datasamples", asset_key=key) for key in train_data_sample_keys]
    test_data_sample_inputs = [InputRef(identifier="datasamples", asset_key=key) for key in test_data_sample_keys]

    train_task = TaskSpec(
        function_key=train_function_key,
        inputs=data_manager_input + train_data_sample_inputs,
        outputs={"model": ComputeTaskOutputSpec(permissions=permissions)},
        worker=client.organization_info().organization_id,
    )

    train_task_key = client.add_task(train_task)

    print(f"Train task key {train_task_key}")

# [12]
    model_input = [
        InputRef(
            identifier="models",
            parent_task_key=train_task_key,
            parent_task_output_identifier="model",
        )
    ]

    predict_task = TaskSpec(
        function_key=predict_function_key,
        inputs=data_manager_input + test_data_sample_inputs + model_input,
        outputs={"predictions": ComputeTaskOutputSpec(permissions=permissions)},
        worker=client.organization_info().organization_id,
    )

    predict_task_key = client.add_task(predict_task)

    predictions_input = [
        InputRef(
            identifier="predictions",
            parent_task_key=predict_task_key,
            parent_task_output_identifier="predictions",
        )
    ]

    test_task = TaskSpec(
        function_key=metric_key,
        inputs=data_manager_input + test_data_sample_inputs + predictions_input,
        outputs={"performance": ComputeTaskOutputSpec(permissions=permissions)},
        worker=client.organization_info().organization_id,
    )

    test_task_key = client.add_task(test_task)

    print(f"Test task key {test_task_key}")

# [13]
# We wait until the task is finished
    test_task = client.wait_task(test_task_key)

    print(f"Test tasks status: {test_task.status}")

    performance = client.get_task_output_asset(test_task.key, identifier="performance")
    print("Metric: ", test_task.function.name)
    print("Performance on the metric: ", performance.asset)

except ImportError as e:
    print(f"Error: Failed to import required module. {str(e)}")
    print("Please make sure all required packages are installed.")
    print("You may need to run: pip install substrafl")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")
    print("Please check your Python environment and try again.")