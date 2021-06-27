""" Model config in json format """

CFG = {
    "data": {
        "path": "E:/Machine Learning Series/Datasets/archive/real_vs_fake/real-vs-fake/",
        "class_mode": "binary",
        "target_size": (150, 150)
    },
    "train": {
        "batch_size": 100,
        "epochs": 10,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [150, 150, 3],
        "layers_stack": {
            "layer_1": 32,
            "layer_2": 64,
            "layer_3": 128,
            "kernel_size": 3,
            "pool_size": 2,
        },
        "output": 2
    }
}
