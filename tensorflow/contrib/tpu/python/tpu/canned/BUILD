py_library(
    name = "tpu_dnn",
    srcs = [
        "tpu_dnn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/tpu:tpu_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:dnn",
        "//tensorflow/python/estimator:head",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/ops/losses",
    ],
)

py_binary(
    name = "tpu_dnn_iris_example",
    srcs = [
        "tpu_dnn_iris_example.py",
    ],
    main = "tpu_dnn_iris_example.py",
    deps = [
        ":tpu_dnn",
        "//tensorflow:tensorflow_py",
    ],
)
