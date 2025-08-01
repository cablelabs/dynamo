# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import logging

from dynamo._core import AggregatedMetrics as AggregatedMetrics

try:
    from dynamo._core import BlockManager as BlockManager
except ImportError:
    pass  # BlockManager is not enabled by default

from dynamo._core import ApproxKvIndexer as ApproxKvIndexer
from dynamo._core import DisaggregatedRouter as DisaggregatedRouter
from dynamo._core import EngineType
from dynamo._core import EntrypointArgs as EntrypointArgs
from dynamo._core import ForwardPassMetrics as ForwardPassMetrics
from dynamo._core import HttpAsyncEngine as HttpAsyncEngine
from dynamo._core import HttpError as HttpError
from dynamo._core import HttpService as HttpService
from dynamo._core import KvEventPublisher as KvEventPublisher
from dynamo._core import KvIndexer as KvIndexer
from dynamo._core import KvMetricsAggregator as KvMetricsAggregator
from dynamo._core import KvRecorder as KvRecorder
from dynamo._core import KvStats as KvStats
from dynamo._core import ModelType as ModelType
from dynamo._core import OverlapScores as OverlapScores
from dynamo._core import RadixTree as RadixTree
from dynamo._core import SpecDecodeStats as SpecDecodeStats
from dynamo._core import WorkerMetricsPublisher as WorkerMetricsPublisher
from dynamo._core import WorkerStats as WorkerStats
from dynamo._core import ZmqKvEventListener as ZmqKvEventListener
from dynamo._core import ZmqKvEventPublisher as ZmqKvEventPublisher
from dynamo._core import ZmqKvEventPublisherConfig as ZmqKvEventPublisherConfig
from dynamo._core import compute_block_hash_for_seq_py as compute_block_hash_for_seq_py
from dynamo._core import make_engine
from dynamo._core import register_llm as register_llm
from dynamo._core import run_input

try:
    from dynamo.llm.tensorrtllm import (  # noqa: F401
        get_llm_engine as get_tensorrtllm_engine,
    )
    from dynamo.llm.tensorrtllm import (  # noqa: F401
        get_publisher as get_tensorrtllm_publisher,
    )
except ImportError:
    pass  # TensorRTLLM is not enabled by default
except Exception as e:
    # Don't let TensorRTLLM break other engines
    logger = logging.getLogger(__name__)
    logger.exception(f"Error importing TensorRT-LLM components: {e}")
