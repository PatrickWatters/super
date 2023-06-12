@dataclasses.dataclass
class BatchMeasurment:
    JobId :int
    BatchId :int
    Epoch: int
    BatchIdx: int
    NumFiles:int
    BatchTime: float
    ImgsPerSec: float
    BatchFetchTime: float
    BatchPrepTime: float
    TransferToGpuTime: float
    ProcessingTime: float
    Loss: float
    Acc1: float
    Acc5: float
    CacheHit: bool

@dataclasses.dataclass
class EpochMeasurment:
    JobId :int
    Epoch: int
    NumBatches: int
    NumFiles:int
    EpochTime: float
    ImgsPerSec: float
    BatchesPerSec: float

    TotalBatchFetchTime: float
    TotalBatchPrepTime: float
    TotalTransferToGpuTime: float
    TotalProcessingTime: float

    Loss: dict
    Acc1: dict
    Acc5: dict
    AvgBatchTime: float
    AvgBatchFetchTime: float
    AvgBatchPrepTime: float
    AvgTransferToGpuTime: float
    AvgProcessingTime: float
    TotalCacheHits:int
    TotaCacheMisses:int
    CacheHitPercentage: float

@dataclasses.dataclass
class JobMeasurment:
    JobId :int
    TotalEpochs: int
    TotalBatches: int
    TotalFiles:int
    JobTime: float
    


    ImgsPerSec:float
    BatchesPerSec: float
    EpochPerSec: float
    TotalBatchFetchTime: float
    TotalBatchPrepTime: float
    TotalTransferToGpuTime: float
    TotalProcessingTime: float

    Loss: dict
    Acc1: dict
    Acc5: dict

    AvgBatchTime: float
    AvgEpochTime: float
    
    AvgBatchFetchTime: float
    AvgBatchPrepTime: float
    AvgTransferToGpuTime: float
    AvgProcessingTime: float

    TotalCacheHits:int
    TotaCacheMisses:int
    CacheHitPercentage: float

import dataclasses

@dataclasses.dataclass
class TrialMeasurment:
    TrialId: int
    TotalJobs: int
    TotalEpochs: int
    TotalBatches: int
    TotalFiles:int
    TotalTime: float
    
    ImgsPerSec:float
    BatchesPerSec: float
    EpochPerSec: float
    JobsPerSec: float


    TotalBatchFetchTime: float
    TotalBatchPrepTime: float
    TotalTransferToGpuTime: float
    TotalProcessingTime: float
    Loss: float
    Acc1: float
    Acc5: float
    
    AvgBatchTime: float
    AvgEpochTime: float
    AvgJobTime: float

    AvgBatchFetchTime: float
    AvgBatchPrepTime: float
    AvgTransferToGpuTime: float
    AvgProcessingTime: float

    TotalCacheHits:int
    TotaCacheMisses:int
    CacheHitPercentage: float