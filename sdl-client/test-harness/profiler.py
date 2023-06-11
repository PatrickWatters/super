import os
import json
import logging
import os
import json
import platform
from gpulogger import GPUSidecarLogger
import csv
import dataclasses

@dataclasses.dataclass
class BatchMeasurment:
    JobId :int
    BatchId :int
    Epoch: int
    BatchIdx: int
    NumFiles:int
    TotalBatchTime: float
    ImgsPerSec: float
    DataFetchTime: float
    DataPrepTime: float
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
    TotalEpochTime: float

    ImgsPerSec: float
    BatchesPerSec: float

    TotalDataFetchTime: float
    TotalDataPrepTime: float
    TotalTransferToGpuTime: float
    TotalProcessingTime: float

    AvgLoss: float
    AvgAcc1: float
    AvgAcc5: float
    AvgBatchTime: float
    AvgDataFetchTime: float
    AvgDataPrepTime: float
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
    TotalTime: float
    
    ImgsPerSec:float
    BatchesPerSec: float
    EpochPerSec: float

    TotalDataFetchTime: float
    TotalDataPrepTime: float
    TotalTransferToGpuTime: float
    TotalProcessingTime: float
  
  
    AvgLoss: float
    AvgAcc1: float
    AvgAcc5: float

    AvgBatchTime: float
    AvgEpochTime: float
    
    AvgDataFetchTime: float
    AvgDataPrepTime: float
    AvgTransferToGpuTime: float
    AvgProcessingTime: float

    TotalCacheHits:int
    TotaCacheMisses:int
    CacheHitPercentage: float

@dataclasses.dataclass
class TrialMeasurment:
    TrialDescription: str
    TotalJobs: int
    TotalEpochs: int
    TotalBatches: int
    TotalFiles:int
    TotalTime: float
    Speed:float
    TotalDataLoadingTime: float
    TotalTransferToGpuTime: float
    TotalComputeTime: float
    Loss: float
    Acc1: float
    Acc5: float
    AvgBatchTime: float
    AvgDataLoadingTime: float
    AvgTransferToGpuTime: float
    AvgComputeTime: float
    TotalHits:int
    TotalMisses:int
    HitPercentage: float

def log_format():
    return "[%(asctime)s][%(levelname)s][%(process)df][%(pathname)s:%(funcName)s:%(lineno)df] %(message)s"

class TrainingProfiler():
    def __init__(self,args,trailid:int, jobid:int, cuda_device_count =0,):
        self.jobid = jobid
        self.output_base_folder = "reports/{self.trailid}"
        self.batch_file= os.path.join(self.output_base_folder + "/{jobid}-batch-stats.csv")
        self.epochs_file= os.path.join(self.output_base_folder + "/{jobid}-epoch-stats.csv")
        self.job_file= os.path.join(self.output_base_folder + "/{jobid}-job-stats.csv")
        self.config_file= os.path.join(self.output_base_folder + "/{jobid}-config.json")
        self.dump_job_cofig(args)
        self.cuda_device_count = cuda_device_count
        self.job_measurment:JobMeasurment
        
    
    def log_batch_stats(self, batch_measurement:BatchMeasurment):
        if not os.path.exists(self.batch_file):
            header = list(batch_measurement.__dict__.keys())
            self.write_to_csv(self.batch_file,header)
        row = list(batch_measurement.__dict__.values())
        self.write_to_csv(self.batch_file,row)

    def log_epoch_stats(self, epoch_measurement:EpochMeasurment):
        if not os.path.exists(self.epochs_file):
                header = list(epoch_measurement.__dict__.keys())
                self.write_to_csv(self.epochs_file,header)
        row = list(epoch_measurement.__dict__.values())
        self.write_to_csv(self.epochs_file,row)
        
        self.job_measurment.TotalTime += epoch_measurement.TotalEpochTime
        self.job_measurment.JobId = self.jobid
        self.job_measurment.TotalEpochs = epoch_measurement.Epoch
        self.job_measurment.TotalBatches += epoch_measurement.NumBatches
        self.job_measurment.TotalFiles += epoch_measurement.NumFiles
        self.job_measurment.TotalDataFetchTime += epoch_measurement.TotalDataFetchTime
        self.job_measurment.TotalDataPrepTime += epoch_measurement.TotalDataPrepTime
        self.job_measurment.TotalTransferToGpuTime += epoch_measurement.TotalTransferToGpuTime
        self.job_measurment.TotalProcessingTime += epoch_measurement.TotalProcessingTime
        self.job_measurment.TotalCacheHits += epoch_measurement.TotalCacheHits
        self.job_measurment.TotaCacheMisses += epoch_measurement.TotaCacheMisses
        self.job_measurment.AvgLoss += epoch_measurement.AvgLoss
        self.job_measurment.AvgAcc1 += epoch_measurement.AvgAcc1
        self.job_measurment.AvgAcc5 += epoch_measurement.AvgAcc5


    def write_to_csv(self, filepath, row):
        #create file if it doesn't exist
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        else:
            with open(filepath, 'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    
    def start_gpu_profiler(self):
        if self.cuda_device_count > 0:
            # gpu_util logger
            gpu_util_log = logging.FileHandler(os.path.join(self.output_base_folder, f"gpuutil-{os.getpid()}.log"))
            gpu_util_log.setFormatter(logging.Formatter(""))
            logging.getLogger("gpuutil").addHandler(gpu_util_log)
            logging.getLogger("gpuutil").setLevel(logging.DEBUG)
            logging.getLogger("gpuutil").propagate = False
            self.gpu_logger = GPUSidecarLogger(refresh_rate=0.5, max_runs=-1)
            print('starting gpu stats logger')
            self.gpu_logger.start()

    def dump_job_cofig(self,args):
        with (self.config_file).open("w") as f:
            metadata = vars(args).copy()
            #metadata["output_base_folder"] = metadata["output_base_folder"].name
            metadata.update(platform.uname()._asdict())
            json.dump(metadata, f)
        
    def stop_gpu_profiler(self):
            self.gpu_logger.stop()

    def gen_final_job_report(self):
        job:JobMeasurment = self.job_measurment 
        job.TotalFiles = job.TotalFiles/job.TotalTime
        job.BatchesPerSec = job.TotalBatches/job.TotalTime
        job.EpochPerSec = job.TotalEpochs/job.TotalTime
        job.ImgsPerSec = job.TotalFiles/job.TotalTime
        job.AvgBatchTime = job.TotalBatches/job.TotalTime
        job.AvgEpochTime = job.TotalEpochs/job.TotalTime
        job.AvgDataFetchTime = job.TotalDataFetchTime/job.TotalTime
        job.AvgDataPrepTime = job.TotalDataPrepTime/job.TotalTime
        job.AvgTransferToGpuTime = job.TotalTransferToGpuTime/job.TotalTime
        job.AvgProcessingTime = job.TotalProcessingTime/job.TotalTime
        job.CacheHitPercentage = job.TotalCacheHits/(job.TotalCacheHits+job.TotaCacheMisses)

        file_path = self.job_file
        if not os.path.exists(file_path):
            header = list(job.__dict__.keys())
            self.write_to_csv(file_path,header)
        row = list(job.__dict__.values())
        self.write_to_csv(file_path,row)