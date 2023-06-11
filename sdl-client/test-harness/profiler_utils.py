import os
import sys
import time
import json
import logging
import os
import json
import platform
from argparse import Namespace
from datetime import datetime
from typing import Any
from typing import Optional
from gpulogger import GPUSidecarLogger
import csv
import dataclasses

@dataclasses.dataclass
class BatchMeasurment:
    JobId :int
    BatchId :int
    Epoch: int
    Batch_Idx: int
    Num_Files:int
    TotalTime: float
    Speed:float
    CacheHit: bool
    DataLoadingTime: float
    TransferToGpuTime: float
    ComputeTime: float
    Loss: float
    Acc1: float
    Acc5: float
    AvgTime: float
    AvgSpeed: float
    AvgDataLoadingTime: float
    AvgTransferToGpuTime: float
    AvgComputeTime: float
    TotalHits:int
    TotalMisses:int
    HitPercentage: float


@dataclasses.dataclass
class EpochMeasurment:
    JobId :int
    Epoch: int
    NumBatches: int
    NumFiles:int
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

@dataclasses.dataclass
class JobMeasurment:
    JobId :int
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
    def __init__(self, args: Namespace,trailid:int, jobid:int, action: str, loglevel: str = "INFO", cuda_device_count =0, ):
        ts = datetime.now().strftime("%Y%m%df%H%M%S")
        #self.output_base_folder = args.output_base_folder / f"{ts}_{action}"
        self.output_base_folder = '/Users/patrickwatters/Projects/super/reports'
        self.jobid = jobid
        #self.output_base_folder.mkdir(exist_ok=False, parents=True)
        self.loglevel = loglevel
        self.args = args
        self.cuda_device_count = cuda_device_count
        self.gpu_logger = None
        self.initialize_logging()
        #self.dump_metadata()
        self.batch_outfile= os.path.join(self.output_base_folder + "/batches-" + str(os.getpid()) + 'csv')
        self.epochs_file= os.path.join(self.output_base_folder + "/epochs-" +str(os.getpid()) +'.csv')
        self.gpu_logger
        self.data_time = 0
        self.memcpy_time =0
        self.active_sub = False
        self.compute_time = 0
        self.epoch_measurement:EpochMeasurment 
        self.job_measurment:JobMeasurment
        
    
    def log_batch_stats(self, batch_measurement:BatchMeasurment):
        file_path = self.batch_outfile
        if not os.path.exists(file_path):
            header = list(batch_measurement.__dict__.keys())
            self.write_to_csv(file_path,header)
        row = list(batch_measurement.__dict__.values())
        self.write_to_csv(file_path,row)

    def log_epoch_stats(self, epoch_measurement:EpochMeasurment):
        file_path = self.epochs_file
        if not os.path.exists(file_path):
                header = list(epoch_measurement.__dict__.keys())
                self.write_to_csv(file_path,header)
        row = list(epoch_measurement.__dict__.values())
        self.write_to_csv(file_path,row)
           

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
    
    def initialize_logging(self):
        logging.basicConfig(level=self.loglevel, format=log_format())
        root = logging.getLogger()
        root.setLevel(self.loglevel)
        #fh_root = logging.FileHandler(os.path.join(self.output_base_folder, f"out-{os.getpid()}.log"))
        #root.addHandler(fh_root)

        #fh_stopwatch = logging.FileHandler(os.path.join(self.output_base_folder, f"results-{os.getpid()}.log"))
        #fh_stopwatch.setFormatter(logging.Formatter(""))
        #logging.getLogger("stopwatch").addHandler(fh_stopwatch)
        #logging.getLogger("stopwatch").setLevel(logging.DEBUG)
        #logging.getLogger("stopwatch").propagate = False

        timeline_log = logging.FileHandler(os.path.join(self.output_base_folder, f"timeline-{os.getpid()}.log"))
        timeline_log.setFormatter(logging.Formatter(""))
        logging.getLogger("timeline").addHandler(timeline_log)
        logging.getLogger("timeline").setLevel(logging.DEBUG)
        logging.getLogger("timeline").propagate = False

        #accuracy_log = logging.FileHandler(os.path.join(self.output_base_folder, f"accuracy-{os.getpid()}.log"))
        #accuracy_log.setFormatter(logging.Formatter(""))
        #logging.getLogger("accuracy").addHandler(accuracy_log)
        #logging.getLogger("accuracy").setLevel(logging.DEBUG)
        #logging.getLogger("accuracy").propagate = False

        if self.cuda_device_count > 0:
            # gpu_util logger
            gpu_util_log = logging.FileHandler(os.path.join(self.output_base_folder, f"gpuutil-{os.getpid()}.log"))
            gpu_util_log.setFormatter(logging.Formatter(""))
            logging.getLogger("gpuutil").addHandler(gpu_util_log)
            logging.getLogger("gpuutil").setLevel(logging.DEBUG)
            logging.getLogger("gpuutil").propagate = False
            self.gpu_logger = GPUSidecarLogger(refresh_rate=0.5, max_runs=-1)
            print('start gpu stats logger')
            self.gpu_logger.start()
    
    def dump_metadata(self):
        with (self.output_base_folder + "/metadata-{self.jobid}.json").open("w") as f:
            metadata = vars(self.args).copy()
            metadata["output_base_folder"] = metadata["output_base_folder"].name
            metadata.update(platform.uname()._asdict())
            json.dump(metadata, f)
    
    
    def start_data_tick(self,batch_timeline_id):
        self.active = True
        self.data_time = time.time()
        logging.getLogger("timeline").debug(
            json.dumps({"item": "batch", "id": batch_timeline_id, "start_time": self.data_time})
        )

    def stop_data_tick(self,batch_timeline_id):
        if self.active:
            self.data_time = time.time() - self.data_time
            self.active = False
            logging.getLogger("timeline").debug(
            json.dumps({"item": "batch", "id": batch_timeline_id, "end_time": self.data_time}))
        else:
            print("ERR in iter {} DATA".format(self.iter))
            raise Exception("Timer stopeed without starting")
       
    def start_memcpy_tick(self,batch_timeline_id):
        self.active_sub = True
        self.memcpy_time = time.time()
        logging.getLogger("timeline").debug(
        json.dumps({"item": "training_batch_to_device", "id": batch_timeline_id, "start_time": self.memcpy_time}))

    def stop_memcpy_tick(self,batch_timeline_id):
        if self.active_sub:
            self.memcpy_time = time.time() - self.memcpy_time
            self.total_memcpy_time += self.memcpy_time
            self.active_sub = False
            json.dumps({"item": "training_batch_to_device", "id": batch_timeline_id, "end_time": self.memcpy_time})
        else:
            print("ERR in iter {} MEMCPY".format(self.iter))
            raise Exception("Timer stopeed without starting")
    
    def start_compute_tick(self,training_batch_timeline_id):
        self.active = True
        self.compute_time = time.time()
        logging.getLogger("timeline").debug(
            json.dumps({"item": "processing_training_batch", "id": training_batch_timeline_id, "start_time": self.compute_time})
        )

    def stop_compute_tick(self,total_time, training_batch_timeline_id, speed, loss, Prec1, Prec5):
        if self.active:
            self.compute_time = time.time() - self.compute_time
            self.active = False
            logging.getLogger("timeline").debug(
            json.dumps({"item": "processing_training_batch", "id": training_batch_timeline_id, "end_time": self.compute_time}))
        else:
            print("ERR in iter {} COMP".format(self.iter))
            raise Exception("Timer stopeed without starting")
        
    def stop_profiler(self):
        if self.cuda_device_count > 0:
            self.gpu_logger.stop()


