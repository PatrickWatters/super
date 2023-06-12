import os
import json
import logging
import os
import json
import platform
from gpulogger import GPUSidecarLogger
import csv
from dataobjects import JobMeasurment, BatchMeasurment, EpochMeasurment, TrialMeasurment
import pandas as pd

def log_format():
    return "[%(asctime)s][%(levelname)s][%(process)df][%(pathname)s:%(funcName)s:%(lineno)df] %(message)s"

class TrainingProfiler():
    def __init__(self,args,trailid:int, jobid:int, cuda_device_count =0,):
        self.jobid = jobid
        self.trialid = trailid
        self.output_base_folder = "reports/{trailid}"
        self.batch_file= os.path.join(self.output_base_folder + "/{jobid}-batch-stats.csv")
        self.epochs_file= os.path.join(self.output_base_folder + "/{jobid}-epoch-stats.csv")
        self.job_file= os.path.join(self.output_base_folder + "/{jobid}-job-stats.csv")
        self.trial_file= os.path.join(self.output_base_folder + "{trailid}-stats.csv")
        self.config_file= os.path.join(self.output_base_folder + "/{jobid}-config.json")
        self.dump_job_cofig(args)
        self.cuda_device_count = cuda_device_count
        self.job_measurment:JobMeasurment(self.jobid,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    
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
        job_measurment:JobMeasurment = self.job_measurment
        job_measurment.JobTime += epoch_measurement.EpochTime
        job_measurment.JobId = self.jobid
        job_measurment.TotalEpochs = epoch_measurement.Epoch
        job_measurment.TotalBatches += epoch_measurement.NumBatches
        job_measurment.TotalFiles += epoch_measurement.NumFiles
        job_measurment.TotalBatchFetchTime += epoch_measurement.TotalBatchFetchTime
        job_measurment.TotalBatchPrepTime += epoch_measurement.TotalBatchPrepTime
        job_measurment.TotalTransferToGpuTime += epoch_measurement.TotalTransferToGpuTime
        job_measurment.TotalProcessingTime += epoch_measurement.TotalProcessingTime
        job_measurment.TotalCacheHits += epoch_measurement.TotalCacheHits
        job_measurment.TotaCacheMisses += epoch_measurement.TotaCacheMisses
        job_measurment.Loss = epoch_measurement.Loss
        job_measurment.Acc1 = epoch_measurement.Acc1
        job_measurment.Acc5 = epoch_measurement.Acc5
    
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
        job.ImgsPerSec = job.TotalFiles/job.JobTime
        job.BatchesPerSec = job.TotalBatches/job.JobTime
        job.EpochPerSec = job.TotalEpochs/job.JobTime
        job.AvgBatchTime = job.JobTime/job.TotalBatches
        job.AvgEpochTime = job.JobTime/job.TotalEpochs
        job.AvgBatchFetchTime = job.TotalBatchFetchTime/job.TotalBatches
        job.AvgBatchPrepTime = job.TotalBatchPrepTime/job.TotalBatches
        job.AvgTransferToGpuTime = job.TotalTransferToGpuTime/job.TotalBatches
        job.AvgProcessingTime = job.TotalProcessingTime/job.TotalBatches
        job.CacheHitPercentage = job.TotalCacheHits/(job.TotalCacheHits+job.TotaCacheMisses)

        file_path = self.job_file
        if not os.path.exists(file_path):
            header = list(job.__dict__.keys())
            self.write_to_csv(file_path,header)
        row = list(job.__dict__.values())
        self.write_to_csv(file_path,row)
    
    def gen_trial_report(self):
        trail_measurment = TrialMeasurment(self.trialid,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
        for path in os.listdir(self.output_base_folder):
         if 'job-stats.csv' in path:
             df = pd.read_csv(path)
             jobdata =  pd.read_csv(path).df.to_dict()
             trail_measurment.TotalJobs +=1
             trail_measurment.TotalEpochs += jobdata['TotalEpochs']
             trail_measurment.TotalBatches += jobdata['TotalBatches']
             trail_measurment.TotalFiles += jobdata['TotalFiles']
             trail_measurment.TotalBatchFetchTime += jobdata['TotalBatchFetchTime']
             trail_measurment.TotalBatchPrepTime += jobdata['TotalBatchPrepTime']
             trail_measurment.TotalTransferToGpuTime += jobdata['TotalTransferToGpuTime']
             trail_measurment.TotalProcessingTime += jobdata['TotalProcessingTime']
             trail_measurment.Loss[jobdata['JobId']] = jobdata['Loss']
             trail_measurment.Acc1[jobdata['JobId']] = jobdata['Acc1']
             trail_measurment.Acc5[jobdata['JobId']] = jobdata['Acc5']
             trail_measurment.TotalTime = jobdata['JobTime']
             trail_measurment.TotalCacheHits += jobdata['TotalCacheHits']
             trail_measurment.TotaCacheMisses += jobdata['TotaCacheMisses']
   
        trail_measurment.CacheHitPercentage=trail_measurment.TotalCacheHits/(trail_measurment.TotalCacheHits+trail_measurment.TotaCacheMisses)
        trail_measurment.ImgsPerSec = trail_measurment.TotalFiles/trail_measurment.TotalTime
        trail_measurment.BatchesPerSec = trail_measurment.TotalBatches/trail_measurment.TotalTime
        trail_measurment.EpochPerSec = trail_measurment.TotalEpochs/trail_measurment.TotalTime
        trail_measurment.JobsPerSec = trail_measurment.TotalJobs/trail_measurment.TotalTime
        
        trail_measurment.AvgBatchTime = trail_measurment.TotalTime/trail_measurment.TotalBatches
        trail_measurment.AvgEpochTime = trail_measurment.TotalTime/trail_measurment.TotalEpochs
        trail_measurment.AvgJobTime = trail_measurment.TotalTime/trail_measurment.TotalJobs
        trail_measurment.AvgBatchFetchTime = trail_measurment.TotalBatchFetchTime/trail_measurment.TotalBatches
        trail_measurment.AvgBatchPrepTime = trail_measurment.TotalBatchPrepTime/trail_measurment.TotalBatches
        trail_measurment.AvgTransferToGpuTime = trail_measurment.TotalTransferToGpuTime/trail_measurment.TotalBatches
        trail_measurment.AvgProcessingTime = trail_measurment.TotalProcessingTime/trail_measurment.TotalBatches

        file_path = self.trial_file

        if os.path.exists(file_path):
            os.remove(file_path)
            
        header = list(trail_measurment.__dict__.keys())
        self.write_to_csv(file_path,header)
        row = list(trail_measurment.__dict__.values())
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