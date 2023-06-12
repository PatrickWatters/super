from reporting.dataobjects import BatchMeasurment, EpochMeasurment, TrialMeasurment
import os
import pandas as pd
import csv

def gen_trial_csv(trial_folder, trial_id):    
    
    trail_measurment = TrialMeasurment(trial_id,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

    for path in os.listdir(path):
         if 'job-stats.csv' in path:
             df = pd.read_csv(path)
             jobdata =  pd.read_csv(path).df.to_dict()
             trail_measurment.TotalJobs +=1
             trail_measurment.TotalEpochs += jobdata['TotalEpochs']
             trail_measurment.TotalBatches += jobdata['TotalBatches']
             trail_measurment.TotalFiles += jobdata['TotalFiles']
             trail_measurment.TotalDataFetchTime += jobdata['TotalDataFetchTime']
             trail_measurment.TotalDataPrepTime += jobdata['TotalDataPrepTime']
             trail_measurment.TotalTransferToGpuTime += jobdata['TotalTransferToGpuTime']
             trail_measurment.TotalProcessingTime += jobdata['TotalProcessingTime']
             trail_measurment.Loss[jobdata['JobId']] = jobdata['Loss']
             trail_measurment.Acc1[jobdata['JobId']] = jobdata['Acc1']
             trail_measurment.Acc5[jobdata['JobId']] = jobdata['Acc5']
             trail_measurment.TotalTime = jobdata['TotalTime']
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
    trail_measurment.AvgDataFetchTime = trail_measurment.TotalTime/trail_measurment.TotalDataFetchTime
    trail_measurment.AvgDataPrepTime = trail_measurment.TotalTime/trail_measurment.TotalDataPrepTime
    trail_measurment.AvgTransferToGpuTime = trail_measurment.TotalTime/trail_measurment.TotalTransferToGpuTime
    trail_measurment.AvgProcessingTime = trail_measurment.TotalTime/trail_measurment.TotalProcessingTime

    outputpath =os.path.join(trial_folder + "/trial-{trial_id}.csv")
    if not os.path.exists(outputpath):
        header = list(trail_measurment.__dict__.keys())
        write_to_csv(outputpath,header)
    row = list(trail_measurment.__dict__.values())
    write_to_csv(outputpath,row)
    
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

