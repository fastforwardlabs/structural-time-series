import os
import time
import cdsw

def fit_models_parallel():
    '''
    Use the CDSW Workers API (via Python SDK) to launch each model fitting script in parallel

    Docs - https://docs.cloudera.com/machine-learning/cloud/distributed-computing/topics/ml-workers-api.html

    '''
    # Launch a separate worker to run each script independently
    
    base_path = os.getcwd()
    script_path = base_path + '/scripts'

    scripts = os.listdir(script_path)
    scripts = [script_path+'/'+script for script in scripts if script[0:3] in ['fit','mak']]

    for script in scripts:
        cdsw.launch_workers(n=1, cpu=1, memory=3, script=script)
    
    # Force session to persist until each worker job has completed
    # Check for completion every minute
    
    complete = False
    
    while complete == False:
        
        time.sleep(60)
        
        workers = cdsw.list_workers()
        workers_status = [wkr['status'] for wkr in workers]
        
        if all(status == 'succeeded' for status in workers_status):
            complete = True


if __name__ == "__main__":
    fit_models_parallel()