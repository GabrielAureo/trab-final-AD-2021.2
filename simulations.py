import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def mm1_queue(lamda, mu, max_time = 1000, max_events = 1000):
  time = 0
  nevents = 0
  ledger = []
  #fila de eventos, arrival @ t = 0
  equeue = [[time, 'a']]
  #variavel aleatória que representa o no de pessoas na fila
  N = 0
  while(nevents < max_events and time < max_time):
    nevents += 1
    event = equeue.pop(0)
    time, etype = event

    if(etype == 'a'):
   
      N += 1
      if(N == 1):
        service_time = time + np.random.exponential(1/mu)
        equeue.append([service_time, 's'])

      equeue.append([time +  np.random.exponential(1/lamda), 'a'])
      equeue = sorted(equeue, key = lambda x : x[0])
    else:
      N -= 1
      if(N > 0):
        equeue.append([time +  np.random.exponential(1/mu), 's'])
        equeue = sorted(equeue, key = lambda x : x[0])
        
    ledger.append([time, N, etype])

  return ledger


def mm1_simulation(lamda, mu, max_time = 1000, max_events = 1000, runs = 10):
    ledger_df = []
    for i in range(runs):
        ledger = mm1_queue(lamda, mu, max_time = max_time, max_events = max_events)
        ledger = pd.DataFrame(ledger, columns = ['time','N', 'op'])

        ledger['run'] = i
        ledger['holding'] = ledger.time.shift(-1) - ledger.time
        ledger_df.append(ledger)
        
    ledger_df = pd.concat(ledger_df)
    ledger_df.dropna(how = 'any', axis = 0, inplace = True)
    
    return ledger_df


def customers_mean(ledger):
    runs_groupby = ledger.groupby('run')
    total_time = runs_groupby.time.last()
    temp_df = ledger.copy()
    temp_df['area'] = temp_df.holding * temp_df.N
    areas = temp_df.groupby('run').area.sum()

    return (areas/total_time).mean(),confidence_interval(areas/total_time)

def confidence_interval(samples, confidence_rate = 1.95):
    x = samples.mean()
    n = samples.count()
    s = samples.std()
    z = confidence_rate
    return (x - z*(s/ math.sqrt(n) ), x+ z*(s/  math.sqrt(n) ))
