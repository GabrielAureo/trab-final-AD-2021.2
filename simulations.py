import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def m_queue(lamda, mu, max_time = 1000, max_events = 1000, kind = 'm'):
  assert kind in ['m','d']
  wait_function = lambda x: np.random.exponential(1/x) if kind == 'm' else 1/x
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
        service_time = time + wait_function(mu)
        equeue.append([service_time, 's'])
      equeue.append([time +  np.random.exponential(1/lamda), 'a'])
      equeue = sorted(equeue, key = lambda x : x[0])
    else:
      N -= 1
      if(N > 0):
        equeue.append([time +   wait_function(mu), 's'])
        equeue = sorted(equeue, key = lambda x : x[0])
        
    ledger.append([time, N, etype])

  return ledger

def queue_sim(lamda, mu, max_time = 1000, max_events = 1000, runs = 10, kind = 'm'):
  ledger_df = []

  for i in range(runs):
      ledger = m_queue(lamda, mu, max_time = max_time, max_events = max_events, kind = kind)
      ledger = pd.DataFrame(ledger, columns = ['time','N', 'op'])
      ledger['run'] = i
      ledger['holding'] = ledger.time.shift(-1) - ledger.time
      ledger_df.append(ledger)

  ledger_df = pd.concat(ledger_df)

  ledger_df.dropna(how = 'any', axis = 0, inplace = True)
  return ledger_df


def utilization(ledger):
  busy_time = ledger[ledger.N > 0].groupby('run').holding.sum()
  total_time = ledger.groupby('run').holding.sum()
  
  utilization = busy_time/total_time

  ci = confidence_interval(utilization)

  return utilization.mean(), ci

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

def md1_queue(lamda, mu, max_time = 1000, max_events = 1000):
  time = 0
  nevents = 0
  ledger = []
  #fila de eventos, arrival @ t = 0
  equeue = [[time, 'a']]

  N = 0
  while(nevents < max_events and time < max_time):
    
    nevents += 1
    event = equeue.pop(0)
    time, etype = event

    if(etype == 'a'):
   
      N += 1
      if(N == 1):
        service_time = time + 1/mu
        equeue.append([service_time, 's'])
      equeue.append([time +  np.random.exponential(1/lamda), 'a'])
      equeue = sorted(equeue, key = lambda x : x[0])
    else:
      N -= 1
      if(N > 0):
        equeue.append([time + 1/mu, 's'])
        equeue = sorted(equeue, key = lambda x : x[0])
        
    ledger.append([time, N, etype])
  return ledger

def md1_simulation(lamda, mu, max_time = 1000, max_events = 1000, runs = 10):
  ledger_df = []

  for i in range(runs):
      ledger = md1_queue(lamda, mu, max_time = max_time, max_events = max_events)
      ledger = pd.DataFrame(ledger, columns = ['time','N', 'op'])
      ledger['run'] = i
      ledger['holding'] = ledger.time.shift(-1) - ledger.time
      ledger_df.append(ledger)

  ledger_df = pd.concat(ledger_df)

  ledger_df.dropna(how = 'any', axis = 0, inplace = True)
  return ledger_df

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

def mean_wait(ledger):
  services_df = ledger[ledger.op =='s']
  services_by_run = services_df.groupby('run')
  arrivals_df = ledger[ledger.op =='a']
  arrivals_by_run = arrivals_df.groupby('run')
  means = []
  for idx, services in services_by_run:
    services_count = services.shape[0]
    corresponding_arrivals = arrivals_by_run.get_group(idx).head(services_count)
    waits = services.time.reset_index() - corresponding_arrivals.time.reset_index()
    means.append(waits.time.mean())

  means = pd.Series(means)

  ci = confidence_interval(means)

  return means.mean(), ci
