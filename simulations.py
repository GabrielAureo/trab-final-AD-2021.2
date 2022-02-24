import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class Simulation:
  def __init__(self, lamda, mu, data):
    self.lamda = lamda,
    self.mu = mu
    self.data = data
    self.rho = lamda/mu
  def __repr__(self) -> str:
    return self.data.__repr__()
  def __str__(self) -> str:
    return self.data.__str__()
  

class MM1Simulation(Simulation):
  def __init__(self, lamda, mu, data):
    super().__init__(lamda, mu, data)
  def __repr__(self) -> str:
    return super().__repr__()
  def __str__(self) -> str:
    return super().__str__()
  # calcula a média analítica de clientes
  def average_customers(self):
    if(self.rho > 1):
      return math.inf
    return self.rho/(1 - self.rho)


class MD1Simulation(Simulation):
  def __init__(self, lamda, mu, data):
    super().__init__(lamda, mu, data)
    self.D = 1/self.mu

# simula uma única fila mm1 ou md1, baseado no parametro kind
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

# simula multiplas filas mm1 ou md1
def queue_sim(lamda, mu, max_time = 1000, max_events = 1000, runs = 10, kind = 'm'):
  assert kind in ['m','d']
  ledger_df = []

  for i in range(runs):
      ledger = m_queue(lamda, mu, max_time = max_time, max_events = max_events, kind = kind)
      ledger = pd.DataFrame(ledger, columns = ['time','N', 'op'])
      ledger['run'] = i
      ledger['holding'] = ledger.time.shift(-1) - ledger.time
      ledger_df.append(ledger)

  ledger_df = pd.concat(ledger_df)

  ledger_df.dropna(how = 'any', axis = 0, inplace = True)

  if(kind == 'm'):
    sim = MM1Simulation(lamda, mu, ledger_df)
  else:
    sim = MD1Simulation(lamda, mu, ledger_df)
  return sim


def utilization(simulation_obj):
  simulation = simulation_obj.data
  busy_time = simulation[simulation.N > 0].groupby('run').holding.sum()
  total_time = simulation.groupby('run').holding.sum()
  
  utilization = busy_time/total_time

  ci = confidence_interval(utilization)

  return utilization.mean(), ci



def customers_mean(simulation_obj):
  simulation = simulation_obj.data
  runs_groupby = simulation.groupby('run')
  total_time = runs_groupby.time.last()
  temp_df = simulation.copy()
  temp_df['area'] = temp_df.holding * temp_df.N
  areas = temp_df.groupby('run').area.sum()

  return (areas/total_time).mean(),confidence_interval(areas/total_time)

def confidence_interval(samples, confidence_rate = 1.95):
  x = samples.mean()
  n = samples.count()
  s = samples.std()
  z = confidence_rate
  return (x - z*(s/ math.sqrt(n) ), x+ z*(s/  math.sqrt(n) ))

def mean_wait(simulation_obj):
  simulation = simulation_obj.data
  services_df = simulation[simulation.op =='s']
  services_by_run = services_df.groupby('run')
  arrivals_df = simulation[simulation.op =='a']
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
