import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime
import abc

#classe abstrata das simulações
class Simulation:
  def __init__(self, lamda, mu, data, service_durations):
    self.lamda = lamda
    self.mu = mu
    self.data = data
    self.service_durations = service_durations
    self.rho = lamda/mu
  
  @abc.abstractmethod
  def __repr__(self) -> str:
    pass
  @abc.abstractmethod
  def __str__(self) -> str:
    pass
  
#classe Simulação MM1
class MM1Simulation(Simulation):
  def __init__(self, lamda, mu, data, service_durations):
    super().__init__(lamda, mu, data, service_durations)
  def __repr__(self) -> str:
    return super().__repr__()
  def __str__(self) -> str:
    return super().__str__()

  # calcula a média analítica de clientes
  def average_customers(self):
    if(self.rho > 1):
      return math.inf
    return (self.lamda)/(self.mu - self.lamda)

  def average_wait(self):
    if (self.rho > 1 ):
      return math.inf
    return self.lamda/ (self.mu * (self.mu - self.lamda))

  def average_delay(self):
    if(self.rho > 1):
      return math.inf
    return (1/ (self.mu - self.lamda))

  def utilization(self):
    return self.rho

  def export_to_excel(self):
    time_str = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    type_str = "MM1"
    writer = pd.ExcelWriter(f"Simulation_{type_str}_{time_str}_{self.lamda}_{self.mu}.xlsx", engine='xlsxwriter')
    self.data.to_excel(writer, 'simulation_data')
    self.service_durations.to_excel(writer, 'services_data' )
    writer.save()
    return

#classe Simulação MD1
class MD1Simulation(Simulation):
  def __init__(self, lamda, mu, data):
    super().__init__(lamda, mu, data)
    self.D = 1/self.mu
  def __repr__(self) -> str:
    return super().__repr__()
  def __str__(self) -> str:
    return super().__str__()

# simula uma única fila mm1 ou md1, baseado no parametro kind
def m_queue(lamda, mu, max_time = 1000, max_events = 1000, kind = 'm'):
  assert kind in ['m','d']
  wait_function = lambda x: np.random.exponential(1/x) if kind == 'm' else 1/x
  time = 0
  nevents = 0
  ledger = []
  #fila de eventos, arrival @ t = 0
  equeue = [{
    'type' : 'a',
    'time' : time
    }]

  
  def schedule_service():
    service_duration = wait_function(mu)
    service_time = time + service_duration
    equeue.append({
      'type' : 's',
      'time' : service_time,
      'duration' : service_duration
      })
  def schedule_arrival():
    equeue.append({
      'type' : 'a',
      'time' : time +  np.random.exponential(1/lamda),
      })
    

  #variavel aleatória que representa o no de pessoas na fila
  N = 0
  while(nevents < max_events and time < max_time):
    nevents += 1
    event = equeue.pop(0)
    time, etype = event['time'] , event['type']

    if(etype == 'a'):
   
      N += 1
      if(N == 1):
        schedule_service()
      schedule_arrival()
      equeue = sorted(equeue, key = lambda x : x['time'])
    else:
      N -= 1
      if(N > 0):
        service_duration = wait_function(mu)
        service_time = time + service_duration
        equeue.append({
          'type' : 's',
          'time' : service_time,
          'duration' : service_duration
          })
        equeue = sorted(equeue, key = lambda x : x['time'])
    
    event['N'] = N
    ledger.append(event)

  return ledger

# simula multiplas filas mm1 ou md1
def queue_sim(lamda, mu, max_time = 1000, max_events = 1000, runs = 10, kind = 'm', export = False):
  assert kind in ['m','d']

  ledger_df = []

  for i in range(runs):
      ledger = m_queue(lamda, mu, max_time = max_time, max_events = max_events, kind = kind)
      ledger = pd.DataFrame(ledger)
      ledger['run'] = i
      ledger['holding'] = ledger.time.shift(-1) - ledger.time
      ledger_df.append(ledger)

  ledger_df = pd.concat(ledger_df)

  #remove a última amostra, pois não é possível calcular o tempo do estado
  ledger_df.dropna(how = 'any', axis = 0, subset = ['holding'], inplace = True)
  ledger_df.reset_index(inplace=True, drop=True)

  if(kind == 'm'):
    service_df = ledger_df['duration'].rename({
      'duration' : 'service_duration'
    })

    service_df.dropna(inplace=True)

    ledger_df.drop(columns=['duration'], inplace= True)

  

  if(kind == 'm'):
    sim = MM1Simulation(lamda = lamda, mu = mu, data = ledger_df, service_durations = service_df)
  else:
    sim = MD1Simulation(lamda = lamda, mu = mu, data = ledger_df)

  if (export):
    sim.export_to_excel()
  return sim



