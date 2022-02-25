import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

class Simulation:
  def __init__(self, lamda, mu, data):
    self.lamda = lamda
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
    return (self.lamda)/(self.mu - self.lamda)
  def average_wait(self):
    if (self.rho > 1 ):
      return math.inf
    return (1/ (self.mu - self.lamda))
    #return self.lamda/ (self.mu * (self.lamda - self.mu))
  def average_time(self):
    if(self.rho > 1):
      return math.inf
    return self.lamda/ (self.mu * (self.mu - self.lamda))
  def utilization(self):
    return self.rho


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
def queue_sim(lamda, mu, max_time = 1000, max_events = 1000, runs = 10, kind = 'm'):
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
  ledger_df.reset_index(inplace=True)

  if(kind == 'm'):
    sim = MM1Simulation(lamda = lamda, mu = mu, data = ledger_df)
  else:
    sim = MD1Simulation(lamda = lamda, mu = mu, data = ledger_df)
  return sim


#calcula a utilização
#retorna um dicionário com a utilização simulada e analitica
def utilization(simulation_obj):
  simulation = simulation_obj.data
  busy_time = simulation[simulation.N > 0].groupby('run').holding.sum()
  total_time = simulation.groupby('run').holding.sum()
  
  utilization = busy_time/total_time

  ci = confidence_interval(utilization)

  return {
    "Utilization":
    {
      "Simulated":{
        "Mean": utilization.mean(),
        "Confidence Interval" : confidence_interval(utilization)
      },
      "Analytical":{
        'Mean' : simulation_obj.utilization()
      }
    }
  }


#calcula a média de clientes na fila
#retorna um dicionário com a média simulada e analitica
def customers_metrics(simulation_obj):
  simulation = simulation_obj.data
  runs_groupby = simulation.groupby('run')
  total_time = runs_groupby.time.last()
  temp_df = simulation.copy()
  temp_df['area'] = temp_df.holding * temp_df.N
  areas = temp_df.groupby('run').area.sum()

  customers_df = {
    'Average Customers' :{
      'Simulated': {
        'Mean' : (areas/total_time).mean(),
        'Confidence Interval' : confidence_interval(areas/total_time)
    },
      'Analytical' :{
        'Mean' : simulation_obj.average_customers()
      }
    }
  }  
  return customers_df

def confidence_interval(samples, confidence_rate = 1.95):
  x = samples.mean()
  n = samples.count()
  s = samples.std()
  z = confidence_rate
  return (x - z*(s/ math.sqrt(n) ), x+ z*(s/  math.sqrt(n) ))

#calcula o tempo médio de espera e o tempo médio total que o cliente fica na fila
#retorna um dicionário com as métricas simuladas e analíticas
def wait_metrics(simulation_obj):
  simulation = simulation_obj.data
  services_df = simulation[simulation.type =='s']
  services_by_run = services_df.groupby('run')
  arrivals_df = simulation[simulation.type =='a']

  arrivals_by_run = arrivals_df.groupby('run')
  means = {
    'spent_time':[],
    'wait' : []
  }
  for idx, services in services_by_run:
    services = services.reset_index()[['time', 'duration']]
    services_count = services.shape[0]
    corresponding_arrivals = arrivals_by_run.get_group(idx).head(services_count).reset_index().time
    arrivals_services = pd.merge(left = services, right = corresponding_arrivals,\
      left_index= True, right_index= True)
    arrivals_services.rename(columns ={
      'time_x' : 'service_end',
      'time_y' : 'arrival'
    }, inplace = True)
    arrivals_services['service_start'] = arrivals_services['service_end'] - arrivals_services['duration']

    spent_times = arrivals_services.service_end  - arrivals_services.arrival
    waits = arrivals_services.service_start - arrivals_services.arrival

    means['spent_time'].append(spent_times.mean())
    means['wait'].append(waits.mean())

  means = pd.DataFrame(means)


  return {
    "Spent Time" :{
      'Simulated':{
        "Average" : means.spent_time.mean(),
        "Confidence Interval" : confidence_interval(means.spent_time)
      },
      'Analitical': simulation_obj.average_time()
      
    },

    "Wait" :{
      'Simulated' :{
        "Average" : means.wait.mean(),
        "Confidence Interval" : confidence_interval(means.wait)
      },
      'Analitical':  simulation_obj.average_wait()
      
    }
  }


def metrics(simulation_obj):
  _wait_metrics = wait_metrics(simulation_obj)
  _customers_metrics = customers_metrics(simulation_obj)
  _utilization_metric = utilization(simulation_obj)


  _wait_metrics = pd.json_normalize(_wait_metrics)
  _customers_metrics = pd.json_normalize(_customers_metrics)
  _utilization_metric = pd.json_normalize(_utilization_metric)
  
  return pd.concat([_wait_metrics, _customers_metrics, _utilization_metric])

lamda = 100
mu = 1
sim = queue_sim(lamda, mu, max_events=1000, max_time=1000)

# print(sim.data.head(100))
# print(wait_metrics(sim))

# sim = queue_sim(lamda = lamda, mu = mu)
# print(metrics(sim))

# print(pd.DataFrame(wait_metrics(sim) ))
# print(f"Analytical - Spent time: {1/(mu - lamda)}")
# print(f"Analytical - Wait: {lamda/ (mu * (mu - lamda))}")