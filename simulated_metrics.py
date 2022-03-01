from simulations import Simulation, MM1Simulation, MD1Simulation, queue_sim
import math
import pandas as pd
import numpy as np

#calcula a utilização
#retorna um dicionário com a utilização simulada e analitica
def utilization(simulation_obj : Simulation):
  simulation = simulation_obj.data
  busy_time = simulation[simulation.N > 0].groupby('run').holding.sum()
  total_time = simulation.groupby('run').holding.sum()
  
  utilization = busy_time/total_time

  ci = confidence_interval(utilization)

  return {
    "Utilization":{
      "Simulated": utilization.mean(),
      "Confidence Interval" : ci,
      "Analytical": simulation_obj.utilization()
    }

  }


#calcula a média de clientes na fila
#retorna um dicionário com a média simulada e analitica
def customers_metrics(simulation_obj : Simulation):
  simulation = simulation_obj.data
  runs_groupby = simulation.groupby('run')
  total_time = runs_groupby.time.last()
  temp_df = simulation.copy()
  temp_df['area'] = temp_df.holding * temp_df.N
  areas = temp_df.groupby('run').area.sum()

  customers_df = {
    'Average Customers' :{
      'Simulated':  (areas/total_time).mean(),
      'Confidence Interval' : confidence_interval(areas/total_time),
      'Analytical' : simulation_obj.average_customers()
    }
  }  
  return customers_df

def confidence_interval(samples, confidence_rate = 1.95):
  #print(samples)
  x = samples.mean()
  n = samples.count()
  s = samples.std()
  z = confidence_rate
  return (x - z*(s/ math.sqrt(n) ), x+ z*(s/  math.sqrt(n) ))

#calcula o tempo médio de espera e o tempo médio total que o cliente fica na fila
#retorna um dicionário com as métricas simuladas e analíticas
def wait_metrics(simulation_obj : Simulation):
  simulation = simulation_obj.data
  services_df = simulation[simulation.type =='s']
  services_df = pd.merge(left= services_df, right = simulation_obj.service_durations, left_index=True, right_index=True)

  services_by_run = services_df.groupby('run')
  arrivals_df = simulation[simulation.type =='a']

  arrivals_by_run = arrivals_df.groupby('run')
  means = {
    'delay':[],
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

    delays = arrivals_services.service_end  - arrivals_services.arrival
    waits = arrivals_services.service_start - arrivals_services.arrival

    means['delay'].append(delays.mean())
    means['wait'].append(waits.mean())

  means = pd.DataFrame(means)


  return {
    "Delay" :{
      "Simulated" : means.delay.mean(),
      "Confidence Interval" : confidence_interval(means.delay),
      'Analytical': simulation_obj.average_delay()
      
    },

    "Wait" :{
      'Simulated' : means.wait.mean(),
      "Confidence Interval" : confidence_interval(means.wait),
      'Analytical':  simulation_obj.average_wait()
      
    }
  }


def metrics(simulation_obj : Simulation):
  _wait_metrics = wait_metrics(simulation_obj)
  _customers_metrics = customers_metrics(simulation_obj)
  _utilization_metric = utilization(simulation_obj)


  _wait_metrics = pd.DataFrame.from_dict(_wait_metrics, orient= 'index')
  _customers_metrics = pd.DataFrame.from_dict(_customers_metrics, orient= 'index')
  _utilization_metric = pd.DataFrame.from_dict(_utilization_metric, orient= 'index')

  metrics_df = pd.concat([_wait_metrics, _customers_metrics, _utilization_metric])\
    .stack()\
    .to_frame(name = 'Metrics')
  return metrics_df


def delay_dist(simulation_obj : Simulation):
  simulation = simulation_obj.data

  services_df = simulation[simulation.type =='s']
  services_df = pd.merge(left= services_df, right = simulation_obj.service_durations, left_index=True, right_index=True)
  services_by_run = services_df.groupby('run')
  arrivals_df = simulation[simulation.type =='a']
  arrivals_by_run = arrivals_df.groupby('run')

  delays = []
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

    delays_N_run = arrivals_services.service_end  - arrivals_services.arrival
    delays_N_run['run'] = idx
    delays.append(delays_N_run)
  delays = pd.DataFrame(delays)
  delays.run = delays.run.astype(int)
  delays = delays.set_index('run')
  delays = delays.stack()
  delays = delays.droplevel(1)
  
  return histogram(delays)

def histogram(delays, n_bins = 10):
  run_group = delays.groupby('run')
  mean_max = run_group.max().mean()
  mean_min = run_group.min().mean()
  bin_size = (mean_max - mean_min)/n_bins

  bins = np.linspace(mean_min, mean_max, n_bins)

  cuts = []
  for i in range(1, n_bins):
    bin = ( mean_min  + ((i - 1) * bin_size), mean_min  + (i * bin_size))
    cut = delays[ delays.between( bin[0], bin[1], inclusive = False) ]
    cut = cut.to_frame(name = 'delay')
    cut['cut'] = [bin for _ in range(cut.shape[0])]
    cuts.append(cut)
  
  cuts = pd.concat(cuts)
  cuts_group = cuts.groupby(['run', 'cut'])
  cuts_by_run = cuts_group.sum()
  pdf_by_run = cuts_by_run/ cuts_by_run.sum(level = 'run')
  pdf = pdf_by_run.mean(level = 'cut')
  cdf_by_run = pdf_by_run.groupby('run').apply(lambda x: x.cumsum())
  cdf = cdf_by_run.mean(level = 'cut')
  # cuts = cuts.reset_index()
  # cuts = cuts.set_index(['run', 'bin'])
  # cuts = cuts.sort_index()

  # mean_delay_by_cut_and_bin = cuts.mean(level = ['run','bin'])

  # print(mean_delay_by_cut_and_bin.mean(level = 'bin'))
  # mean_delay_by_cut_and_bin.to_clipboard()
  return pdf, cdf
    

def customers_dist(simulation_obj : Simulation):
  simulation = simulation_obj.data
  

  def pdf():
    pdf = simulation.groupby(['run', 'N'])['holding'].sum()
    pdf_per_run = pdf/pdf.groupby('run').sum()
    ci_df = pdf_per_run.groupby('N').apply(confidence_interval).to_frame(name = 'pdf_CI')
    # remove valores (nan, nan) resultantes de amostras únicas
    ci_df = ci_df[~ci_df.pdf_CI.apply( lambda x: np.isnan(x[0]) & np.isnan(x[1]))]
    pdf = pdf_per_run.groupby('N').mean()
    pdf = pdf.rename('pdf')
    pdf_df = pd.merge(left = pdf, right = ci_df, left_on='N', right_on='N')
    return pdf_df
  
  def cdf():
    pdf = simulation.groupby(['run', 'N'])['holding'].sum()
    
    pdf_per_run = pdf/pdf.groupby('run').sum()
    cdf_per_run = pdf_per_run.groupby(['run']).apply(
      lambda x: x.sort_index().cumsum(axis = 0)
    )

    ci_df = cdf_per_run.groupby('N').apply(confidence_interval).to_frame(name = 'cdf_CI')
    # remove valores (nan, nan) resultantes de amostras únicas
    ci_df = ci_df[~ci_df.cdf_CI.apply( lambda x: np.isnan(x[0]) & np.isnan(x[1]))]
    cdf = cdf_per_run.groupby('N').mean()
    cdf = cdf.rename('cdf')
    cdf_df = pd.merge(left = cdf, right = ci_df, left_on='N', right_on='N')
    return cdf_df
  
  _pdf = pdf()
  _cdf = cdf()

  result = pd.merge(left = _pdf, right = _cdf, left_on='N', right_on='N')

  return result
  
from simulations import queue_sim

# sim = queue_sim(1,1.2 , max_time= 9999, max_events=9999, runs = 10)
# print(customers_dist(sim))
#print(delay_dist(sim))