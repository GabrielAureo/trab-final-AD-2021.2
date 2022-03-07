from __future__ import annotations
import math
from IPython.core.display_functions import display
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulations import Simulation
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
      "Analytical": simulation_obj.utilization
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
      'Analytical' : simulation_obj.average_customers
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
def wait_metric(simulation_obj : Simulation):
  simulation = simulation_obj.data
  services_df = simulation[simulation.type =='s']

  services_by_run = services_df.groupby('run')
  arrivals_df = simulation[simulation.type =='a']

  arrivals_by_run = arrivals_df.groupby('run')
  means = {
    #'delay':[],
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
    
    #arrivals_services['service_start'] = arrivals_services['service_end'] - arrivals_services['duration']

    waits = arrivals_services.service_end  - arrivals_services.arrival
    #queue_waits = arrivals_services.service_start - arrivals_services.arrival

    #means['queue_wait'].append(queue_waits.mean())
    means['wait'].append(waits.mean())
  means = pd.DataFrame(means)


  return {
    # "Delay" :{
    #   "Simulated" : means.queue_wait.mean(),
    #   "Confidence Interval" : confidence_interval(means.queue_wait),
    #   'Analytical': simulation_obj.average_queue_wait()
      
    # },

    "Wait" :{
      'Simulated' : means.wait.mean(),
      "Confidence Interval" : confidence_interval(means.wait),
      'Analytical':  simulation_obj.average_wait
      
    }
  }


def metrics(simulation_obj : Simulation):
  _wait_metrics = wait_metric(simulation_obj)
  _customers_metrics = customers_metrics(simulation_obj)
  _utilization_metric = utilization(simulation_obj)


  _wait_metrics = pd.DataFrame.from_dict(_wait_metrics, orient= 'index')
  _customers_metrics = pd.DataFrame.from_dict(_customers_metrics, orient= 'index')
  _utilization_metric = pd.DataFrame.from_dict(_utilization_metric, orient= 'index')

  metrics_df = pd.concat([_wait_metrics, _customers_metrics, _utilization_metric])\
    .stack()\
    .to_frame(name = 'Metrics')
  return metrics_df


def wait_dist(simulation_obj : Simulation):
  simulation = simulation_obj.data

  services_df = simulation[simulation.type =='s']
  services_by_run = services_df.groupby('run')
  arrivals_df = simulation[simulation.type =='a']
  arrivals_by_run = arrivals_df.groupby('run')

  waits = []
  for idx, services in services_by_run:
    services = services.reset_index()['time']
    services_count = services.shape[0]
    corresponding_arrivals = arrivals_by_run.get_group(idx).head(services_count).reset_index().time

    arrivals_services = pd.merge(left = services, right = corresponding_arrivals,\
      left_index= True, right_index= True)
    
    arrivals_services.rename(columns ={
      'time_x' : 'service_end',
      'time_y' : 'arrival'
    }, inplace = True)

    waits_N_run = arrivals_services.service_end  - arrivals_services.arrival
    waits_N_run['run'] = idx
    waits.append(waits_N_run)

  waits = pd.DataFrame(waits)
  waits.run = waits.run.astype(int)
  waits = waits.set_index('run')
  waits = waits.stack()
  waits = waits.droplevel(1)


  pdf_by_run = _histogram_by_run(waits)
  pdf = pdf_by_run.groupby('cut').mean()
  pdf_CI = pdf_by_run.groupby('cut')['wait'].apply(confidence_interval)
  cdf_by_run = pdf_by_run.groupby('run').apply(lambda x: x.cumsum())
  cdf = cdf_by_run.groupby('cut').mean()
  cdf_CI = cdf_by_run.groupby('cut')['wait'].apply(confidence_interval)

  return_df = pd.DataFrame({
    'pdf' : pdf.wait,
    'pdf_CI' : pdf_CI,
    'cdf' : cdf.wait,
    'cdf_CI' : cdf_CI
  })
  return return_df

def _histogram_by_run(waits, n_bins = 10):
  run_group = waits.groupby('run')
  mean_max = run_group.max().mean()
  mean_min = run_group.min().mean()
  bin_size = (mean_max - mean_min)/n_bins

  bins = np.linspace(mean_min, mean_max, n_bins)

  cuts = []
  for i in range(1, n_bins):
    bin = ( mean_min  + ((i - 1) * bin_size), mean_min  + (i * bin_size))
    cut = waits[ waits.between( bin[0], bin[1], inclusive = 'left') ]
    cut = cut.to_frame(name = 'wait')
    cut['cut'] = [bin for _ in range(cut.shape[0])]
    cuts.append(cut)
  
  cuts = pd.concat(cuts)
  cuts_group = cuts.groupby(['run', 'cut'])
  cuts_by_run = cuts_group.sum()
  total_wait_per_run = cuts_by_run.groupby('run').sum()
  pdf_by_run = cuts_by_run/ total_wait_per_run
  #pdf = pdf_by_run.mean(level = 'cut')
  
  # cuts = cuts.reset_index()
  # cuts = cuts.set_index(['run', 'bin'])
  # cuts = cuts.sort_index()

  # mean_delay_by_cut_and_bin = cuts.mean(level = ['run','bin'])

  # print(mean_delay_by_cut_and_bin.mean(level = 'bin'))
  # mean_delay_by_cut_and_bin.to_clipboard()
  return pdf_by_run
    

def customers_dist(simulation_obj : 'Simulation'):
  simulation = simulation_obj.data

  def pdf():

    total_state_time_per_run = simulation.groupby(['run', 'N'])['holding'].sum()
    total_time_per_run = simulation.groupby(['run'])['holding'].sum()
    pdf_per_run = total_state_time_per_run/total_time_per_run
    ci_df = pdf_per_run.groupby('N').apply(confidence_interval).to_frame(name = 'pdf_CI')
    # remove valores (nan, nan) resultantes de amostras únicas
    ci_df = ci_df[~ci_df.pdf_CI.apply( lambda x: np.isnan(x[0]) & np.isnan(x[1]))]
    
    _pdf = pdf_per_run.groupby('N').mean()
    # print(pdf_per_run)
    # _pdf = pdf_per_run.mean(level='N')
    _pdf = _pdf.rename('pdf')
    pdf_df = pd.merge(left = _pdf, right = ci_df, left_on='N', right_on='N')
    return pdf_df
  
  def cdf():
    max_state = simulation.N.max()
    total_state_time_per_run = simulation.groupby(['run', 'N'])['holding'].sum()
    total_time_per_run = simulation.groupby(['run'])['holding'].sum()
    pdf_per_run = total_state_time_per_run/total_time_per_run
    cdf_per_run = pdf_per_run.groupby('run').apply(
       lambda x: x.cumsum(axis = 0)
    )

    def fill_max(x):
      max_range = pd.Series({(x.idxmax()[0],i): 1.0 for i in range(x.idxmax()[1] + 1, max_state + 1)}, dtype=np.float64)
      x = pd.concat([x,max_range])
      return x

    cdf_per_run = cdf_per_run.groupby('run').apply(fill_max)
    #print(cdf_per_run)
    _cdf = cdf_per_run.groupby('N').mean()
    _cdf = _cdf.rename('cdf')

    ci_df = cdf_per_run.groupby('N').apply(confidence_interval).to_frame(name = 'cdf_CI')
    #remove valores (nan, nan) resultantes de amostras únicas
    ci_df = ci_df[~ci_df.cdf_CI.apply( lambda x: np.isnan(x[0]) & np.isnan(x[1]))]
    
    cdf_df = pd.merge(left = _cdf, right = ci_df, left_on='N', right_on='N')
    return cdf_df
  
  _pdf = pdf()
  _cdf = cdf()
  result = pd.merge(left = _pdf, right = _cdf, left_on='N', right_on='N')
  result.sort_index(inplace=True)
  return result

