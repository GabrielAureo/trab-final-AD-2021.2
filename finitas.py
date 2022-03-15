import simulations
import pandas as pd
from math import sqrt

# 5.1: calcula a proporção de árvores finitas
def q1numerica(lamda,mu,kind,numSimulacoes=60,minimoParaInfinitude=300,runs=50,max_time=2000,max_events=2000):
	listaAmostras = []

	for i in range(0,numSimulacoes):
		trace = simulations.queue_sim(lamda,mu,max_time=2000,max_events=2000,runs=runs,kind=kind,export=False).data

		arvoresFinitas = 0
		arvoresInfinitas = 0

		for runAtual in range(0,runs):
			traceDaRun = trace[trace.run==runAtual]

			# Número de árvores finitas = número de retornos ao estado N=0
			traceN0 = traceDaRun[traceDaRun.N==0]
			arvoresFinitas += len(traceN0)

			# Contamos a árvore como infinita se ela for incompleta e tiver mais de minimoParaInfinitude passos
			if len(traceN0) > 0:
				locUltimoN0 = traceN0.index.values[-1]
				locUltimoPasso = traceDaRun.index.values[-1]
				if locUltimoPasso - locUltimoN0 >= minimoParaInfinitude:
					arvoresInfinitas += 1
			else:
				arvoresInfinitas += 1

			listaAmostras.append(arvoresFinitas/(arvoresFinitas+arvoresInfinitas))

	# Tiramos a média e intervalo de confiança
	amostras = pd.Series(listaAmostras)
	media = amostras.mean()
	z = 1.96
	s = amostras.std()
	n = numSimulacoes
	intervalo = (media - z*(s/sqrt(n)), media + z*(s/sqrt(n)))

	print("\tFração de árvores finitas (resultado numérico):", media)
	print("\tIntervalo de confiança:", intervalo)

# Como a solução analítica é a solução de uma equação, preferimos resolver sem o Python
# Essa função apenas imprime as soluções que achamos com outros meios, para questões de referência
def q1analitica(caso):
	if caso == 0 or caso == 1 or caso == 4 or caso == 5:
		res = 1
	elif caso == 2:
		res = 0.83333333
	elif caso == 3:
		res = 0.5
	elif caso == 6:
		res = 0.686302
	elif caso == 7:
		res = 0.203188

	print("\tFração de árvores finitas (resultado analítico):", res)
