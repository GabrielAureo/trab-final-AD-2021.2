import simulations
import pandas as pd
import queue
from treelib import Tree,Node
from math import pow,exp,factorial,sqrt
import matplotlib.pyplot as plt


Z = 1.96 # Grau de confiança a ser usado em todas as questões

# Cria árvores (processos de ramificação) baseado em um trace da simulação
def gerarArvores(lamda,mu,kind,runs=300,max_time=3000,max_events=3000,minimoParaInfinitude=300):
	trace = simulations.queue_sim(lamda,mu,max_time=max_time,max_events=max_events,runs=runs,kind=kind,export=False).data

	arvores = []
	for runAtual in range(0,runs):
		traceDaRun = trace[trace.run==runAtual]

		filaDeNos = queue.Queue()
		arvAtual = None
		noAtual = None

		for passo in range(0,len(traceDaRun)):
			if traceDaRun.iloc[passo]['type'] == 'a':
				if noAtual == None: # Se há uma chegada e nenhum nó está sendo servido, temos um período ocupado novo
					arvAtual = Tree()
					arvores.append(arvAtual)
					arvAtual.create_node("Chegada no instante " + str(passo), passo)
					noAtual = passo
				else: # Se há uma chegada, o novo cliente é filho do cliente que está sendo servido no momento
					arvAtual.create_node("Chegada no instante " + str(passo), passo, noAtual)
					filaDeNos.put(passo) # Temos uma fila de nós para que todos os clientes constem como servidos na ordem certa
			else:
				if not filaDeNos.empty():
					noAtual = filaDeNos.get()
				else:
					noAtual = None

		# Consideramos árvores infinitas, mas as marcamos como tal
		if traceDaRun.iloc[len(traceDaRun)-1]['N'] != 0:
			if arvAtual.size() <= minimoParaInfinitude:
				arvores.remove(arvAtual)
			else:
				idRaiz = arvAtual.root
				arvAtual.get_node(idRaiz).data = "inf"

	return arvores

def plotarCDF(dicionario): # Plota a CDF dado uma distribuição em forma de dicionário
	largura = 1.0
	cdf = {}

	for i in range(0,max(dicionario.keys())): # "Normaliza" o dicionário para incluir todas as frequências no intervalo, mesmo que sejam 0
		if i not in dicionario:
			dicionario[i] = 0

	for valor in dicionario:
		cdf[valor] = 0
		for i in range(valor,-1,-1):
			cdf[valor] += dicionario[i]

	cdfSort = dict(sorted(cdf.items()))
	plt.plot(cdfSort.keys(), cdfSort.values(), color='r', marker='o')
	plt.show()

# Solução numérica do subitem 1 do item 5.2
# Calcula distribuição dos graus de saída
def q1numerica(lamda,mu,kind,runs=300,max_time=3000,max_events=3000):
	trace = simulations.queue_sim(lamda,mu,max_time=3000,max_events=3000,runs=runs,kind=kind,export=False).data

	grausDeSaida = {}
	nosTotais = 1
	for runAtual in range(0,runs):
		#print("Percorrendo run", runAtual+1, "de", runs)
		traceDaRun = trace[trace.run==runAtual]

		chegadasNesseNo = -1 # Começamos em -1 para que a primeira chegada não conte a si mesma

		for passo in range(0,len(traceDaRun)):
			# Como cada nó equivale ao início de um serviço, contamos as chegadas entre serviços
			if traceDaRun.iloc[passo]['type'] == 'a':
				chegadasNesseNo += 1
			else:
				if chegadasNesseNo not in grausDeSaida:
					grausDeSaida[chegadasNesseNo] = 1
				else:
					grausDeSaida[chegadasNesseNo] += 1
				if traceDaRun.iloc[passo]['N'] == 0:
					chegadasNesseNo = -1 # O próximo nó vai ser uma raiz, então fazemos como acima
					nosTotais += 1
				else:
					chegadasNesseNo = 0
					nosTotais += 1

	for indice in grausDeSaida: # Transformamos as quantidades absolutas de ocorrências em proporções
		grausDeSaida[indice] = grausDeSaida[indice]/nosTotais

	print("Distribuição dos graus de saída (numérica):\n\t", grausDeSaida)
	print("\tCDF:")
	plotarCDF(grausDeSaida)

# Solução analítica do subitem 1 do item 5.2
# Calcula distribuição de graus de saída (até um certo ponto)
# Mostramos como as fórmulas são derivadas no relatório
def q1analitica(lamda,mu,kind):
	distr = {}
	res = 1
	xAtual = 0
	while res > 0.000005:
		if kind=='m':
			res = (pow(lamda,xAtual)*mu)/(pow((lamda+mu),xAtual+1))
		else:
			res = (pow((lamda/mu),xAtual)*exp(-lamda/mu))/factorial(xAtual)
		distr[xAtual] = res
		xAtual += 1

	print("Distribuição dos graus de saída (analítica):\n\t", distr)
	print("\tCDF:")
	plotarCDF(distr)

# Solução numérica do subitem 2 do item 5.2
# Calcula grau médio de saída da raiz, já tendo as árvores construídas
def q2numerica(arvores):
	graus = []

	for arvore in arvores:
		graus.append(len(arvore.children(arvore.root))) # Tamanho da lista de filhos da raiz, i.e., o grau de saída

	amostras = pd.Series(graus)
	media = amostras.mean()
	s = amostras.std()
	n = amostras.count()
	if n>0:
		intervalo = (media - Z*(s/sqrt(n)), media + Z*(s/sqrt(n)))
	else:
		intervalo = (0,0)

	print("Grau médio de saída da raiz (numérico):", media)
	print("\tIntervalo de confiança:", intervalo)

# Solução analítica do subitem 2 do item 5.2
# Calcula grau médio de saída da raiz, usando lamda e mu
# A maneira como derivamos a fórmula é explicada no relatório
def q2analitica(lamda,mu,kind):
	print("\tGrau médio de saída da raiz (analítico):", lamda/mu)

# Solução numérica do subitem 3 do item 5.2
# Calcula média do grau de saída máximo, já tendo as árvores construídas
def q3numerica(arvores):
	grausMaximos = []

	for arvore in arvores:
		dicDeNos = arvore.nodes # Obtem um índice com todos os nós da árvore
		max = 0
		for no in dicDeNos: # Achamos o grau máximo da árvore
			grau = len(arvore.children(no)) # Grau de saída desse nó em particular
			if grau > max:
				max = grau
		grausMaximos.append(max)

	amostras = pd.Series(grausMaximos)
	media = amostras.mean()
	s = amostras.std()
	n = amostras.count()
	if n>0:
		intervalo = (media - Z*(s/sqrt(n)), media + Z*(s/sqrt(n)))
	else:
		intervalo = (0,0)

	print("Média do grau de saída máximo (numérica):", media)
	print("\tIntervalo de confiança:", intervalo)

# Solução numérica do subitem 4 do item 5.2
# Calcula altura média da árvore, já tendo as árvores construídas
# As alturas começam em 0 (isso é, uma árvore que só tem a raiz tem altura 0)
def q4numerica(arvores):
	alturas = []

	for arvore in arvores:
		if arvore.get_node(arvore.root).data == "inf": # Se árvore for infinita, consideramos altura infinita
			alturas.append(float('inf'))
		else:
			alturas.append(arvore.depth()) # .depth() retorna a altura da árvore

	amostras = pd.Series(alturas)
	media = amostras.mean()
	s = amostras.std()
	n = amostras.count()
	if n>0:
		intervalo = (media - Z*(s/sqrt(n)), media + Z*(s/sqrt(n)))
	else:
		intervalo = (0,0)

	print("Altura média das árvores (numérica):", media)
	print("\tIntervalo de confiança:", intervalo)

# Calcula funções geradoras dos tamanhos das gerações, usando recursão (fórmula 4.4 do Dobrow)
# Função auxiliar para o resultado pseudo-analítico do subitem 4
def recursaoGeradora(lamda,mu,vezes,kind):
	if kind=='m':
		if vezes == 0:
			return 0
		if vezes == 1:
			return mu/(lamda+mu)
		else:
			return mu/(mu + (lamda*(1-recursaoGeradora(lamda,mu,vezes-1,kind))))

	else:
		if vezes == 0:
			return 0
		if vezes == 1:
			return exp(-lamda/mu)
		else:
			return exp((lamda/mu)*(-1+recursaoGeradora(lamda,mu,vezes-1,kind)))

# Aproxima a solução analítica do subitem 4 do item 5.2
# Aproxima a altura média da árvore
# Detalhes sobre o procedimento estão no relatório
def q4analitica(lamda,mu,kind):
	if lamda >= mu:
		total = None # Consideramos os resultados indefinidos nesses casos; isso é explicado no relatório
	else:
		total = 0
		a = 1
		res = 0
		while a < 20 or res > 0.0001:
			res = a*(recursaoGeradora(lamda,mu,a+1,kind) - recursaoGeradora(lamda,mu,a,kind))
			total += res
			a += 1

	print("\tAltura média das árvores (aproximação do somatório):", total)

# Calcula a solução numérica do subitem 5 do item 5.2
# Acha a média das alturas dos nós, tendo as árvores
# Como no subitem 4, as alturas começam em 0 (altura da raiz é 0)
def q5numerica(arvores):
	alturas = []

	for arvore in arvores:
		if arvore.get_node(arvore.root).data == "inf": # Se árvore infinita, consideramos sua média infinita
			alturas.append(float('inf'))
		else:
			dicDeNos = arvore.nodes # Lista de nós da árvore
			for no in dicDeNos:
				alturas.append(arvore.depth(no)) # Coletamos todas as alturas

	amostras = pd.Series(alturas)
	media = amostras.mean()
	s = amostras.std()
	n = amostras.count()
	if n>0:
		intervalo = (media - Z*(s/sqrt(n)), media + Z*(s/sqrt(n)))
	else:
		intervalo = (0,0)

	print("Média das alturas dos nós (numérica):", media)
	print("\tIntervalo de confiança:", intervalo)

# Calcula a solução analítica do subitem 5 do item 5.2
# Acha a média das alturas dos nós
# As fórmulas são derivadas no relatório
def q5analitica(lamda,mu,kind):
	if lamda>=mu:
		total = float('inf')
	else:
		total = -lamda/(lamda-mu)

	print("\tMédia das alturas dos nós (analítica):", total)

# Calcula a solução numérica do subitem 6 do item 5.2
# Acha a média da duração do período ocupado
# Roda uma simulação nova, pois as árvores não nos seriam muito úteis
def q6numerica(lamda,mu,kind,runs=300,max_time=3000,max_events=3000,minimoParaInfinitude=300):
	trace = simulations.queue_sim(lamda,mu,max_time=3000,max_events=3000,runs=runs,kind=kind,export=False).data

	duracoes = []

	for runAtual in range(0,runs):
		traceDaRun = trace[trace.run==runAtual]

		for passo in range(0,len(traceDaRun)):
			if traceDaRun.iloc[passo]['type'] == 'a' and traceDaRun.iloc[passo]['N'] == 1: # Começo de uma árvore
				tempoInicial = traceDaRun.iloc[passo]['time']
				passoInicial = passo
			if traceDaRun.iloc[passo]['type'] == 's' and traceDaRun.iloc[passo]['N'] == 0: # Final de uma árvore
				tempoFinal = traceDaRun.iloc[passo]['time']
				duracoes.append(tempoFinal-tempoInicial) # Tiramos o tempo total da árvore
		if traceDaRun.iloc[len(traceDaRun)-1]['N'] != 0 and len(traceDaRun)-1-passoInicial > minimoParaInfinitude:
			# Se árvore infinita, consideramos tempo como infinito
			duracoes.append(float('inf'))

	amostras = pd.Series(duracoes)
	media = amostras.mean()
	z = 1.96
	s = amostras.std()
	n = amostras.count()
	intervalo = (media - z*(s/sqrt(n)), media + z*(s/sqrt(n)))

	print("Média da duração do período ocupado (numérica):", media)
	print("\tIntervalo de confiança:", intervalo)

# Solução analítica do subitem 6 do item 5.2
# Acha a média da duração do período ocupado
# A fórmula é explicada no relatório
def q6analitica(lamda,mu,kind):
	if lamda >= mu:
		total = float('inf')
	else:
		total = -1/(lamda-mu)

	print("\tMédia da duração do período ocupado (analítica):", total)

# Solução numérica do subitem 7 do item 5.2
# Acha a média do número de clientes atendidos por período ocupado, tendo as árvores
def q7numerica(arvores):
	numerosDeClientes = []

	for arvore in arvores:
		if arvore.get_node(arvore.root).data == "inf": # Se a árvore é infinita consideramos o no. de clientes infinito
			numerosDeClientes.append(float('inf'))
		else:
			numerosDeClientes.append(arvore.size()) # .size() retorna o número de nós da árvore

	amostras = pd.Series(numerosDeClientes)
	media = amostras.mean()
	s = amostras.std()
	n = amostras.count()
	if n>0:
		intervalo = (media - Z*(s/sqrt(n)), media + Z*(s/sqrt(n)))
	else:
		intervalo = (0,0)

	print("Média do número de clientes (numérica):", media)
	print("\tIntervalo de confiança:", intervalo)

# Solução analítica do subitem 7 do item 5.2
# Acha média do número de clientes atendidos por período ocupado
# A fórmula é explicada no relatório
def q7analitica(lamda,mu,kind):
	if lamda >= mu:
		total = float('inf')
	else:
		total = -mu/(lamda-mu)

	print("\tMédia do número de clientes (analítica):", total)
