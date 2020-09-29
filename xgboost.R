#Estudos para SECII
#Tutorial XGBoost - eXtreme Gradient Boosting 
#https://xgboost.readthedocs.io/en/latest/

# Definindo o diretório de trabalho
getwd()
setwd("/cloud/project")

##########
# Neste exemplo, pretendemos prever se um cogumelo pode ser comestível.
##########

# Pacotes
install.packages("xgboost")
install.packages("Ckmeans.1d.dp")
install.packages("DiagrammeR")
require(xgboost)
require(Ckmeans.1d.dp)
require(DiagrammeR)

# Datasets
# https://archive.ics.uci.edu/ml/datasets/mushroom
?agaricus.train
data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')

# Coletando subsets de treino e de teste
dados_treino <- agaricus.train
dados_teste <- agaricus.test

# Resumo dos dados, matrix
str(dados_treino)

# Visualizando as dimensões ... matrix
dim(dados_treino$data) #linhas x colunas 
dim(dados_teste$data) #dim() retorna o número de linhas e colunas em um vetor de inteiros.

# Visualizando os dados
View(dados_treino) #matrix esparsa, com vários campos em ZERO .. precisamos usar uma matrix densa para uma melhor performance
View(dados_teste)

# Classes a serem previstas
class(dados_treino$data)[1] #formato de matrix
class(dados_treino$label) #int 0 ou 1 .. label, num

###############
# MODELO
###############

?xgboost #DOCUMENTACAO DO XGBoost no R

modelo_v1 <- xgboost(data = dados_treino$data, #DADOS DE TREINO
                     label = dados_treino$label, #LABEL
                     max.depth = 2,     # Profundidade da Árvore
                     eta = 1,           # Parâmetro de Regularização para evitar Overfiting 
                     nthread = 2,       # Número de thread da CPU que vamos usar (processamento paralelo???) 
                     nround = 2,        # Número de passadas no modelo para diminuir o erro do modelo
                     objective = "binary:logistic")

# Imprimindo o modelo Baseline (modelo inicial a ser batido, otimizado)
modelo_v1

# Versão 2 do modelo, buscando melhorar a performance do modelo entregando os dados em forma de matriz densa
# Matriz Densa - Vamos entregar os dados de treino no formato que o XGBoost espera...(Matriz Densa)
?xgb.DMatrix
dtrain <- xgb.DMatrix(data = dados_treino$data, label = dados_treino$label) #DmATRIX Matrix DENSA
dtrain
class(dtrain)

#################
# OTIMIZANDO O MODELO ENTREGANDO UMA MATRIX DENSA
#################

modelo_v2 <- xgboost(data = dtrain, 
                     max.depth = 2, 
                     eta = 1, 
                     nthread = 2, 
                     nround = 2, 
                     objective = "binary:logistic")

# Imprimindo o modelo
modelo_v2

# Criando um modelo e imprimindo as etapas realizadas por meio do hiperparâmetro "verbose"
modelo_v3 <- xgboost(data = dtrain, 
                     max.depth = 2, 
                     eta = 1, 
                     nthread = 2, 
                     nround = 2, 
                     objective = "binary:logistic", 
                     verbose = 2)  # Para visualizar o progresso do aprendizado internamente. 
                                   # Com objetivo de definir os melhores parâmetros, que é a chave da qualidade do modelo.
                                   # Somente nota-se com grandes amostras devido a velocidade de ex. 

# Imprimindo o modelo
modelo_v3

# Fazendo previsões
pred <- predict(modelo_v3, dados_teste$data)

# Tamanho do vetor de previsões
print(length(pred))

# Previsões
print(head(pred))

# Transformando as previsões em classificação (int)
prediction <- as.numeric(pred > 0.5)
print(head(prediction))


###############
# AVALIAÇÃO DO MODELO
###############

#CALCULO DO ERRO
err <- mean(as.numeric(pred > 0.5) != dados_teste$label) #Compara as previsoes com os dados de teste
print(paste("test-error = ", err))

# Criando 2 matrizes densas
dtrain <- xgb.DMatrix(data = dados_treino$data, label = dados_treino$label)
dtest <- xgb.DMatrix(data = dados_teste$data, label = dados_teste$label)

###########
# TUNAGEM? DE HyperParâmetros
##########


# Criando uma watchlist
# Uma maneira de medir o progresso no aprendizado de um modelo é fornecer ao XGBoost um segundo 
# conjunto de dados já classificado. Portanto, ele pode aprender no primeiro conjunto de dados e
# testar seu modelo no segundo. Algumas métricas são medidas após cada rodada durante o aprendizado.
# A principal diferença comparado ao erro médio é que, acima, foi após a construção do modelo e agora 
# é durante a construção !!

watchlist <- list(train = dtrain, test = dtest)
watchlist

# Criando um modelo
?xgb.train
modelo_v4 <- xgb.train(data = dtrain, 
                       max.depth = 2, 
                       eta = 1, 
                       nthread = 2, 
                       nround = 2, 
                       watchlist = watchlist, # A principal diferença é que, acima, foi após a construção do modelo
                       # e agora é durante a construção que medimos os erros.
                       objective = "binary:logistic")

# Obtendo o label
label = getinfo(dtest, "label")

# Fazendo previsões
pred <- predict(modelo_v4, dtest)

# Erro
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error = ", err))

# Criando a Matriz de Importância de Atributos (like random forest)
importance_matrix <- xgb.importance(model = modelo_v4)
print(importance_matrix)

# Plot
xgb.plot.importance(importance_matrix = importance_matrix)

################
# DUMP DEPLOY BUILD MODEL
################

xgb.dump(modelo_v4, with_stats = T)

# Plot do modelo
#xgb.plot.tree(model = modelo_v1)
#xgb.plot.tree(model = modelo_v2)
#xgb.plot.tree(model = modelo_v3)
xgb.plot.tree(model = modelo_v4)


# Salvando o modelo
xgb.save(modelo_v4, "xgboost.model")

# Carregando o modelo treinado
bst2 <- xgb.load("xgboost.model") 

# Fazendo previsões com a build do modelo
pred2 <- predict(bst2, dados_teste$data)
pred2


#############################################################################################################################
# FIM # ???????
#############################################################################################################################




