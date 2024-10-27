import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('d:\\data science\\imdb_labelled.txt', sep='\t', names=['text', 'target'])
print(df.head())


#_______________________________________________________________

contagem_Classes_pra_balanceamento = df.groupby('target').count()
total = contagem_Classes_pra_balanceamento.sum()
porcentagem = contagem_Classes_pra_balanceamento / total * 100
print('porcentagem de cada classe')
print(porcentagem)
#_______________________________________________________________



#_______________________________________________________________
texto_com_emails = df.drop('target', axis=1)
X = texto_com_emails
print('texto com emails')
print(X)
#_______________________________________________________________

#_______________________________________________________________
resultado_spam_ou_ham = df['target']
y = resultado_spam_ou_ham
print('resultado spam ou ham')
print(y)
#_______________________________________________________________


# _________________________________________________________________
x_de_treino, x_de_teste, y_de_treino, y_de_teste = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = x_de_treino
print('X_train')
print(X_train )


X_test = x_de_teste
print('X_test')
print(X_test)

y_train = y_de_treino
print('y_train')
print(y_train)

y_test = y_de_teste
print('y_test')
print(y_test)
# _________________________________________________________________


# _________________________________________________________________
funcao_pra_vetorizar = TfidfVectorizer(stop_words='english')
vectorizer = funcao_pra_vetorizar
# _________________________________________________________________


# _________________________________________________________________
espaco_de_caracteristica_e_vetorizacao = vectorizer.fit_transform(X_train['text'])
X_train_tfidf = espaco_de_caracteristica_e_vetorizacao

print('X_train_tfidf')
print(X_train_tfidf)
# _________________________________________________________________


# _________________________________________________________________
veroticacao = vectorizer.transform(X_test['text'])
X_test_tfidf = veroticacao
print('X_test_tfidf')
print(X_test_tfidf)

# _________________________________________________________________

model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)
# _________________________________________________________________


# _________________________________________________________________

probabilidade_de_positivar = model.predict_proba(X_test_tfidf)[:, 1]
y_pred_proba = probabilidade_de_positivar
print('y_pred_proba') 
print(y_pred_proba) 
# _________________________________________________________________

# _________________________________________________________________

probabilidade_de_positivar_dos_positivos= model.predict_proba(X_test_tfidf)[y_test == 1][:, 1]
y_pred_proba2 = probabilidade_de_positivar_dos_positivos

print('y_pred_proba2')
print(y_pred_proba2)
# _________________________________________________________________

y_pred_proba2 = y_pred_proba2.tolist()  
y_pred_proba = y_pred_proba.tolist()  


import numpy as np
import matplotlib.pyplot as plt


bins = np.arange(0, 1.1, 0.1)


hist_y_pred_proba, _ = np.histogram(y_pred_proba, bins=bins)
hist_y_pred_proba2, _ = np.histogram(y_pred_proba2, bins=bins)


bar_width = 0.35
x_pos = np.arange(len(hist_y_pred_proba))  


plt.figure(figsize=(10, 6))


plt.bar(x_pos, hist_y_pred_proba, width=bar_width, label='Toda a População', color='blue')


plt.bar(x_pos + bar_width, hist_y_pred_proba2, width=bar_width, label='População Positivada', color='green')


plt.xticks(x_pos + bar_width / 2, [f'{int(b * 100)}%' for b in bins[:-1]])


plt.title('Distribuição de Probabilidades de Spam (Acumulado)')
plt.xlabel('Probabilidade de ser Spam (%)')
plt.ylabel('Quantidade Acumulada (%)')


plt.legend()


plt.show()




from sklearn.metrics import roc_curve, auc


false_positive_rate, true_positive_rate , thresholds = roc_curve(y_test, y_pred_proba)
fpr = false_positive_rate
tpr = true_positive_rate



roc_auc = auc(fpr, tpr)

# Plotando a curva ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linha diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid()
plt.show()
