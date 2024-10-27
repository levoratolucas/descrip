import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier


df = pd.read_csv('d:\\data science\\amazon_cells_labelled.txt', sep='\t', names=['text', 'target'])
# print(df.head())


contagem_Classes = df.groupby('target').count()
total = contagem_Classes.sum()
porcentagem = contagem_Classes / total * 100
# print(porcentagem)


X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train['text'])

X_test_tfidf = vectorizer.transform(X_test['text'])


model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)


y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]
y_pred = model.predict(X_test_tfidf)
y_pred_proba2 = model.predict_proba(X_test_tfidf)[:, 1]
# print(y_pred_proba2)
# print('y_pred')
# print(y_pred)

populacao = y_pred_proba

populacao_positivado = y_pred_proba[y_test == 1]


populacao_positivado = populacao_positivado.tolist()  
populacao = populacao.tolist()  


# import numpy as np
# import matplotlib.pyplot as plt


# bins = np.arange(0, 1.1, 0.1)


# hist_populacao, _ = np.histogram(populacao, bins=bins)
# hist_populacao_positivado, _ = np.histogram(populacao_positivado, bins=bins)


# bar_width = 0.35
# x_pos = np.arange(len(hist_populacao))  


# plt.figure(figsize=(10, 6))


# plt.bar(x_pos, hist_populacao, width=bar_width, label='Toda a População', color='blue')


# plt.bar(x_pos + bar_width, hist_populacao_positivado, width=bar_width, label='População Positivada', color='green')


# plt.xticks(x_pos + bar_width / 2, [f'{int(b * 100)}%' for b in bins[:-1]])


# plt.title('Distribuição de Probabilidades de Spam (Acumulado)')
# plt.xlabel('Probabilidade de ser Spam (%)')
# plt.ylabel('Quantidade Acumulada (%)')


# plt.legend()


# plt.show()

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


print('Acurácia:', accuracy)
print('Precisão:', precision)
print('Recall:', recall)
# Cálculo da curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



print(thresholds)



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