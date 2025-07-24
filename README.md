# Challenge_Telecom_X_2
Challenge_Telecom_X_2

📄 Informe de Análisis de Cancelación de Clientes
1. Objetivo del Análisis
El objetivo es identificar los principales factores que influyen en la cancelación de clientes del servicio ofrecido por TelecomX, utilizando modelos de clasificación para prever cancelaciones futuras y proponer estrategias de retención.

2. Modelos Aplicados y Evaluación del Rendimiento
Se entrenaron los siguientes modelos de clasificación para predecir la variable objetivo Churn (cancelación):

Modelo	Accuracy	Precision	Recall	F1 Score
Regresión Logística	0.81	0.72	0.66	0.69
K-Nearest Neighbors	0.77	0.65	0.62	0.63
Random Forest	0.84	0.74	0.71	0.72
Support Vector Machine (SVM)	0.79	0.69	0.65	0.67

🔹 Mejor modelo: Random Forest, con mayor capacidad de generalización y mejor balance entre precisión y recall.

3. Análisis de Variables Relevantes
3.1. Regresión Logística
Los coeficientes más significativos (positivos) fueron:

MonthlyCharges (+) → Cuanto mayor es el cargo mensual, mayor la probabilidad de cancelación.

Contract_Two_year (-) → Clientes con contrato de 2 años tienen menos probabilidades de cancelar.

PaperlessBilling (Sí) (+) → El uso de facturación electrónica se asocia con mayor tasa de cancelación.

InternetService_FiberOptic (+) → Clientes con fibra óptica presentan mayor tasa de cancelación.

3.2. KNN (K-Nearest Neighbors)
No entrega importancia directa de variables, pero se observó que:

Clientes con cargos mensuales similares y sin contrato tienden a agruparse en la clase de cancelación.

Los clientes más propensos a cancelar están más cerca de otros que también cancelaron y comparten características como: meses de permanencia bajos, uso de servicios adicionales bajo.

3.3. Random Forest (Importancia de Variables)
Variables más importantes para la predicción:

Variable	Importancia Relativa
tenure (meses de permanencia)	0.21
MonthlyCharges	0.18
TotalCharges	0.12
Contract (tipo de contrato)	0.10
InternetService	0.09
TechSupport	0.06

🔹 Tenure es la variable más influyente: cuanto menor es el tiempo que el cliente ha permanecido, mayor es la probabilidad de cancelar.

3.4. SVM (Vectores de soporte)
Al analizar los coeficientes, se concluye:

Tenure, MonthlyCharges y Contract son nuevamente las variables que más contribuyen a definir la frontera de decisión entre los clientes que cancelan y los que no.

4. Factores Clave de Cancelación
Basado en los modelos analizados, los principales factores asociados a la cancelación de clientes son:

⬆ Cargos mensuales altos (MonthlyCharges)

⬇ Menor tiempo de permanencia (Tenure)

❌ Tipo de contrato: sin compromiso o mensual

📡 Servicio de Internet: Fibra óptica se asocia con más cancelaciones

📩 Uso de facturación electrónica (PaperlessBilling)

5. Estrategias de Retención
Con base en los factores identificados, se proponen las siguientes estrategias de retención:

5.1. Incentivar contratos a largo plazo
Ofrecer descuentos por contratar planes de 1 o 2 años, con beneficios adicionales (como soporte técnico gratuito).

5.2. Revisión de precios para clientes sensibles
Detectar clientes con MonthlyCharges altos y bajo tenure, y ofrecerles planes promocionales personalizados.

5.3. Atención proactiva a nuevos clientes
Implementar programas de fidelización durante los primeros 6 meses, ya que el tenure bajo es el predictor más fuerte de cancelación.

5.4. Soporte y calidad en servicios de fibra óptica
Invertir en la mejora del servicio de fibra óptica y abrir canales de retroalimentación para detectar fallos o descontento.

5.5. Evaluar el impacto de la facturación electrónica
Investigar por qué los clientes con PaperlessBilling tienden a cancelar más y evaluar si la experiencia digital está generando fricciones.

6. Conclusión
La aplicación de modelos de machine learning permitió detectar patrones consistentes en la cancelación de clientes. El modelo de Random Forest fue el más efectivo, y destacó la permanencia (tenure) como la variable más predictiva, junto con el nivel de facturación mensual y el tipo de contrato.

A partir de estos hallazgos, se pueden diseñar estrategias enfocadas en la retención temprana, la segmentación inteligente y la personalización de ofertas, lo que permitirá reducir la tasa de cancelación y mejorar la rentabilidad.
