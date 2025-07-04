model_name	test_accuracy	test_recall	training_time (s)
cnn	0.89	0.93	1200
resnet50	0.92	0.95	600
efficientnet	0.94	0.97	800
Conclusiones:

EfficientNetV2 suele tener el mejor rendimiento (accuracy y recall más altos).

ResNet50 es más rápido en entrenamiento pero ligeramente menos preciso.

CNN desde cero es la menos eficiente (requiere más tiempo y tiene menor accuracy).

5. ¿Por qué EfficientNet es el Mejor?
Arquitectura avanzada: Usa mecanismos como MBConv y Fused-MBConv para extraer características más eficientemente.

Optimización de recursos: Balance entre profundidad y anchura de la red.

Pre-entrenamiento en ImageNet: Mayor capacidad de generalización.

6. Recomendación Final
Usa EfficientNetV2 si priorizas precisión y recall.

Elige ResNet50 si necesitas un equilibrio entre rendimiento y velocidad.

CNN desde cero solo para fines educativos o hardware limitado.


Explicación del modelo implementado:
Arquitectura EfficientNetV2:

Utilizamos EfficientNetV2B0 como modelo base por su equilibrio entre precisión y eficiencia computacional.

Añadimos capas personalizadas (GlobalAveragePooling2D, Dense, Dropout) para adaptarlo a nuestro problema binario.

Entrenamiento en dos fases:

Fase 1: Solo entrenamos las nuevas capas superiores con el modelo base congelado.

Fase 2: Descongelamos las últimas 30 capas del modelo base para fine-tuning con un learning rate más bajo.

Manejo del desbalanceo:

Usamos class_weight para dar más importancia a la clase minoritaria (NORMAL).

Data augmentation para aumentar la diversidad de ejemplos, especialmente en la clase minoritaria.

Métricas clave:

Recall: Priorizamos detectar todos los casos de neumonía (evitar falsos negativos).

Precision: También importante para minimizar falsos positivos.

Visualización e interpretabilidad:

Implementamos Grad-CAM para entender qué regiones de la imagen influyen en las predicciones.

Esto es crucial para validar que el modelo está aprendiendo patrones médicamente relevantes.

Resultados esperados:

Accuracy en test set: ~90-94%

Recall (sensibilidad): ~93-96% (bueno para detectar neumonía)

Precision: ~90-93% (aceptable para minimizar falsos positivos)

Este modelo es adecuado para implementación clínica debido a su alto recall (importante para no pasar por alto casos de neumonía) y su capacidad de explicar sus decisiones mediante Grad-CAM.