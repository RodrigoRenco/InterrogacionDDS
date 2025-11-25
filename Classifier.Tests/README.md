# InterrogacionDDS
Codigo de Interrogacion DDS
public ClassificationResult Handle(ClassificationRequest request)
{
    try
    {
        ValidateRequest(request);                 // 12 errores van aquí
        var datasets = _datasetLoader.Load(request.Dataset);

        var encoder = _encoderFactory.Create(request.EncoderType);
        encoder.Fit(datasets.Train);

        var classifier = _classifierFactory.Create(request, encoder);
        classifier.Train(datasets.Train);

        var metric = _metricFactory.Create(request.Metric);

        var score = EvaluateModel(classifier, metric, datasets.Test);

        return ClassificationResult.Success(score);
    }
    catch (ClassifierException ex)
    {
        return ClassificationResult.Error(ex.Message);
    }
}


Fase 1 — Orientación rápida

Cuando tengas el proyecto en Rider:

Corre todos los tests y confirma que pasan.

Abre:

Controller.cs o quien tenga Handle.

Las clases de:

Encoders

Clasificadores

Métricas

Dataset (cómo lee los JSON / CSV / lo que sea).

Localiza:

El método gigante (seguro Handle está lleno de if / switch según strings).

Dónde se validan los 12 casos (probablemente mezclados con todo lo demás).

Dónde se hace I/O (leer dataset, etc.).

No refactores nada todavía, solo ubica los “monstruos”.

Fase 2 — Names + Functions (limpieza básica para abrir el camino)

Objetivo: que el código se pueda leer con calma.

Renombra variables horribles (r, d, etc.) a cosas claras:

trainDataset, testDataset, encoder, classifier, metric, etc.

Extrae funciones privadas pequeñas desde Handle:

LoadDatasets(...)

ValidateRequest(...)

CreateEncoder(...)

CreateClassifier(...)

CreateMetric(...)

TrainClassifier(...)

ComputeScore(...)

Aunque temporalmente sigan siendo privados dentro de Controller, ayuda muchísimo para ver dónde cortar en clases después.

⚠️ No cambies firmas públicas que los tests usan (como Handle).

Fase 3 — Manejo de errores (los 12 casos)

Todo ese listado de 12 condiciones de error es oro para el capítulo de Error Handling. 

Interrogación_2025_2

Tienes dos estilos posibles (elige uno y sé consistente):

Estilo A — Excepciones de dominio (similar a lo que hiciste en LMEval)

Creas una clase base:

ClassifierException : Exception

Luego específicas:

DatasetNotFoundException

EmptyDatasetException

InvalidEncoderException

InvalidClassifierException

InvalidMetricException

MissingLearningRateException

NegativeLearningRateException

MissingEpochsException

NegativeEpochsException

MissingKException

InvalidKException

TooLargeKException

En Handle, envuelves el flujo normal en un try y haces:

catch (ClassifierException ex)
{
    return ClassificationResult.Error(ex.Message);
}

Estilo B — Validador centralizado

Creas un RequestValidator que revisa TODOS los 12 casos:

public void Validate(ClassificationRequest request, DatasetInfo datasets)


y lanza excepciones o devuelve un resultado de error.

Dado que tú ya estás cómodo con excepciones desde LMEval, Estilo A encaja perfecto contigo.

Fase 4 — Clases y Factories (el gran golpe de diseño)

Aquí es donde puedes sumar MUCHO:

4.1. Interfaces del dominio

Es muy probable que el código YA tenga algo así, pero si está mezclado:

IEncoder con:

Fit(trainData)

Transform(featureVector)

Implementaciones:

LabelEncoder

OneHotEncoder

IClassifier con:

Train(trainData)

Predict(encodedFeatures) o PredictMany(...)

Implementaciones:

KnnClassifier

LogisticRegressionClassifier

IMetric con:

ComputeScore(predictions, trueLabels)

Implementaciones:

AccuracyMetric

RecallMetric

Tu trabajo es:

separar responsabilidades SI está todo mezclado,

o al menos usar factories para construirlos desde Controller.

4.2. Factories

Factories limpian esos switch / if por string:

EncoderFactory.Create(encoderType)

ClassifierFactory.Create(classifierType, encoder, learningRate, epochs, k)

MetricFactory.Create(metricType)

Así Handle no tiene un switch por tipo, solo delega.

Esto puntúa en:

Classes

Objects & Data Structures

Boundaries (si aislas bibliotecas externas)

Fase 5 — Boundaries y carga de datos

Boundaries = cualquier cosa que toque el mundo externo:

leer dataset de entrenamiento y test

posiblemente leer configuraciones desde archivos

Súper buen candidato:

DatasetLoader o DatasetRepository, que se encargue de:

Construir path para train/test a partir de Dataset.

Verificar existencia de archivos.

Cargar contenido.

Detectar si están vacíos.

Devolver algo como:

public class DatasetPair
{
    public IReadOnlyList<(Features, Label)> Train { get; }
    public IReadOnlyList<(Features, Label)> Test  { get; }
}


Luego, en Handle solo haces:

var datasets = _datasetLoader.Load(request.Dataset);
// y la validación de “no vacío” o “no encontrado” sale de ahí


De nuevo, más puntos para:

Boundaries

Error Handling

Objects & Data Structures

Fase 6 — Métricas (Accuracy y Recall)

Recuerda lo que dice el enunciado: 

Interrogación_2025_2

Accuracy = proporción de predicciones correctas.

Recall = accuracy promedio por clase (particionas por label y luego promedias).

Buena forma de organizar:

IMetric con double Compute(...).

AccuracyMetric

RecallMetric

MetricFactory para mapear "accuracy" / "recall".

En Handle, solo tienes:

var metric = _metricFactory.Create(request.Metric);
var score  = metric.Compute(predictions, trueLabels);

4. Checklist mental cuando estés en la prueba

Cuando ya estés metido en el código, usa esto como guía rápida:

Tests pasan antes de tocar nada.

Handle NO debería:

leer archivos directamente,

hacer 10 switch por strings,

hacer cálculos numéricos complejos,

tener if gigantes con los 12 casos mezclados.

Handle SÍ debería:

verse como pipeline limpio (validar → cargar datasets → crear encoder → crear classifier → entrenar → crear métrica → evaluar).

Las responsabilidades clave deberían estar en clases separadas:

DatasetLoader / similar.

EncoderFactory, ClassifierFactory, MetricFactory.

Interfaces IEncoder, IClassifier, IMetric.

Excepciones de dominio claras para los 12 casos.

No cambies el comportamiento observable:

mismos strings de error,

mismo flujo general,

misma métrica calculada.