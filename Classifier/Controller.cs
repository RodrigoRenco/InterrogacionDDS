using System.Text.Json;
using Classifier.AuxiliarClases;
using Classifier.ClassificationExceptions;
using Classifier.Encoders;

namespace Classifier;

public class Controller //ESTE ES EL COTOTO, 400 lines
{
    public readonly Dictionary<string, int> _vocabulary = new();
    public readonly List<Tuple<double[], string>> _trainingData = new();
    public readonly Dictionary<string, double[]> _weightsByLabel = new();

    private TrainingDataSetLoader _TrainingDataSetLoader;
    private TestingDataSetLoader _TestingDataSetLoader;
    private EncoderHandler _encoderHandler;
    
    public List<Sample> trainDataset;
    public List<Sample> testDataset;

    //Classes y facories nuevas
    //TrainingDataSetLoader() listo
    //TestingDataSetLoader() listo
    
    //FACTORY de Encoder, hay dos grupos, 
    //FACTORY DE TrainClasiffier
    //FACTORY de PREDICT
    
    //ObtainAllTestDatasetLabels()
    //ObtainAllTestDatasetFeatures()
    public ClassificationResult Handle(ClassificationRequest request)
    {
        ClassificationResult result = new ClassificationResult();

        _TrainingDataSetLoader = new();
        _TestingDataSetLoader = new();
        _encoderHandler = new();

        try
        {
            trainDataset = _TrainingDataSetLoader.ObtainDataset(request.Dataset);
            testDataset = _TestingDataSetLoader.ObtainDataset(request.Dataset);
        }
        catch (ClassificationExceptionBase error)
        {
            result.ProcessException(error.Message);
            return result;
        }

        //EcoderHandler();
        // Fit the encoder with the training dataset //REVISAR Esto es arreglable, mas manejo de error
        if (request.EncoderType == "label")
            FitLabelEncoder(trainDataset);
        else if(request.EncoderType == "one-hot")
            FitOneHotEncoder(trainDataset);
        else
        {
            result.ErrorMessage = "Invalid encoder type";
            return result;
        }
        // We expect to add new encoders in the future //SIGNIFICA HACER FACTORY de Fit-Encoder
        
        // Train the classifier with the training dataset //ACA TAMBIEN FACTORY DE TrainClasiffier
        (bool, string?) trainResult;
        if (request.ClassifierType == "knn")
            trainResult = TrainKNNClassifier(trainDataset, request.K, request.EncoderType);
        else if (request.ClassifierType == "logistic-regression")
            trainResult = TrainLogisticRegressionClassifier(trainDataset, request.Epochs, request.LearningRate, request.EncoderType);
        else
            trainResult = (false, "Invalid classifier type");
        if (!trainResult.Item1)
        {
            result.ErrorMessage = trainResult.Item2;
            return result;
        }
        // We expect to add new classifiers in the future //SIGNIFICA HACER FACTORY TrainClasiffier
        
        
        // Compute metric //OTRA Clase que se encargue de esto, mas FACTORY de PREDICT
        if (request.Metric == "accuracy")
        {
            int correct = 0;
            for (int i = 0 ; i < testDataset.Count; i++)
            {
                Sample sample = testDataset[i];
                (bool, string) prediction;
                // Predict label
                if (request.ClassifierType == "knn")
                    prediction = PredictKNN(sample.Features, request.EncoderType, request.K);
                else if (request.ClassifierType == "logistic-regression")
                    prediction = PredictLogisticRegression(sample.Features, request.EncoderType);
                else
                    prediction = (false, "Invalid classifier type");
                if (!prediction.Item1)
                {
                    result.ErrorMessage = prediction.Item2;
                    return result;
                }
                // We expect to add new classifiers in the future //SIGNIFICA HACER FACTORY de PREDICT
                
                if (prediction.Item2 == sample.Label)
                    correct++;
            }

            result.Score = (double) correct/testDataset.Count;
        }
        else if (request.Metric == "recall")
        {
            // Get all distinct labels from the test dataset //CLASE que se haga cargo: ObtainAllTestDatasetLabels()
            HashSet<string> labels = new();
            foreach (Sample sample in testDataset)
                labels.Add(sample.Label);
            string[] classes = labels.ToArray();

            // Compute accuracy for each class
            List<double> recalls = new();

            foreach (string label in classes) //Clase o funcion aparte, CODIGO REPETIDO?
            {
                Sample[] samples = testDataset.Where(p => p.Label == label).ToArray();
                int correct = 0;
                for (int i = 0; i < samples.Length; i++)
                {
                    Sample sample = samples[i];
                    (bool, string) prediction;
                    // Predict label
                    if (request.ClassifierType == "knn")
                        prediction = PredictKNN(sample.Features, request.EncoderType, request.K);
                    else if (request.ClassifierType == "logistic-regression")
                        prediction = PredictLogisticRegression(sample.Features, request.EncoderType);
                    else
                        prediction = (false, "Invalid classifier type");
                    if (!prediction.Item1)
                    {
                        result.ErrorMessage = prediction.Item2;
                        return result;
                    }
                    // We expect to add new classifiers in the future

                    if (sample.Label == prediction.Item2)
                        correct++;
                }

                recalls.Add((double)correct / samples.Length);
            }

            result.Score = recalls.Average();
        }
        else
            result.ErrorMessage = "Invalid metric";
        // We expect to add more metrics in the future

        return result;
    }
    
    private void FitLabelEncoder(List<Sample> trainingDataset) //ESTA FEO, creo que sacable
    {
        _vocabulary.Clear();
        
        // Get all distinct features from the dataset //NUEVA CLASE, se usaria dos veces
        HashSet<string> features = new();
        foreach (Sample sample in trainingDataset)
        foreach (string feature in sample.Features)
            features.Add(feature);
        string[] allTokens = features.ToArray();
        
        // Build vocabulary mapping
        int index = 0;
        foreach (string token in allTokens)
            _vocabulary[token] = index++;
    }

    private double[] EncodeLabel(string[] sample) //REVISAR, que es vec, Nombre, muchos niveles
    {
        double[] vector = new double[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            string token = sample[i];
            if (_vocabulary.ContainsKey(token))
            {
                int idx = _vocabulary[token];
                vector[i] = idx;
            }
        }
        return vector;
    }
    
    private void FitOneHotEncoder(List<Sample> trainingDataset)  //ESTA FEO creo que sacable
    {
        _vocabulary.Clear();
        
        // Get all distinct features from the dataset
        HashSet<string> features = new();
        foreach (Sample sample in trainingDataset)
        foreach (string feature in sample.Features)
            features.Add(feature);
        string[] allTokens = features.ToArray();
        
        // Build vocabulary mapping
        int index = 0;
        foreach (string token in allTokens)
            _vocabulary[token] = index++;
    }

    private double[] EncodeOneHot(string[] sample) //REVISAR, que es vec, Nombre, muchos niveles
    {
        double[] vec = new double[_vocabulary.Count];
        foreach (string token in sample)
        {
            if (_vocabulary.ContainsKey(token))
            {
                int idx = _vocabulary[token];
                vec[idx] = 1.0;
            }
        }
        return vec;
    }
    
    private (bool, string?) TrainKNNClassifier(List<Sample> trainingDataset, int? vectorsNumber, string encoderType) //Manejo de errores  //Clase, Factory de Encoders
    {
        // Validate neighbor count (hyperparameter k)
        if (vectorsNumber == null)
            return (false, "number of neighbors must be specified");
        if (vectorsNumber <= 0)
            return (false, "number of neighbors must be positive");
        if (trainingDataset.Count < vectorsNumber)
            return (false, "number of neighbors must be smaller than the number of training samples");
        
        
        foreach (Sample sample in trainingDataset) //Clase, Factory de Encoders
        {
            double[] encodedFeatures;
            if(encoderType == "one-hot")
                encodedFeatures = EncodeOneHot(sample.Features);
            else if (encoderType == "label")
                encodedFeatures = EncodeLabel(sample.Features);
            else
                return (false, "Invalid encoder type");
            // We expect to add new encoders in the future
            
            _trainingData.Add(new Tuple<double[], string>(encodedFeatures, sample.Label));
        }

        return (true, null);
    }
    
    
    public (bool, string) PredictKNN(string[] features, string encoderType, int? k) //Factorie de Encoders
    {
        double[] encodedFeatures;
        if(encoderType == "one-hot")
            encodedFeatures = EncodeOneHot(features);
        else if (encoderType == "label")
            encodedFeatures = EncodeLabel(features);
        else
            return (false, "Invalid encoder type");
        // We expect to add new encoders in the future
        
        var notOrderedDistances =
            _trainingData.Select(t => new { Distance = Euclides(encodedFeatures, t.Item1), Label = t.Item2 });
        var orderedDistances = notOrderedDistances.OrderBy(t => t.Distance);
        var nearedDistances = orderedDistances.Take((int)k!);

        return (true, nearedDistances.GroupBy(t => t.Label).OrderByDescending(g => g.Count()).First().Key);
    }

    public double Euclides(double[] features1, double[] features2)
    {
        double sum = 0;
        for (int i = 0; i < features1.Length; i++)
            sum += Math.Pow(features1[i] - features2[i], 2);
        return Math.Sqrt(sum);
    }

    // LA FUNCION DE ABAJO TIENE MUCHO QUE ARREGLAR
    private (bool, string?) TrainLogisticRegressionClassifier(List<Sample> trainingDataset, int? epochs, double? learningRate, string encoderType) //MANEJO ERRORES //CLASE Obtain distinct Dataset Labels, Factories Encoders
    {
        //Validate hyperparameters
        if (learningRate == null)
            return (false, "learning rate must be specified");
        if (learningRate < 0)
            return (false, "learning rate can not be negative");
        if (epochs == null)
            return (false, "number of epochs must be specified");
        if (epochs < 0)
            return (false, "epochs can not be negative");
        
        // The actual training logic would go here
        
        // Get distinct labels from dataset //CLASE Obtain distinct Dataset Labels
        HashSet<string> hashSet = new();
        foreach (Sample sample in trainingDataset)
            hashSet.Add(sample.Label);
        string[] labels = hashSet.ToArray();
        
        // Encode first sample to determine feature count
        double[] encodedFeatures;
        if(encoderType == "one-hot")
            encodedFeatures = EncodeOneHot(trainingDataset[0].Features);
        else if (encoderType == "label")
            encodedFeatures = EncodeLabel(trainingDataset[0].Features);
        else
            return (false, "Invalid encoder type");
        // We expect to add new encoders in the future
        int featureCount = encodedFeatures.Length;
        
        // Initialize weights
        _weightsByLabel.Clear();
        foreach (string label in labels)
            _weightsByLabel[label] = new double[featureCount + 1];
        
        // Train using One-vs-Rest strategy //CLASE para encargarse
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int j = 0; j < trainingDataset.Count; j++)
            {
                Sample sample = trainingDataset[j];
                double[] encoded;
                if(encoderType == "one-hot")
                    encoded = EncodeOneHot(sample.Features);
                else if (encoderType == "label")
                    encoded = EncodeLabel(sample.Features);
                else
                    return (false, "Invalid encoder type");
                
                
                double[] inputVector = BuildInputVector(encoded);
                
                // Update weights for each label in sample
                foreach (string label in labels)
                {
                    int target = sample.Label == label ? 1 : 0;
                    double prediction = PredictWithSigmoid(_weightsByLabel[label], inputVector);
                    
                    // Update Weights and bias
                    for (int i = 0; i < _weightsByLabel[label].Length; i++)
                        _weightsByLabel[label][i] += (double)learningRate * (target - prediction) * inputVector[i];
                }
            }
        }
        return (true, null);
    }

    public double[] BuildInputVector(double[] encoded)
    {
        double[] inputVectorWithBias = new double[encoded.Length + 1];
        inputVectorWithBias[0] = 1;
        for (int i = 0; i < encoded.Length; i++)
            inputVectorWithBias[i + 1] = encoded[i];
        return inputVectorWithBias;
    }

    public double PredictWithSigmoid(double[] weights, double[] inputVector) //Considerar Clase Predictor
    {
        //Compute Dot Product
        double dotProduct = 0;
        for (int i = 0; i < weights.Length; i++)
            dotProduct += weights[i] * inputVector[i];
                    
        // Compute Prediction with Sigmoid Function
        return 1.0 / (1.0 + Math.Exp(-dotProduct));
    }

    private (bool, string) PredictLogisticRegression(string[] features, string encoderType) //Considerar Clase Predictor
    {
        double[] encoded;
        if(encoderType == "one-hot")
            encoded = EncodeOneHot(features);
        else if (encoderType == "label")
            encoded = EncodeLabel(features);
        else
            return (false, "Invalid encoder type");
        // We expect to add new encoders in the future
        
        var inputVector = BuildInputVector(encoded);
        // Predict the label with the highest score
        string bestLabel = null!;
        double bestScore = double.NegativeInfinity;

        foreach (var kvp in _weightsByLabel)
        {
            double score = PredictWithSigmoid(kvp.Value, inputVector);
            
            // Select Best Label
            if (score > bestScore)
            {
                bestScore = score;
                bestLabel = kvp.Key;
            }
        }

        return (true, bestLabel);
    }
    
    
    //Metodos mios
}