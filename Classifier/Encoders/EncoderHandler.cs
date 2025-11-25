using System.Text.Json;
using Classifier.AuxiliarClases;
using Classifier.ClassificationExceptions;
using Classifier.Encoders;
namespace Classifier.Encoders;

public class EncoderHandler
{
    public readonly Dictionary<string, int> _vocabulary = new();
    public readonly ObtainerAllFeatresDataSet ObtainerAllFeatresDataSet;

    public EncoderHandler()
    {
        ObtainerAllFeatresDataSet = new ObtainerAllFeatresDataSet();
    }
    
    private void MakeFitEncoder(List<Sample> trainingDataset) //ESTA FEO, creo que sacable
    {
        _vocabulary.Clear();

        string[] allTokens = ObtainerAllFeatresDataSet.GetAllFeatures(trainingDataset);
        
        int index = 0;
        foreach (string token in allTokens)
            _vocabulary[token] = index++;
    }
    
    
    
    
    private double[] EncodeLabel(string[] sample) //REVISAR, que es vec, Nombre, muchos niveles
    {
        double[] vec = new double[sample.Length];
        for (int i = 0; i < sample.Length; i++)
        {
            string token = sample[i];
            if (_vocabulary.ContainsKey(token))
            {
                int idx = _vocabulary[token];
                vec[i] = idx;
            }
        }
        return vec;
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
}