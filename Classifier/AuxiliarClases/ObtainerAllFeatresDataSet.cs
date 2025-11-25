namespace Classifier.AuxiliarClases;

public class ObtainerAllFeatresDataSet
{

    public string[] GetAllFeatures(List<Sample> trainingDataset)
    {
        HashSet<string> features = new();
        foreach (Sample sample in trainingDataset)
        foreach (string feature in sample.Features)
            features.Add(feature);
        string[] allTokens = features.ToArray();
        return allTokens;
    }
    
}