namespace Classifier.AuxiliarClases;

public class ObtainAllLabelsDataSet
{
    public char[] GetAllLables(List<Sample> trainingDataset)
    {
        HashSet<char> lables = new();
        foreach (Sample sample in trainingDataset)
        foreach (char lable in sample.Label)
            lables.Add(lable);
        char[] allLabels = lables.ToArray();
        return allLabels;
    }
}