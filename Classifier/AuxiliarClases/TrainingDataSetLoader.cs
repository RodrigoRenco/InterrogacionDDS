using Classifier.ClassificationExceptions;
using System.Text.Json;
namespace Classifier.AuxiliarClases;

public class TrainingDataSetLoader()
{
    private readonly string _trainingDatasetDir = "data/datasets";
    public List<Sample> trainDataset;
    
    public List<Sample> ObtainDataset(string DataSetDir)
    {
        string path = Path.Combine(_trainingDatasetDir, $"{DataSetDir}-train.json");
        if (!File.Exists(path))
        {
            throw new GenericException($"Dataset file not found: {DataSetDir}-train");
        }

        trainDataset = JsonSerializer.Deserialize<List<Sample>>(File.ReadAllText(path))!;
        if (trainDataset.Count == 0)
        {
            throw new GenericException($"Dataset {DataSetDir}-train is empty");
        }

        return trainDataset;
    }
    
    
}