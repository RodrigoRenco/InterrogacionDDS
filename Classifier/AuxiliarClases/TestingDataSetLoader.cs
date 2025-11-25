using Classifier.ClassificationExceptions;
using System.Text.Json;
namespace Classifier.AuxiliarClases;

public class TestingDataSetLoader
{
    private readonly string _testingDatasetDir = "data/datasets";
    public List<Sample> testDataset;
    public List<Sample> ObtainDataset(string DataSetDir)
    {
        string path = Path.Combine(_testingDatasetDir, $"{DataSetDir}-test.json");
        if (!File.Exists(path))
        {
            throw new GenericException($"Dataset file not found: {DataSetDir}-test");
        }

        testDataset = JsonSerializer.Deserialize<List<Sample>>(File.ReadAllText(path))!;
        if (testDataset.Count == 0)
        {
            throw new GenericException($"Dataset {DataSetDir}-test is empty");
        }

        return testDataset;
    }
    
}