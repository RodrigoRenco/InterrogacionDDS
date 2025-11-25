namespace Classifier;

public class ClassificationResult
{
    public double? Score;
    public string? ErrorMessage;

    public void ProcessException(string message)
    {
        ErrorMessage = message;
        Score = null;
    }
}